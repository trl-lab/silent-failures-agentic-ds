import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "decision_code"
y = train_df[target_col].copy()

# Ensure binary integer target
if y.dtype == "object" or str(y.dtype) == "category":
    le_target = LabelEncoder()
    y = le_target.fit_transform(y.astype(str))
else:
    unique_vals = np.sort(y.unique())
    if not np.array_equal(unique_vals, [0, 1]):
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)

train_df = train_df.drop(columns=[target_col], errors="ignore")
test_df = test_df.drop(columns=[target_col], errors="ignore")

# Drop explicit 'id' columns if present
for col in ["id", "ID", "Id"]:
    if col in train_df.columns:
        train_df = train_df.drop(columns=[col])
    if col in test_df.columns:
        test_df = test_df.drop(columns=[col])

n_train = len(train_df)
df = pd.concat([train_df, test_df], ignore_index=True)

# 2. Preprocessing & Feature Engineering
date_cols = []
text_cols = []
cat_cols = []
num_cols = []

for col in df.columns:
    if "date" in col.lower() or df[col].dtype == "datetime64[ns]":
        date_cols.append(col)
    elif df[col].dtype in ["object", "string", "category"]:
        # Check average text length
        sample_str = df[col].dropna().astype(str)
        if len(sample_str) > 0 and sample_str.str.len().mean() > 30:
            text_cols.append(col)
        else:
            cat_cols.append(col)
    else:
        num_cols.append(col)

# Handle Date Columns
parsed_date_cols = []
for col in date_cols:
    dt_series = pd.to_datetime(df[col], errors="coerce")
    if dt_series.notnull().sum() > 0:
        df[f"{col}_year"] = dt_series.dt.year
        df[f"{col}_month"] = dt_series.dt.month
        df[f"{col}_day"] = dt_series.dt.day
        df[f"{col}_dayofweek"] = dt_series.dt.dayofweek
        df[f"{col}_dayofyear"] = dt_series.dt.dayofyear
        df[f"{col}_timestamp"] = dt_series.astype("int64") // 10**9
        parsed_date_cols.append((col, dt_series))

# Pairwise date differences
for i in range(len(parsed_date_cols)):
    for j in range(i + 1, len(parsed_date_cols)):
        col1, dt1 = parsed_date_cols[i]
        col2, dt2 = parsed_date_cols[j]
        df[f"{col1}_{col2}_diff_days"] = (dt1 - dt2).dt.days

df = df.drop(columns=date_cols, errors="ignore")

# Handle Text Columns with TF-IDF + SVD
svd_dfs = []
for col in text_cols:
    text_data = df[col].fillna("").astype(str)
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    tfidf_mat = tfidf.fit_transform(text_data)
    n_comp = min(10, tfidf_mat.shape[1] - 1) if tfidf_mat.shape[1] > 1 else 1
    if n_comp > 0:
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        svd_mat = svd.fit_transform(tfidf_mat)
        svd_cols = [f"{col}_svd_{k}" for k in range(n_comp)]
        svd_df = pd.DataFrame(svd_mat, columns=svd_cols)
        svd_dfs.append(svd_df)

df = df.drop(columns=text_cols, errors="ignore")
if svd_dfs:
    df = pd.concat([df] + svd_dfs, axis=1)

# Handle Categorical Columns
for col in cat_cols:
    freq = df[col].value_counts().to_dict()
    df[f"{col}_freq"] = df[col].map(freq)

    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].fillna("Missing").astype(str))

# Convert all column types to float32/int32 to ensure LightGBM compatibility
for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].fillna("Missing").astype(str))
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-999).astype(np.float32)

# Separate Train and Test
X = df.iloc[:n_train].reset_index(drop=True)
X_test = df.iloc[n_train:].reset_index(drop=True)

# 3. Stratified 5-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    trn_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=trn_data)

    model = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds += (
        model.predict(X_test, num_iteration=model.best_iteration) / skf.n_splits
    )

# 4. Evaluation Metric
cv_auc = roc_auc_score(y, oof_preds)
print(f"5-Fold CV ROC AUC Score: {cv_auc:.5f}")

# 5. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(X_test)), "decision_code": test_preds})
submission.to_csv("./working/submission.csv", index=False)
