import os
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Load datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "num_countries_affected"

# Encode binary target
le = LabelEncoder()
y = le.fit_transform(train_df[target_col].astype(str))

train_len = len(train_df)
combined = pd.concat(
    [train_df.drop(columns=[target_col], errors="ignore"), test_df],
    ignore_index=True,
)

# Identify column types
drop_cols = []
text_cols = []
date_cols = []
cat_cols = []
num_cols = []

for col in combined.columns:
    if "id" in col.lower() and combined[col].nunique() == len(combined):
        drop_cols.append(col)
        continue

    if pd.api.types.is_datetime64_any_dtype(combined[col]):
        date_cols.append(col)
        continue
    elif combined[col].dtype == object:
        sample = combined[col].dropna().astype(str).head(50)
        is_date = False
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() > 0.8:
                is_date = True
        except Exception:
            pass

        if is_date:
            date_cols.append(col)
        else:
            avg_len = sample.str.len().mean() if len(sample) > 0 else 0
            if avg_len > 40 or combined[col].nunique() > 1000:
                text_cols.append(col)
            else:
                cat_cols.append(col)
    elif pd.api.types.is_numeric_dtype(combined[col]):
        num_cols.append(col)

# Date feature engineering
for col in date_cols:
    d_series = pd.to_datetime(combined[col], errors="coerce")
    combined[f"{col}_year"] = d_series.dt.year.fillna(-1).astype(int)
    combined[f"{col}_month"] = d_series.dt.month.fillna(-1).astype(int)
    combined[f"{col}_day"] = d_series.dt.day.fillna(-1).astype(int)
    combined[f"{col}_dayofweek"] = d_series.dt.dayofweek.fillna(-1).astype(int)
    combined[f"{col}_dayofyear"] = d_series.dt.dayofyear.fillna(-1).astype(int)
    drop_cols.append(col)

# Text feature engineering (TF-IDF + SVD)
for col in text_cols:
    combined[col] = combined[col].fillna("").astype(str)
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w+\b",
        min_df=2,
    )
    svd = TruncatedSVD(n_components=10, random_state=42)
    tfidf_mat = tfidf.fit_transform(combined[col])
    svd_mat = svd.fit_transform(tfidf_mat)
    for i in range(svd_mat.shape[1]):
        combined[f"{col}_svd_{i}"] = svd_mat[:, i]
    drop_cols.append(col)

# Categorical feature encoding
for col in cat_cols:
    combined[col] = combined[col].fillna("missing").astype(str)
    freq = combined[col].value_counts()
    combined[f"{col}_freq"] = combined[col].map(freq)
    combined[col] = combined[col].astype("category")

# Drop raw date and processed text columns
combined.drop(columns=drop_cols, inplace=True, errors="ignore")

X_train = combined.iloc[:train_len].copy()
X_test = combined.iloc[train_len:].copy()

# Stratified 5-Fold Cross Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(train_len)
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
    "random_state": 42,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1500,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

cv_roc_auc = roc_auc_score(y, oof_preds)
print(f"5-Fold CV Out-of-Fold ROC-AUC: {cv_roc_auc:.5f}")

# Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "num_countries_affected": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
