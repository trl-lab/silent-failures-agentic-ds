import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

os.makedirs("./working", exist_ok=True)

# Separate target and features
target_col = "funding"
y = train_df[target_col].values
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

# Drop 'id' column if present in features
if "id" in X_train.columns:
    X_train = X_train.drop(columns=["id"])
if "id" in X_test.columns:
    X_test = X_test.drop(columns=["id"])

# Combine features for consistent preprocessing
df_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)
n_train = len(X_train)

# 2. Feature Preprocessing & Engineering
# Identify column types
cat_cols = []
num_cols = []
text_cols = []

for col in df_all.columns:
    if pd.api.types.is_datetime64_any_dtype(df_all[col]):
        df_all[f"{col}_year"] = df_all[col].dt.year
        df_all[f"{col}_month"] = df_all[col].dt.month
        df_all[f"{col}_day"] = df_all[col].dt.day
        df_all[f"{col}_dayofweek"] = df_all[col].dt.dayofweek
        df_all = df_all.drop(columns=[col])
    elif df_all[col].dtype == "object" or isinstance(
        df_all[col].dtype, pd.CategoricalDtype
    ):
        # Check if text or simple categorical
        sample_str = df_all[col].dropna().astype(str)
        if len(sample_str) > 0 and sample_str.str.len().mean() > 50:
            text_cols.append(col)
        else:
            cat_cols.append(col)
    else:
        num_cols.append(col)

# Text Feature Processing
text_features = []
for col in text_cols:
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    text_data = df_all[col].fillna("").astype(str)
    tfidf_mat = tfidf.fit_transform(text_data)
    svd = TruncatedSVD(
        n_components=min(10, tfidf_mat.shape[1] - 1 if tfidf_mat.shape[1] > 1 else 1),
        random_state=42,
    )
    svd_feats = svd.fit_transform(tfidf_mat)
    svd_cols = [f"{col}_svd_{i}" for i in range(svd_feats.shape[1])]
    svd_df = pd.DataFrame(svd_feats, columns=svd_cols)
    text_features.append(svd_df)
    df_all = df_all.drop(columns=[col])

if text_features:
    df_all = pd.concat([df_all] + text_features, axis=1)

# Categorical Feature Conversion
for col in cat_cols:
    df_all[col] = df_all[col].astype(str).astype("category")

# Frequency Encoding for high cardinality categories
for col in cat_cols:
    freq = df_all[col].value_counts()
    df_all[f"{col}_freq"] = df_all[col].map(freq)

# Split back to train and test
X_train_proc = df_all.iloc[:n_train].copy()
X_test_proc = df_all.iloc[n_train:].copy()

# 3. Model Training with 5-Fold Cross Validation
use_log = y.min() >= 0
y_train_target = np.log1p(y) if use_log else y

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(X_test_proc))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "random_state": 42,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_proc, y_train_target)):
    X_tr, y_tr = X_train_proc.iloc[train_idx], y_train_target[train_idx]
    X_va, y_va = X_train_proc.iloc[val_idx], y_train_target[val_idx]

    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)

    model = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    val_pred = model.predict(X_va)
    oof_preds[val_idx] = np.expm1(val_pred) if use_log else val_pred

    test_fold_pred = model.predict(X_test_proc)
    test_preds += (
        np.expm1(test_fold_pred) if use_log else test_fold_pred
    ) / kf.n_splits

# 4. Evaluation Metric
oof_preds = np.clip(oof_preds, a_min=0, a_max=None) if use_log else oof_preds
cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"Validation RMSE: {cv_rmse:.4f}")

# 5. Generate Submission File
submission = pd.DataFrame(
    {
        "id": np.arange(len(test_preds)),
        "funding": np.clip(test_preds, a_min=0, a_max=None) if use_log else test_preds,
    }
)
submission.to_csv("./working/submission.csv", index=False)
