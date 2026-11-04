import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# 1. Load data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# 2. Process target
target_col = "status"
y_raw = train_df[target_col]

if y_raw.dtype == object or isinstance(y_raw.iloc[0], str):
    # Common status values in Kickstarter: 'successful' vs 'failed', 'canceled', etc.
    unique_vals = y_raw.dropna().unique()
    if "successful" in unique_vals:
        y = (y_raw == "successful").astype(int).values
    elif "success" in unique_vals:
        y = (y_raw == "success").astype(int).values
    else:
        # Default mapping for binary string
        val_map = {val: idx for idx, val in enumerate(sorted(unique_vals))}
        y = y_raw.map(val_map).astype(int).values
else:
    y = y_raw.astype(int).values

# Align features between train and test
feature_cols = [
    col for col in test_df.columns if col in train_df.columns and col != target_col
]
n_train = len(train_df)
combined = pd.concat(
    [train_df[feature_cols], test_df[feature_cols]], axis=0
).reset_index(drop=True)

# 3. Feature Engineering
# Datetime processing
date_cols = [
    c
    for c in combined.columns
    if any(k in c.lower() for k in ["date", "time", "created", "launched", "deadline"])
]
for col in date_cols:
    # Try parsing timestamps (int/float unix timestamps or strings)
    if pd.api.types.is_numeric_dtype(combined[col]):
        # Check if unix timestamp in seconds
        sample_val = (
            combined[col].dropna().iloc[0] if len(combined[col].dropna()) > 0 else 0
        )
        if sample_val > 1e8:
            combined[col] = pd.to_datetime(combined[col], unit="s", errors="coerce")
    else:
        combined[col] = pd.to_datetime(combined[col], errors="coerce")

# Durations if standard kickstarter timestamps exist
if "launched_at" in combined.columns and "deadline" in combined.columns:
    combined["duration_days"] = (
        combined["deadline"] - combined["launched_at"]
    ).dt.total_seconds() / 86400.0
if "created_at" in combined.columns and "launched_at" in combined.columns:
    combined["prep_time_days"] = (
        combined["launched_at"] - combined["created_at"]
    ).dt.total_seconds() / 86400.0

for col in date_cols:
    if pd.api.types.is_datetime64_any_dtype(combined[col]):
        combined[f"{col}_year"] = combined[col].dt.year
        combined[f"{col}_month"] = combined[col].dt.month
        combined[f"{col}_day"] = combined[col].dt.day
        combined[f"{col}_dayofweek"] = combined[col].dt.dayofweek
        combined[f"{col}_hour"] = combined[col].dt.hour
        combined = combined.drop(columns=[col])

# Text features
text_cols = [
    c for c in combined.columns if c.lower() in ["name", "blurb", "desc", "description"]
]
for col in text_cols:
    combined[col] = combined[col].fillna("").astype(str)
    combined[f"{col}_char_len"] = combined[col].apply(len)
    combined[f"{col}_word_count"] = combined[col].apply(lambda s: len(s.split()))

    # TF-IDF + SVD for text embeddings
    if combined[col].nunique() > 10:
        tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
        tfidf_mat = tfidf.fit_transform(combined[col])
        svd = TruncatedSVD(n_components=10, random_state=42)
        svd_mat = svd.fit_transform(tfidf_mat)
        for i in range(10):
            combined[f"{col}_svd_{i}"] = svd_mat[:, i]
    combined = combined.drop(columns=[col])

# Numeric and categorical handling
for col in combined.columns:
    if combined[col].dtype == object:
        combined[col] = combined[col].astype("category")
    elif str(combined[col].dtype) == "category":
        pass
    else:
        # Check numeric skewed distributions (like goal)
        if "goal" in col.lower() and pd.api.types.is_numeric_dtype(combined[col]):
            combined[f"{col}_log"] = np.log1p(np.maximum(0, combined[col]))

# Separate back into train and test
X_train = combined.iloc[:n_train].reset_index(drop=True)
X_test = combined.iloc[n_train:].reset_index(drop=True)

# 4. Cross-Validation & LightGBM Modeling
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(X_test))

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "random_state": 42,
    "verbose": -1,
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
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

cv_score = roc_auc_score(y, oof_preds)
print(f"5-Fold Cross-Validation ROC AUC: {cv_score:.6f}")

# 5. Create submission file
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "status": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
