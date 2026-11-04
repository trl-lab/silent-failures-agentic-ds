import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Ensure working directory exists
os.makedirs("./working", exist_ok=True)

# 1. Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# 2. Process target column
target_col = "serious_ae"
y_raw = train[target_col].copy()

if (
    y_raw.dtype == bool
    or y_raw.dtype == object
    or str(y_raw.dtype).startswith("string")
):
    unique_vals = y_raw.dropna().unique()
    mapping = {}
    for val in unique_vals:
        s_val = str(val).lower().strip()
        if s_val in ["1", "true", "t", "y", "yes", "1.0"]:
            mapping[val] = 1
        elif s_val in ["0", "false", "f", "n", "no", "0.0"]:
            mapping[val] = 0
    y = y_raw.map(mapping).fillna(0).astype(int)
else:
    y = y_raw.astype(int)

# Separate features
X = train.drop(columns=[target_col]).copy()
X_test = test.copy()

# Remove ID-like columns if present in feature set
id_cols = [c for c in X.columns if c.lower() in ["id", "row_id", "index"]]
for col in id_cols:
    if col in X.columns:
        X = X.drop(columns=[col])
    if col in X_test.columns:
        X_test = X_test.drop(columns=[col])

# 3. Preprocessing: Date parsing and Categorical encoding
all_cols = list(X.columns)
date_cols = []
cat_cols = []

for col in all_cols:
    if X[col].dtype == object or str(X[col].dtype).startswith("string"):
        non_nulls = X[col].dropna()
        if len(non_nulls) > 0:
            try:
                date_sample = pd.to_datetime(non_nulls.iloc[:100], errors="coerce")
                if date_sample.notna().sum() > 70:
                    date_cols.append(col)
                    continue
            except Exception:
                pass
        cat_cols.append(col)
    elif str(X[col].dtype) == "category":
        cat_cols.append(col)

# Extract datetime features
for col in date_cols:
    X_dt = pd.to_datetime(X[col], errors="coerce")
    X_test_dt = pd.to_datetime(X_test[col], errors="coerce")

    X[f"{col}_year"] = X_dt.dt.year
    X[f"{col}_month"] = X_dt.dt.month
    X[f"{col}_day"] = X_dt.dt.day
    X[f"{col}_dayofweek"] = X_dt.dt.dayofweek

    X_test[f"{col}_year"] = X_test_dt.dt.year
    X_test[f"{col}_month"] = X_test_dt.dt.month
    X_test[f"{col}_day"] = X_test_dt.dt.day
    X_test[f"{col}_dayofweek"] = X_test_dt.dt.dayofweek

    X = X.drop(columns=[col])
    X_test = X_test.drop(columns=[col])

# Standardize categorical columns across train and test
for col in cat_cols:
    categories = pd.concat([X[col], X_test[col]], axis=0).dropna().unique()
    X[col] = pd.Categorical(X[col], categories=categories)
    X_test[col] = pd.Categorical(X_test[col], categories=categories)

# 4. Stratified K-Fold Training with LightGBM
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

    trn_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
    val_data = lgb.Dataset(
        X_va, label=y_va, reference=trn_data, categorical_feature=cat_cols
    )

    model = lgb.train(
        params,
        trn_data,
        num_boost_round=1000,
        valid_sets=[trn_data, val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    oof_preds[val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    test_preds += (
        model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits
    )

# 5. Evaluate overall OOF ROC AUC score
cv_auc = roc_auc_score(y, oof_preds)
print(f"5-Fold CV ROC AUC Score: {cv_auc:.5f}")

# 6. Save test predictions to submission.csv
submission = pd.DataFrame({"id": np.arange(len(test)), "serious_ae": test_preds})
submission.to_csv("./working/submission.csv", index=False)
