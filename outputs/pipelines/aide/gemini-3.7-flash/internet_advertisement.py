import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Ensure working directory exists
os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify target and features
target_col = "class"
feature_cols = [col for col in train_df.columns if col != target_col]

# Process target
if train_df[target_col].dtype == object or str(train_df[target_col].dtype).startswith(
    "category"
):
    target_str = train_df[target_col].astype(str).str.strip().str.lower()
    if "ad" in target_str.values or "ad." in target_str.values:
        y = target_str.isin(["ad", "ad.", "1", "true", "yes"]).astype(int).values
    else:
        # Fallback to binary factorize
        y, _ = pd.factorize(target_str)
else:
    y = train_df[target_col].astype(int).values

# Clean and convert feature columns to float
X = pd.DataFrame(index=train_df.index)
X_test = pd.DataFrame(index=test_df.index)

for col in feature_cols:
    X[col] = pd.to_numeric(train_df[col], errors="coerce")
    X_test[col] = pd.to_numeric(test_df[col], errors="coerce")

# Model parameters
lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# 5-fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1500,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred
    test_preds += (
        model.predict(X_test, num_iteration=model.best_iteration) / skf.n_splits
    )

# Compute overall CV ROC AUC score
cv_auc = roc_auc_score(y, oof_preds)
print(f"5-Fold CV ROC AUC Score: {cv_auc:.6f}")

# Generate and save submission
submission = pd.DataFrame({"id": np.arange(len(test_df)), "class": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
