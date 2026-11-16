import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# Load train and test data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify target column and feature columns
target_col = "outcome"

# Encode target into binary (0: failure, 1: success)
if (
    train_df[target_col].dtype == object
    or str(train_df[target_col].dtype) == "category"
):
    unique_vals = train_df[target_col].dropna().unique()
    # Check if target values look like numeric strings or textual
    if set(unique_vals).issubset({"0", "1", 0, 1}):
        y_train = train_df[target_col].astype(int).values
    else:
        y_train = (
            train_df[target_col]
            .astype(str)
            .str.lower()
            .isin(["1", "success", "succeed", "true"])
            .astype(int)
            .values
        )
else:
    y_train = train_df[target_col].astype(int).values

# Filter out target and non-predictive batch columns if present
drop_cols = [target_col] + [
    c for c in ["study", "run", "batch", "id"] if c in train_df.columns
]
feature_cols = [c for c in train_df.columns if c not in drop_cols]

X_train = train_df[feature_cols].copy().values
X_test = test_df[feature_cols].copy().values

# Calculate scale_pos_weight for class imbalance
num_neg = np.sum(y_train == 0)
num_pos = np.sum(y_train == 1)
scale_pos_weight = num_neg / max(num_pos, 1)

# Stratified 5-Fold Cross-Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(y_train))
test_preds = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_va, y_va = X_train[val_idx], y_train[val_idx]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1.0,  # balanced sampling
        random_state=42 + fold,
        eval_metric="auc",
        early_stopping_rounds=30,
        n_jobs=-1,
    )

    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    oof_preds[val_idx] = model.predict_proba(X_va)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / n_splits

# Compute validation metric
cv_roc_auc = roc_auc_score(y_train, oof_preds)
print(f"Validation Out-of-Fold ROC-AUC: {cv_roc_auc:.5f}")

# Save test predictions to ./working/submission.csv
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "outcome": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
