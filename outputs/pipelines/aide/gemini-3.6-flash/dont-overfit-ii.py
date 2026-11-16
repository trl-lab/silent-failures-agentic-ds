import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def add_row_statistics(X_mat):
    """Computes row-wise summary statistics per sample without data leakage."""
    mean_val = np.mean(X_mat, axis=1, keepdims=True)
    std_val = np.std(X_mat, axis=1, keepdims=True)
    min_val = np.min(X_mat, axis=1, keepdims=True)
    max_val = np.max(X_mat, axis=1, keepdims=True)
    median_val = np.median(X_mat, axis=1, keepdims=True)
    sum_val = np.sum(X_mat, axis=1, keepdims=True)
    skew_val = np.mean((X_mat - mean_val) ** 3, axis=1, keepdims=True) / (
        std_val**3 + 1e-8
    )

    return np.hstack(
        [
            X_mat,
            mean_val,
            std_val,
            min_val,
            max_val,
            median_val,
            sum_val,
            skew_val,
        ]
    )


# Load input data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

target_col = "target"
feature_cols = [c for c in train.columns if c not in ["id", target_col]]

X_raw = train[feature_cols].values
y = train[target_col].values
X_test_raw = test[feature_cols].values

# Compute row-wise aggregate features
X = add_row_statistics(X_raw)
X_test = add_row_statistics(X_test_raw)

n_seeds = 10
n_splits = 5
c_grid = np.logspace(-3, 1, 100)

oof_preds_matrix = np.zeros((n_seeds, len(train)))
test_preds = np.zeros(len(test))

for seed_idx, seed in enumerate(range(42, 42 + n_seeds)):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegressionCV(
            Cs=c_grid,
            cv=5,
            penalty="l1",
            solver="liblinear",
            scoring="roc_auc",
            random_state=seed,
            max_iter=1000,
        )
        model.fit(X_train_scaled, y_train)

        oof_preds_matrix[seed_idx, val_idx] = model.predict_proba(X_val_scaled)[:, 1]
        test_preds += model.predict_proba(X_test_scaled)[:, 1] / (n_seeds * n_splits)

# Average OOF predictions across all seeds
oof_preds = oof_preds_matrix.mean(axis=0)

# Print Out-Of-Fold Validation Score
cv_auc = roc_auc_score(y, oof_preds)
print(f"10-Seed 5-Fold CV ROC AUC Score: {cv_auc:.5f}")

# Save Submission File
os.makedirs("./working", exist_ok=True)
if "id" in test.columns:
    test_ids = test["id"].astype(int)
else:
    test_ids = np.arange(len(test))

submission = pd.DataFrame({"id": test_ids, "target": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
