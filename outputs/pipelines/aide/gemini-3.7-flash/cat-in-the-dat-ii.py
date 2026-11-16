import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Load datasets
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "target"
id_col = "id"
features = [c for c in train_df.columns if c not in [id_col, target_col]]

# Fill missing values and convert all feature columns to strings for uniform categorical encoding
y = train_df[target_col].values
test_ids = test_df[id_col].values

X_train_df = train_df[features].copy()
X_test_df = test_df[features].copy()

for col in features:
    X_train_df[col] = X_train_df[col].astype(str).fillna("missing")
    X_test_df[col] = X_test_df[col].astype(str).fillna("missing")

# One-hot encode all features jointly (train + test vocabulary) using sparse matrices
ohe = OneHotEncoder(sparse_output=True, handle_unknown="ignore", dtype=np.float32)
all_data = pd.concat([X_train_df, X_test_df], axis=0, ignore_index=True)
ohe.fit(all_data)

X_train_sparse = ohe.transform(X_train_df)
X_test_sparse = ohe.transform(X_test_df)

del all_data, X_train_df, X_test_df

# 5-Fold Stratified Cross Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(y), dtype=np.float64)
test_preds = np.zeros(X_test_sparse.shape[0], dtype=np.float64)

print(
    f"Starting 5-Fold Stratified Cross-Validation on {X_train_sparse.shape[0]} samples with {X_train_sparse.shape[1]} sparse features..."
)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_sparse, y)):
    X_tr, y_tr = X_train_sparse[train_idx], y[train_idx]
    X_val, y_val = X_train_sparse[val_idx], y[val_idx]

    model = LogisticRegression(
        C=0.1, max_iter=1000, solver="lbfgs", random_state=42 + fold, n_jobs=-1
    )
    model.fit(X_tr, y_tr)

    val_pred = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_pred
    test_preds += model.predict_proba(X_test_sparse)[:, 1] / n_splits

    fold_auc = roc_auc_score(y_val, val_pred)
    print(f"Fold {fold + 1} ROC AUC: {fold_auc:.5f}")

overall_auc = roc_auc_score(y, oof_preds)
print(f"Overall OOF ROC AUC: {overall_auc:.5f}")

# Save submission file
os.makedirs("./working", exist_ok=True)
submission_path = "./working/submission.csv"
sub = pd.DataFrame({"id": test_ids, "target": test_preds})
sub.to_csv(submission_path, index=False)
print(f"Submission successfully saved to {submission_path} with shape {sub.shape}")
