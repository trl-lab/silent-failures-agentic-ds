import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

# Ensure working directory exists
os.makedirs("./working", exist_ok=True)

# Load train and test data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Separate features and target
target_col = "target"
id_col = "id"

y = train_df[target_col].values
train_ids = train_df[id_col].values
test_ids = test_df[id_col].values

features = [col for col in train_df.columns if col not in [id_col, target_col]]

# Convert all features to string for uniform categorical encoding
for col in features:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

# One-hot encode all features into sparse matrices
encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
X_train = encoder.fit_transform(train_df[features])
X_test = encoder.transform(test_df[features])

# 5-fold Stratified Cross-Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train[train_idx], y[train_idx]
    X_val, y_val = X_train[val_idx], y[val_idx]

    model = LogisticRegression(
        C=0.1, max_iter=1000, solver="lbfgs", random_state=42, n_jobs=-1
    )
    model.fit(X_tr, y_tr)

    val_pred = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_pred
    test_preds += model.predict_proba(X_test)[:, 1] / n_splits

    fold_auc = roc_auc_score(y_val, val_pred)
    print(f"Fold {fold + 1} ROC-AUC: {fold_auc:.5f}")

overall_auc = roc_auc_score(y, oof_preds)
print(f"Overall OOF ROC-AUC: {overall_auc:.5f}")

# Save test predictions to submission.csv
submission = pd.DataFrame({id_col: test_ids, target_col: test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully.")
