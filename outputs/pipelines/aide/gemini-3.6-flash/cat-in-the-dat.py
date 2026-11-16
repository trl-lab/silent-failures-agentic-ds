import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

# Load train and test data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Separate target and feature columns
y = train["target"].values
train_id = train["id"].values
test_id = test["id"].values

feature_cols = [c for c in train.columns if c not in ["id", "target"]]

# Combine feature sets to fit OneHotEncoder across all unique categorical levels
df_all = pd.concat([train[feature_cols], test[feature_cols]], axis=0)

encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
encoder.fit(df_all)

X_train = encoder.transform(train[feature_cols])
X_test = encoder.transform(test[feature_cols])

# Set up 5-Fold Stratified Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train[train_idx], y[train_idx]
    X_va, y_va = X_train[val_idx], y[val_idx]

    model = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")
    model.fit(X_tr, y_tr)

    oof_preds[val_idx] = model.predict_proba(X_va)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits

cv_auc = roc_auc_score(y, oof_preds)
print(f"5-Fold CV ROC AUC Score: {cv_auc:.5f}")

# Save test predictions to submission.csv
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": test_id, "target": test_preds})
submission.to_csv("./working/submission.csv", index=False)
