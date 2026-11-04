import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

os.makedirs("./working", exist_ok=True)

train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

feature_cols = [c for c in train_df.columns if c not in ["id", "target"]]

for col in feature_cols:
    train_df[col] = train_df[col].fillna("MISSING").astype(str)
    test_df[col] = test_df[col].fillna("MISSING").astype(str)

ohe = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
X_train = ohe.fit_transform(train_df[feature_cols])
X_test = ohe.transform(test_df[feature_cols])
y_train = train_df["target"].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_va, y_va = X_train[val_idx], y_train[val_idx]

    model = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs", random_state=42)
    model.fit(X_tr, y_tr)

    oof_preds[val_idx] = model.predict_proba(X_va)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits

cv_auc = roc_auc_score(y_train, oof_preds)
print(f"5-Fold CV ROC AUC: {cv_auc:.5f}")

submission = pd.DataFrame({"id": test_df["id"], "target": test_preds})
submission.to_csv("./working/submission.csv", index=False)
