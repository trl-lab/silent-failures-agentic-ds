import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb

train_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/santander-customer-satisfaction/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/santander-customer-satisfaction/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

y = train_df["TARGET"].values
X = train_df.drop(columns=["TARGET"])

if "ID" in X.columns:
    X = X.drop(columns=["ID"])
if "id" in X.columns:
    X = X.drop(columns=["id"])

X_test = test_df.copy()
if "TARGET" in X_test.columns:
    X_test = X_test.drop(columns=["TARGET"])
if "ID" in X_test.columns:
    X_test = X_test.drop(columns=["ID"])
if "id" in X_test.columns:
    X_test = X_test.drop(columns=["id"])

constant_cols = [c for c in X.columns if X[c].nunique() <= 1]
X = X.drop(columns=constant_cols)
X_test = X_test.drop(columns=constant_cols, errors="ignore")

dup_cols = X.T[X.T.duplicated()].index
X = X.drop(columns=dup_cols)
X_test = X_test.drop(columns=dup_cols, errors="ignore")

X["num_zeros"] = (X == 0).sum(axis=1)
X_test["num_zeros"] = (X_test == 0).sum(axis=1)

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds_lgb = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    model_lgb = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    test_preds_lgb += model_lgb.predict_proba(X_test)[:, 1] / n_splits

    model_xgb = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42 + fold,
        n_jobs=-1,
        eval_metric="auc",
        early_stopping_rounds=50,
        tree_method="hist"
    )
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    test_preds_xgb += model_xgb.predict_proba(X_test)[:, 1] / n_splits

final_preds = 0.5 * test_preds_lgb + 0.5 * test_preds_xgb

submission = pd.DataFrame({
    "id": np.arange(len(final_preds)),
    "TARGET": final_preds
})

submission.to_csv("predictions.csv", index=False)