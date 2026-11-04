import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "antibody_truth"
y = train_df[target_col].copy()

# Ensure target is binary integer (0/1)
if y.dtype == bool or y.dtype == object:
    unique_vals = sorted(y.unique())
    mapping = {val: i for i, val in enumerate(unique_vals)}
    y = y.map(mapping).astype(int)
else:
    y = y.astype(int)

feature_cols = [c for c in train_df.columns if c != target_col]
X = train_df[feature_cols].copy()
X_test = test_df[feature_cols].copy()

# Identify categorical features and preprocess missing values
cat_cols = []
num_cols = []
for c in feature_cols:
    if X[c].dtype == "object" or X[c].dtype.name == "category":
        cat_cols.append(c)
        X[c] = X[c].fillna("missing").astype(str)
        X_test[c] = X_test[c].fillna("missing").astype(str)
    else:
        num_cols.append(c)
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce")

# For LightGBM categorical handling
X_lgb = X.copy()
X_test_lgb = X_test.copy()
for c in cat_cols:
    X_lgb[c] = X_lgb[c].astype("category")
    X_test_lgb[c] = X_test_lgb[c].astype("category")

# 2. 5-Fold Cross-Validation Setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds_lgb = np.zeros(len(X))
test_preds_lgb = np.zeros(len(X_test))

oof_preds_cb = np.zeros(len(X))
test_preds_cb = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_fold, y_train_fold = X_lgb.iloc[train_idx], y.iloc[train_idx]
    X_val_fold, y_val_fold = X_lgb.iloc[val_idx], y.iloc[val_idx]

    # LightGBM Model
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbosity=-1,
    )
    lgb_model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    oof_preds_lgb[val_idx] = lgb_model.predict_proba(X_val_fold)[:, 1]
    test_preds_lgb += lgb_model.predict_proba(X_test_lgb)[:, 1] / n_splits

    # CatBoost Model
    X_train_cb, X_val_cb = X.iloc[train_idx], X.iloc[val_idx]
    cb_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        cat_features=cat_cols if cat_cols else None,
        eval_metric="AUC",
        random_seed=42 + fold,
        verbose=False,
    )
    cb_model.fit(
        X_train_cb,
        y_train_fold,
        eval_set=(X_val_cb, y_val_fold),
        early_stopping_rounds=50,
        verbose=False,
    )

    oof_preds_cb[val_idx] = cb_model.predict_proba(X_val_cb)[:, 1]
    test_preds_cb += cb_model.predict_proba(X_test)[:, 1] / n_splits

# Ensemble predictions
oof_preds = 0.5 * oof_preds_lgb + 0.5 * oof_preds_cb
test_preds = 0.5 * test_preds_lgb + 0.5 * test_preds_cb

# 3. Validation Metric
cv_auc = roc_auc_score(y, oof_preds)
print(f"5-Fold Cross-Validation ROC-AUC: {cv_auc:.5f}")

# 4. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_df)), "antibody_truth": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
