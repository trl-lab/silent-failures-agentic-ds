import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# Load train and test data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Set target and feature column lists
target_col = "outcome"
y_raw = train_df[target_col]

# Handle target mapping robustly for numeric or string classes
if y_raw.dtype == "object" or isinstance(y_raw.iloc[0], str):
    unique_vals = sorted(y_raw.unique())
    mapping = {val: i for i, val in enumerate(unique_vals)}
    y = y_raw.map(mapping)
else:
    y = y_raw.astype(int)

# Identify numerical parameter feature columns
feature_cols = [c for c in train_df.columns if c != target_col and c.lower() != "id"]

X = train_df[feature_cols].copy()
X_test = test_df[feature_cols].copy()


# Domain-informed Feature Engineering
def extract_features(df):
    feats = df.copy()

    # Aggregated parameters statistics across physical parameters
    feats["param_mean"] = feats[feature_cols].mean(axis=1)
    feats["param_std"] = feats[feature_cols].std(axis=1)
    feats["param_min"] = feats[feature_cols].min(axis=1)
    feats["param_max"] = feats[feature_cols].max(axis=1)
    feats["param_sum"] = feats[feature_cols].sum(axis=1)
    feats["param_l2"] = np.sqrt((feats[feature_cols] ** 2).sum(axis=1))
    feats["param_skew"] = feats[feature_cols].skew(axis=1)

    # Distance to boundaries (0 and 1) since parameters are bounded [0, 1]
    boundary_dist = np.minimum(feats[feature_cols], 1.0 - feats[feature_cols])
    feats["min_boundary_dist"] = boundary_dist.min(axis=1)
    feats["mean_boundary_dist"] = boundary_dist.mean(axis=1)

    return feats


X_fe = extract_features(X)
X_test_fe = extract_features(X_test)

# Setup 5-Fold Stratified Cross Validation with Multi-Model Ensemble
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_fe, y)):
    X_tr, y_tr = X_fe.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X_fe.iloc[val_idx], y.iloc[val_idx]

    # Model 1: XGBoost
    model_xgb = XGBClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        eval_metric="auc",
        early_stopping_rounds=50,
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    # Model 2: LightGBM
    model_lgb = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbosity=-1,
    )
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # Model 3: Extra Trees Classifier
    model_et = ExtraTreesClassifier(
        n_estimators=300, max_depth=8, random_state=42 + fold, n_jobs=-1
    )
    model_et.fit(X_tr, y_tr)

    # Ensemble validation predictions
    val_xgb = model_xgb.predict_proba(X_va)[:, 1]
    val_lgb = model_lgb.predict_proba(X_va)[:, 1]
    val_et = model_et.predict_proba(X_va)[:, 1]

    val_blend = 0.45 * val_xgb + 0.45 * val_lgb + 0.10 * val_et
    oof_preds[val_idx] = val_blend

    # Ensemble test predictions
    test_xgb = model_xgb.predict_proba(X_test_fe)[:, 1]
    test_lgb = model_lgb.predict_proba(X_test_fe)[:, 1]
    test_et = model_et.predict_proba(X_test_fe)[:, 1]

    test_blend = 0.45 * test_xgb + 0.45 * test_lgb + 0.10 * test_et
    test_preds += test_blend / skf.n_splits

# Compute and display Out-Of-Fold CV ROC AUC metric
cv_auc = roc_auc_score(y, oof_preds)
print(f"Validation ROC AUC: {cv_auc:.5f}")

# Save test set predictions to submission.csv
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "outcome": test_preds})
submission.to_csv("./working/submission.csv", index=False)
