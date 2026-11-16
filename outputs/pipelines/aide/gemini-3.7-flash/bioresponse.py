import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)

# Load datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Activity"
feature_cols = [c for c in train_df.columns if c != target_col]

X = train_df[feature_cols].values.astype(np.float32)
y = train_df[target_col].values.astype(int)
X_test = test_df[feature_cols].values.astype(np.float32)

# Filter out zero-variance features
selector = VarianceThreshold(threshold=0.0)
X = selector.fit_transform(X)
X_test = selector.transform(X_test)

# Stratified 5-Fold Cross-Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

n_train = len(train_df)
n_test = len(test_df)

oof_xgb = np.zeros(n_train)
oof_lgb = np.zeros(n_train)
oof_cat = np.zeros(n_train)
oof_hgb = np.zeros(n_train)
oof_et = np.zeros(n_train)
oof_rf = np.zeros(n_train)

test_xgb = np.zeros(n_test)
test_lgb = np.zeros(n_test)
test_cat = np.zeros(n_test)
test_hgb = np.zeros(n_test)
test_et = np.zeros(n_test)
test_rf = np.zeros(n_test)

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.03,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 3,
    "random_state": SEED,
    "n_estimators": 2000,
    "n_jobs": -1,
}

lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "random_state": SEED,
    "n_estimators": 2000,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # 1. XGBoost
    model_xgb = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=50)
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_pred_xgb = model_xgb.predict_proba(X_val)[:, 1]
    oof_xgb[val_idx] = val_pred_xgb
    test_xgb += model_xgb.predict_proba(X_test)[:, 1] / n_splits

    # 2. LightGBM
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    val_pred_lgb = model_lgb.predict_proba(X_val)[:, 1]
    oof_lgb[val_idx] = val_pred_lgb
    test_lgb += model_lgb.predict_proba(X_test)[:, 1] / n_splits

    # 3. CatBoost
    model_cat = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        eval_metric="Logloss",
        random_seed=SEED + fold,
        early_stopping_rounds=50,
        verbose=False,
    )
    model_cat.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    val_pred_cat = model_cat.predict_proba(X_val)[:, 1]
    oof_cat[val_idx] = val_pred_cat
    test_cat += model_cat.predict_proba(X_test)[:, 1] / n_splits

    # 4. HistGradientBoosting
    model_hgb = HistGradientBoostingClassifier(
        learning_rate=0.03,
        max_iter=600,
        early_stopping=True,
        n_iter_no_change=25,
        random_state=SEED + fold,
    )
    model_hgb.fit(X_train, y_train)
    val_pred_hgb = model_hgb.predict_proba(X_val)[:, 1]
    oof_hgb[val_idx] = val_pred_hgb
    test_hgb += model_hgb.predict_proba(X_test)[:, 1] / n_splits

    # 5. ExtraTrees Classifier
    model_et = ExtraTreesClassifier(
        n_estimators=350,
        max_features="sqrt",
        min_samples_split=3,
        random_state=SEED + fold,
        n_jobs=-1,
    )
    model_et.fit(X_train, y_train)
    val_pred_et = model_et.predict_proba(X_val)[:, 1]
    oof_et[val_idx] = val_pred_et
    test_et += model_et.predict_proba(X_test)[:, 1] / n_splits

    # 6. RandomForest Classifier
    model_rf = RandomForestClassifier(
        n_estimators=350,
        max_features="sqrt",
        min_samples_split=3,
        random_state=SEED + fold,
        n_jobs=-1,
    )
    model_rf.fit(X_train, y_train)
    val_pred_rf = model_rf.predict_proba(X_val)[:, 1]
    oof_rf[val_idx] = val_pred_rf
    test_rf += model_rf.predict_proba(X_test)[:, 1] / n_splits

print(f"XGB  OOF Log Loss: {log_loss(y, oof_xgb):.5f}")
print(f"LGB  OOF Log Loss: {log_loss(y, oof_lgb):.5f}")
print(f"CAT  OOF Log Loss: {log_loss(y, oof_cat):.5f}")
print(f"HGB  OOF Log Loss: {log_loss(y, oof_hgb):.5f}")
print(f"ET   OOF Log Loss: {log_loss(y, oof_et):.5f}")
print(f"RF   OOF Log Loss: {log_loss(y, oof_rf):.5f}")

# Multi-model Out-of-Fold Matrix
oof_stack = np.column_stack([oof_xgb, oof_lgb, oof_cat, oof_hgb, oof_et, oof_rf])
test_stack = np.column_stack([test_xgb, test_lgb, test_cat, test_hgb, test_et, test_rf])
n_models = oof_stack.shape[1]


# SLSQP Constrained Optimization to find optimal blend weights
def loss_func(weights):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    blend = np.dot(oof_stack, weights)
    blend = np.clip(blend, 1e-15, 1.0 - 1e-15)
    return log_loss(y, blend)


init_weights = [1.0 / n_models] * n_models
bounds = [(0.0, 1.0)] * n_models
res = minimize(
    loss_func,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints={"type": "eq", "fun": lambda w: 1.0 - sum(w)},
)
optimal_weights = res.x / np.sum(res.x)
print(f"Optimal Blend Weights (XGB, LGB, CAT, HGB, ET, RF): {optimal_weights.round(4)}")

oof_ensemble = np.dot(oof_stack, optimal_weights)
cv_log_loss = log_loss(y, np.clip(oof_ensemble, 1e-15, 1.0 - 1e-15))
print(f"Overall 6-Model Ensemble OOF Log Loss: {cv_log_loss:.5f}")

# Final blended test predictions
test_ensemble = np.dot(test_stack, optimal_weights)
clipped_preds = np.clip(test_ensemble, 1e-15, 1.0 - 1e-15)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(n_test), "Activity": clipped_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Saved predictions to ./working/submission.csv")
