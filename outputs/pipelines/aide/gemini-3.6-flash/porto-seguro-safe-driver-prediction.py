import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# 1. Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

if "id" in test.columns:
    test_id = test["id"].values
else:
    test_id = np.arange(len(test))

y = train["target"].values

# 2. Feature Selection & Baseline Engineering
drop_cols = ["id", "target"]
features = [c for c in train.columns if c not in drop_cols]
features = [c for c in features if not c.startswith("ps_calc_")]

df_all = pd.concat([train[features], test[features]], axis=0).reset_index(drop=True)

# Missing count
df_all["missing_count"] = (df_all == -1).sum(axis=1)

# Frequency encoding for categorical columns
cat_cols = [c for c in features if c.endswith("_cat")]
for col in cat_cols:
    freq = df_all[col].value_counts(normalize=True).to_dict()
    df_all[f"{col}_freq"] = df_all[col].map(freq)

# Domain interaction features
df_all["car_13_15"] = df_all["ps_car_13"] * df_all["ps_car_15"]
df_all["reg_01_02"] = df_all["ps_reg_01"] * df_all["ps_reg_02"]
df_all["car_12_13"] = df_all["ps_car_12"] * df_all["ps_car_13"]
df_all["car_13_div_15"] = df_all["ps_car_13"] / (df_all["ps_car_15"] + 0.001)
df_all["car_13_sq"] = df_all["ps_car_13"] ** 2

# Separate train and test
X = df_all.iloc[: len(train)].copy()
X_test = df_all.iloc[len(train) :].copy()


def eval_gini(y_true, y_pred):
    return 2 * roc_auc_score(y_true, y_pred) - 1


# 3. Stratified K-Fold Cross Validation with Multi-Model Ensemble
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(train))
oof_lgb = np.zeros(len(train))
oof_cb = np.zeros(len(train))

test_xgb = np.zeros(len(test))
test_lgb = np.zeros(len(test))
test_cb = np.zeros(len(test))

xgb_params = {
    "n_estimators": 1500,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 6,
    "tree_method": "hist",
    "eval_metric": "auc",
    "early_stopping_rounds": 50,
    "random_state": 42,
    "n_jobs": -1,
}

lgb_params = {
    "n_estimators": 1500,
    "max_depth": 5,
    "num_leaves": 31,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 50,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

cb_params = {
    "iterations": 1500,
    "depth": 6,
    "learning_rate": 0.03,
    "eval_metric": "AUC",
    "random_seed": 42,
    "early_stopping_rounds": 50,
    "thread_count": -1,
    "verbose": 0,
}

smoothing_weight = 10.0

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr_orig, y_tr = X.iloc[train_idx], y[train_idx]
    X_va_orig, y_va = X.iloc[val_idx], y[val_idx]

    X_tr_fold = X_tr_orig.copy()
    X_va_fold = X_va_orig.copy()
    X_te_fold = X_test.copy()

    global_mean = y_tr.mean()

    # Out-of-fold target encoding for categoricals
    for col in cat_cols:
        grouped = pd.Series(y_tr).groupby(X_tr_orig[col].values)
        counts = grouped.count()
        means = grouped.mean()

        smooth = (counts * means + smoothing_weight * global_mean) / (
            counts + smoothing_weight
        )
        te_map = smooth.to_dict()

        te_col_name = f"{col}_te"
        X_tr_fold[te_col_name] = X_tr_fold[col].map(te_map).fillna(global_mean).values
        X_va_fold[te_col_name] = X_va_fold[col].map(te_map).fillna(global_mean).values
        X_te_fold[te_col_name] = X_te_fold[col].map(te_map).fillna(global_mean).values

    # Replace -1 missing indicators with NaN for native GBDT handling
    X_tr_fold = X_tr_fold.replace(-1, np.nan)
    X_va_fold = X_va_fold.replace(-1, np.nan)
    X_te_fold = X_te_fold.replace(-1, np.nan)

    # 1. XGBoost
    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(X_tr_fold, y_tr, eval_set=[(X_va_fold, y_va)], verbose=False)
    val_pred_xgb = model_xgb.predict_proba(X_va_fold)[:, 1]
    oof_xgb[val_idx] = val_pred_xgb
    test_xgb += model_xgb.predict_proba(X_te_fold)[:, 1] / skf.n_splits

    # 2. LightGBM
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(
        X_tr_fold,
        y_tr,
        eval_set=[(X_va_fold, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    val_pred_lgb = model_lgb.predict_proba(X_va_fold)[:, 1]
    oof_lgb[val_idx] = val_pred_lgb
    test_lgb += model_lgb.predict_proba(X_te_fold)[:, 1] / skf.n_splits

    # 3. CatBoost
    model_cb = CatBoostClassifier(**cb_params)
    model_cb.fit(X_tr_fold, y_tr, eval_set=(X_va_fold, y_va), verbose=False)
    val_pred_cb = model_cb.predict_proba(X_va_fold)[:, 1]
    oof_cb[val_idx] = val_pred_cb
    test_cb += model_cb.predict_proba(X_te_fold)[:, 1] / skf.n_splits

    print(
        f"Fold {fold + 1} Gini Scores -> XGB: {eval_gini(y_va, val_pred_xgb):.5f}, "
        f"LGB: {eval_gini(y_va, val_pred_lgb):.5f}, CB: {eval_gini(y_va, val_pred_cb):.5f}"
    )

# 4. Rank Transform Predictions
oof_xgb_rank = pd.Series(oof_xgb).rank(pct=True).values
oof_lgb_rank = pd.Series(oof_lgb).rank(pct=True).values
oof_cb_rank = pd.Series(oof_cb).rank(pct=True).values

test_xgb_rank = pd.Series(test_xgb).rank(pct=True).values
test_lgb_rank = pd.Series(test_lgb).rank(pct=True).values
test_cb_rank = pd.Series(test_cb).rank(pct=True).values


# 5. Optimize Blending Weights on OOF
def loss_func(weights):
    w1, w2, w3 = weights
    w_sum = w1 + w2 + w3
    if w_sum <= 0:
        return 1.0
    w1, w2, w3 = w1 / w_sum, w2 / w_sum, w3 / w_sum
    blend = w1 * oof_xgb_rank + w2 * oof_lgb_rank + w3 * oof_cb_rank
    return -eval_gini(y, blend)


res = minimize(
    loss_func,
    [0.5, 0.25, 0.25],
    bounds=[(0, 1), (0, 1), (0, 1)],
    method="Nelder-Mead",
)
best_w = res.x / np.sum(res.x)
print(f"Optimal Blend Weights [XGB, LGB, CB]: {best_w}")

oof_ensemble = (
    best_w[0] * oof_xgb_rank + best_w[1] * oof_lgb_rank + best_w[2] * oof_cb_rank
)
test_ensemble = (
    best_w[0] * test_xgb_rank + best_w[1] * test_lgb_rank + best_w[2] * test_cb_rank
)

overall_gini = eval_gini(y, oof_ensemble)
print(f"Overall OOF Ensemble Normalized Gini: {overall_gini:.5f}")

# 6. Save Submission File
os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame({"id": test_id, "target": test_ensemble})
sub.to_csv("./working/submission.csv", index=False)
