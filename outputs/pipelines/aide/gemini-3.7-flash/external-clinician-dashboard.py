import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Load data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Score"
drop_cols = [target_col]
feature_cols = [c for c in train_df.columns if c not in drop_cols]


# Clean feature names for tree model compatibility
def clean_col_name(col):
    return re.sub(r'[\[\]<>,"\']', "_", str(col))


rename_dict = {col: clean_col_name(col) for col in feature_cols}
train_df = train_df.rename(columns=rename_dict)
test_df = test_df.rename(columns=rename_dict)
feature_cols = [rename_dict[c] for c in feature_cols]

# Identify categorical and numerical columns
cat_cols = []
num_cols = []

for col in feature_cols:
    if train_df[col].dtype == "object" or train_df[col].dtype.name == "category":
        cat_cols.append(col)
    else:
        num_cols.append(col)

# Prepare datasets for each model family
# XGBoost & LightGBM: category dtype
X_xgb_lgb = train_df[feature_cols].copy()
X_test_xgb_lgb = test_df[feature_cols].copy()

for col in cat_cols:
    X_xgb_lgb[col] = X_xgb_lgb[col].astype("category")
    X_test_xgb_lgb[col] = pd.Categorical(
        X_test_xgb_lgb[col], categories=X_xgb_lgb[col].cat.categories
    )

# CatBoost: string dtype with NaN filled
X_cb = train_df[feature_cols].copy()
X_test_cb = test_df[feature_cols].copy()

for col in cat_cols:
    X_cb[col] = X_cb[col].astype(str).fillna("missing")
    X_test_cb[col] = X_test_cb[col].astype(str).fillna("missing")

y = train_df[target_col].values

# 5-fold cross-validation setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(train_df))
oof_lgb = np.zeros(len(train_df))
oof_cb = np.zeros(len(train_df))

test_xgb = np.zeros(len(test_df))
test_lgb = np.zeros(len(test_df))
test_cb = np.zeros(len(test_df))

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "enable_categorical": True,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_estimators": 1500,
}

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "random_state": 42,
    "n_estimators": 1500,
}

cb_params = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "learning_rate": 0.05,
    "depth": 6,
    "random_seed": 42,
    "iterations": 1500,
    "verbose": False,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(train_df, y)):
    print(f"--- Fold {fold + 1} ---")
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]

    # 1. XGBoost
    X_tr_xgb, X_v_xgb = X_xgb_lgb.iloc[train_idx], X_xgb_lgb.iloc[val_idx]
    model_xgb = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
    model_xgb.fit(
        X_tr_xgb, y_train_fold, eval_set=[(X_v_xgb, y_val_fold)], verbose=False
    )
    oof_xgb[val_idx] = model_xgb.predict(X_v_xgb)
    test_xgb += model_xgb.predict(X_test_xgb_lgb) / n_splits

    # 2. LightGBM
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr_xgb,
        y_train_fold,
        eval_set=[(X_v_xgb, y_val_fold)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_lgb[val_idx] = model_lgb.predict(X_v_xgb)
    test_lgb += model_lgb.predict(X_test_xgb_lgb) / n_splits

    # 3. CatBoost
    X_tr_cb, X_v_cb = X_cb.iloc[train_idx], X_cb.iloc[val_idx]
    model_cb = CatBoostRegressor(**cb_params, early_stopping_rounds=50)
    model_cb.fit(
        X_tr_cb,
        y_train_fold,
        eval_set=(X_v_cb, y_val_fold),
        cat_features=cat_cols,
        verbose=False,
    )
    oof_cb[val_idx] = model_cb.predict(X_v_cb)
    test_cb += model_cb.predict(X_test_cb) / n_splits

rmse_xgb = np.sqrt(mean_squared_error(y, oof_xgb))
rmse_lgb = np.sqrt(mean_squared_error(y, oof_lgb))
rmse_cb = np.sqrt(mean_squared_error(y, oof_cb))

print(f"OOF RMSE - XGBoost: {rmse_xgb:.4f}")
print(f"OOF RMSE - LightGBM: {rmse_lgb:.4f}")
print(f"OOF RMSE - CatBoost: {rmse_cb:.4f}")


# Optimize blending weights
def loss_func(weights):
    w1, w2, w3 = weights
    pred = w1 * oof_xgb + w2 * oof_lgb + w3 * oof_cb
    return np.sqrt(mean_squared_error(y, pred))


bounds = [(0, 1), (0, 1), (0, 1)]
constraints = {"type": "eq", "fun": lambda w: sum(w) - 1.0}
init_weights = [1 / 3, 1 / 3, 1 / 3]

opt_res = minimize(
    loss_func, init_weights, method="SLSQP", bounds=bounds, constraints=constraints
)
best_weights = opt_res.x
print(
    f"Optimal Ensemble Weights: XGB={best_weights[0]:.4f}, LGB={best_weights[1]:.4f}, CB={best_weights[2]:.4f}"
)

oof_ensemble = (
    best_weights[0] * oof_xgb + best_weights[1] * oof_lgb + best_weights[2] * oof_cb
)
ensemble_rmse = np.sqrt(mean_squared_error(y, oof_ensemble))
print(f"Overall Ensemble OOF RMSE: {ensemble_rmse:.4f}")

# Generate test predictions
test_preds = (
    best_weights[0] * test_xgb + best_weights[1] * test_lgb + best_weights[2] * test_cb
)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_df)), "Score": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
