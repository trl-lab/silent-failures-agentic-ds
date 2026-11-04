import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool

# Ensure output directory exists
os.makedirs("./working", exist_ok=True)

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "SupStruc"
drop_cols = [target_col]

# Identify feature columns
raw_feature_cols = [c for c in train_df.columns if c not in drop_cols]

X_train = train_df[raw_feature_cols].copy()
y_train = train_df[target_col].values
X_test = test_df[raw_feature_cols].copy()

# Handle datetime columns
for col in raw_feature_cols:
    if np.issubdtype(X_train[col].dtype, np.datetime64):
        X_train[f"{col}_year"] = X_train[col].dt.year
        X_train[f"{col}_month"] = X_train[col].dt.month
        X_train[f"{col}_day"] = X_train[col].dt.day
        X_test[f"{col}_year"] = X_test[col].dt.year
        X_test[f"{col}_month"] = X_test[col].dt.month
        X_test[f"{col}_day"] = X_test[col].dt.day
        X_train = X_train.drop(columns=[col])
        X_test = X_test.drop(columns=[col])

feature_cols = list(X_train.columns)

# Prepare datasets for CatBoost and LightGBM
cat_cols = []
X_train_cb = X_train.copy()
X_test_cb = X_test.copy()
X_train_lgb = X_train.copy()
X_test_lgb = X_test.copy()

for col in feature_cols:
    if (
        X_train[col].dtype == "object"
        or X_train[col].dtype.name == "category"
        or str(X_train[col].dtype) == "string"
    ):
        cat_cols.append(col)
        # CatBoost strings
        X_train_cb[col] = X_train[col].fillna("__MISSING__").astype(str)
        X_test_cb[col] = X_test[col].fillna("__MISSING__").astype(str)
        # LightGBM category type
        X_train_lgb[col] = X_train[col].fillna("__MISSING__").astype("category")
        X_test_lgb[col] = X_test[col].fillna("__MISSING__").astype("category")

# 2. 5-Fold Cross Validation Setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

cb_oof = np.zeros(len(train_df))
cb_test_preds = np.zeros(len(test_df))

lgb_oof = np.zeros(len(train_df))
lgb_test_preds = np.zeros(len(test_df))

cb_params = {
    "iterations": 2000,
    "learning_rate": 0.04,
    "depth": 6,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "verbose": 0,
    "early_stopping_rounds": 50,
}

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.04,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}

print("Starting 5-Fold Cross-Validation for CatBoost and LightGBM...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    # --- CatBoost ---
    X_tr_cb, X_val_cb = X_train_cb.iloc[train_idx], X_train_cb.iloc[val_idx]
    train_pool = Pool(X_tr_cb, y_tr, cat_features=cat_cols)
    val_pool = Pool(X_val_cb, y_val, cat_features=cat_cols)
    test_pool = Pool(X_test_cb, cat_features=cat_cols)

    cb_model = CatBoostRegressor(**cb_params)
    cb_model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    cb_val_pred = cb_model.predict(val_pool)
    cb_oof[val_idx] = cb_val_pred
    cb_test_preds += cb_model.predict(test_pool) / n_splits

    # --- LightGBM ---
    X_tr_lgb, X_val_lgb = X_train_lgb.iloc[train_idx], X_train_lgb.iloc[val_idx]
    lgb_train_data = lgb.Dataset(X_tr_lgb, label=y_tr)
    lgb_val_data = lgb.Dataset(X_val_lgb, label=y_val, reference=lgb_train_data)

    lgb_model = lgb.train(
        lgb_params,
        lgb_train_data,
        num_boost_round=2000,
        valid_sets=[lgb_val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    lgb_val_pred = lgb_model.predict(X_val_lgb, num_iteration=lgb_model.best_iteration)
    lgb_oof[val_idx] = lgb_val_pred
    lgb_test_preds += (
        lgb_model.predict(X_test_lgb, num_iteration=lgb_model.best_iteration) / n_splits
    )

    print(
        f"Fold {fold + 1} | CB RMSE: {root_mean_squared_error(y_val, cb_val_pred):.5f} | LGB RMSE: {root_mean_squared_error(y_val, lgb_val_pred):.5f}"
    )

cb_cv_rmse = root_mean_squared_error(y_train, cb_oof)
lgb_cv_rmse = root_mean_squared_error(y_train, lgb_oof)
print(f"\nOverall CatBoost OOF RMSE: {cb_cv_rmse:.5f}")
print(f"Overall LightGBM OOF RMSE: {lgb_cv_rmse:.5f}")


# 3. Find optimal ensemble weights
def blend_obj(weights):
    w_cb, w_lgb = weights
    pred = w_cb * cb_oof + w_lgb * lgb_oof
    return root_mean_squared_error(y_train, pred)


opt_res = minimize(
    blend_obj,
    x0=[0.5, 0.5],
    bounds=[(0, 1), (0, 1)],
    constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0},
)
w_cb, w_lgb = opt_res.x
print(f"Optimal Blend Weights -> CatBoost: {w_cb:.4f}, LightGBM: {w_lgb:.4f}")

ensemble_oof = w_cb * cb_oof + w_lgb * lgb_oof
ensemble_cv_rmse = root_mean_squared_error(y_train, ensemble_oof)
print(f"Ensemble Cross-Validation RMSE: {ensemble_cv_rmse:.5f}")

# 4. Generate test predictions and save submission
final_test_preds = w_cb * cb_test_preds + w_lgb * lgb_test_preds
submission = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "SupStruc": final_test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
