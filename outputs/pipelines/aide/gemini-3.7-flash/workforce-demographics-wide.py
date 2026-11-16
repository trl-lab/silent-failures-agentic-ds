import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

# 1. Load data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Female (%)"
target = train_df[target_col].values

features = [c for c in train_df.columns if c != target_col]

cat_cols = []
num_cols = []
for c in features:
    if train_df[c].dtype == "object" or train_df[c].dtype.name == "category":
        cat_cols.append(c)
    else:
        num_cols.append(c)

# Create copies for different model requirements
cb_train = train_df[features].copy()
cb_test = test_df[features].copy()

# Categorical and numerical preprocessing
for c in cat_cols:
    cb_train[c] = cb_train[c].fillna("Missing").astype(str)
    cb_test[c] = cb_test[c].fillna("Missing").astype(str)

for c in num_cols:
    med = cb_train[c].median()
    cb_train[c] = cb_train[c].fillna(med)
    cb_test[c] = cb_test[c].fillna(med)

# Group aggregations and frequency encoding for tree models
tab_train = cb_train.copy()
tab_test = cb_test.copy()

for cat_c in cat_cols:
    freq = tab_train[cat_c].value_counts(normalize=True).to_dict()
    tab_train[f"{cat_c}_freq"] = tab_train[cat_c].map(freq).fillna(0).astype(float)
    tab_test[f"{cat_c}_freq"] = tab_test[cat_c].map(freq).fillna(0).astype(float)

    for num_c in num_cols[:3]:
        g_mean = tab_train.groupby(cat_c)[num_c].mean().to_dict()
        g_std = tab_train.groupby(cat_c)[num_c].std().to_dict()
        overall_mean = tab_train[num_c].mean()
        tab_train[f"{cat_c}_{num_c}_mean"] = (
            tab_train[cat_c].map(g_mean).fillna(overall_mean).astype(float)
        )
        tab_test[f"{cat_c}_{num_c}_mean"] = (
            tab_test[cat_c].map(g_mean).fillna(overall_mean).astype(float)
        )
        tab_train[f"{cat_c}_{num_c}_std"] = (
            tab_train[cat_c].map(g_std).fillna(0).astype(float)
        )
        tab_test[f"{cat_c}_{num_c}_std"] = (
            tab_test[cat_c].map(g_std).fillna(0).astype(float)
        )

# Also update cb_train/test with the new features
for col in tab_train.columns:
    if col not in cb_train.columns:
        cb_train[col] = tab_train[col]
        cb_test[col] = tab_test[col]

# Integer encode categorical columns for XGBoost / LightGBM
lgb_xgb_train = tab_train.copy()
lgb_xgb_test = tab_test.copy()

for c in cat_cols:
    categories = {
        val: idx
        for idx, val in enumerate(
            np.unique(np.concatenate([lgb_xgb_train[c].values, lgb_xgb_test[c].values]))
        )
    }
    lgb_xgb_train[c] = lgb_xgb_train[c].map(categories).astype("category")
    lgb_xgb_test[c] = lgb_xgb_test[c].map(categories).astype("category")

xgb_train = lgb_xgb_train.copy()
xgb_test = lgb_xgb_test.copy()
for c in cat_cols:
    xgb_train[c] = xgb_train[c].astype("category")
    xgb_test[c] = xgb_test[c].astype("category")

# 2. 5-Fold Cross Validation Setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

cb_oof = np.zeros(len(train_df))
cb_test_preds = np.zeros(len(test_df))

lgb_oof = np.zeros(len(train_df))
lgb_test_preds = np.zeros(len(test_df))

xgb_oof = np.zeros(len(train_df))
xgb_test_preds = np.zeros(len(test_df))

cb_params = {
    "iterations": 1500,
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
    "boosting_type": "gbdt",
    "learning_rate": 0.04,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "random_state": 42,
    "verbose": -1,
    "n_estimators": 1500,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.04,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "enable_categorical": True,
    "n_estimators": 1500,
    "early_stopping_rounds": 50,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(cb_train, target)):
    # CatBoost
    cb_train_pool = Pool(
        cb_train.iloc[train_idx], target[train_idx], cat_features=cat_cols
    )
    cb_val_pool = Pool(cb_train.iloc[val_idx], target[val_idx], cat_features=cat_cols)
    cb_test_pool = Pool(cb_test, cat_features=cat_cols)

    cb_model = CatBoostRegressor(**cb_params)
    cb_model.fit(cb_train_pool, eval_set=cb_val_pool, use_best_model=True)
    cb_oof[val_idx] = cb_model.predict(cb_val_pool)
    cb_test_preds += cb_model.predict(cb_test_pool) / n_splits

    # LightGBM
    lgb_tr_data = lgb.Dataset(lgb_xgb_train.iloc[train_idx], label=target[train_idx])
    lgb_va_data = lgb.Dataset(
        lgb_xgb_train.iloc[val_idx], label=target[val_idx], reference=lgb_tr_data
    )
    lgb_model = lgb.train(
        lgb_params,
        lgb_tr_data,
        valid_sets=[lgb_va_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    lgb_oof[val_idx] = lgb_model.predict(lgb_xgb_train.iloc[val_idx])
    lgb_test_preds += lgb_model.predict(lgb_xgb_test) / n_splits

    # XGBoost
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(
        xgb_train.iloc[train_idx],
        target[train_idx],
        eval_set=[(xgb_train.iloc[val_idx], target[val_idx])],
        verbose=False,
    )
    xgb_oof[val_idx] = xgb_model.predict(xgb_train.iloc[val_idx])
    xgb_test_preds += xgb_model.predict(xgb_test) / n_splits

print(f"CatBoost OOF RMSE: {root_mean_squared_error(target, cb_oof):.5f}")
print(f"LightGBM OOF RMSE: {root_mean_squared_error(target, lgb_oof):.5f}")
print(f"XGBoost  OOF RMSE: {root_mean_squared_error(target, xgb_oof):.5f}")

# 3. Optimize Ensemble Weights
oof_matrix = np.column_stack([cb_oof, lgb_oof, xgb_oof])
test_matrix = np.column_stack([cb_test_preds, lgb_test_preds, xgb_test_preds])


def ensemble_loss(weights):
    w = np.array(weights)
    w = w / np.sum(w)
    blend_oof = np.dot(oof_matrix, w)
    return root_mean_squared_error(target, blend_oof)


initial_weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
bounds = [(0.0, 1.0)] * 3
res = minimize(
    ensemble_loss,
    initial_weights,
    method="SLSQP",
    bounds=bounds,
    constraints={"type": "eq", "fun": lambda w: 1 - sum(w)},
)
optimal_weights = res.x / np.sum(res.x)

final_oof_preds = np.dot(oof_matrix, optimal_weights)
final_test_preds = np.dot(test_matrix, optimal_weights)
ensemble_rmse = root_mean_squared_error(target, final_oof_preds)

print(
    f"Optimal Ensemble Weights: CatBoost={optimal_weights[0]:.4f}, LightGBM={optimal_weights[1]:.4f}, XGBoost={optimal_weights[2]:.4f}"
)
print(f"5-Fold CV Ensemble Out-of-Fold RMSE: {ensemble_rmse:.5f}")

# 4. Save test predictions
os.makedirs("./working", exist_ok=True)
final_test_preds_clipped = np.clip(final_test_preds, 0.0, 100.0)
submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), target_col: final_test_preds_clipped}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
