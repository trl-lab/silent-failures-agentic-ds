import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from scipy.optimize import minimize
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb

# Paths setup
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"
output_dir = "./working"
os.makedirs(output_dir, exist_ok=True)

# Load data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "entityPayment"
features = [col for col in train_df.columns if col != target_col]

# Combine for basic feature engineering
combined = pd.concat([train_df[features], test_df[features]], axis=0).reset_index(
    drop=True
)

# Datetime feature extraction
date_cols = []
for col in combined.columns:
    if (
        np.issubdtype(combined[col].dtype, np.datetime64)
        or "date" in col.lower()
        or "time" in col.lower()
    ):
        try:
            combined[col] = pd.to_datetime(combined[col])
            date_cols.append(col)
        except Exception:
            pass

for col in date_cols:
    combined[f"{col}_year"] = combined[col].dt.year
    combined[f"{col}_month"] = combined[col].dt.month
    combined[f"{col}_day"] = combined[col].dt.day
    combined[f"{col}_dayofweek"] = combined[col].dt.dayofweek
    combined[f"{col}_quarter"] = combined[col].dt.quarter
    combined[f"{col}_is_month_end"] = combined[col].dt.is_month_end.astype(int)
    combined.drop(columns=[col], inplace=True)

# Identify categorical and numerical columns
cat_cols = []
num_cols = []
for col in combined.columns:
    if (
        combined[col].dtype == "object"
        or combined[col].dtype.name == "category"
        or combined[col].dtype == "bool"
    ):
        combined[col] = combined[col].fillna("Missing").astype(str)
        cat_cols.append(col)
    else:
        combined[col] = combined[col].fillna(combined[col].median())
        num_cols.append(col)

# Frequency encoding
for col in cat_cols:
    freq = combined[col].value_counts()
    combined[f"{col}_freq"] = combined[col].map(freq).astype(float)

# Category-level numerical aggregations and relative features
for cat_c in cat_cols:
    for num_c in num_cols:
        group_stats = combined.groupby(cat_c)[num_c].agg(["mean", "std"]).reset_index()
        group_stats.columns = [cat_c, f"{cat_c}_{num_c}_mean", f"{cat_c}_{num_c}_std"]
        combined = combined.merge(group_stats, on=cat_c, how="left")
        combined[f"{cat_c}_{num_c}_diff_mean"] = (
            combined[num_c] - combined[f"{cat_c}_{num_c}_mean"]
        )
        combined[f"{cat_c}_{num_c}_ratio_mean"] = combined[num_c] / (
            combined[f"{cat_c}_{num_c}_mean"].abs() + 1e-5
        )

# Fill any residual NaNs
for col in combined.columns:
    if col in cat_cols:
        combined[col] = combined[col].fillna("Missing").astype(str)
    else:
        combined[col] = combined[col].fillna(0)

n_train = len(train_df)
n_test = len(test_df)
y = train_df[target_col].values
global_mean = np.mean(y)

# 5-fold cross-validation setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Out-of-fold target encoding computation strictly within CV
smooth_weight = 10.0
te_cols_mean = {}
te_cols_std = {}
for col in cat_cols:
    te_cols_mean[col] = np.zeros(n_train)
    te_cols_std[col] = np.zeros(n_train)

test_te_mean = {col: np.zeros(n_test) for col in cat_cols}
test_te_std = {col: np.zeros(n_test) for col in cat_cols}

X_raw_train = combined.iloc[:n_train].copy()
X_raw_test = combined.iloc[n_train:].copy()

for fold, (train_idx, val_idx) in enumerate(kf.split(X_raw_train, y)):
    y_tr = y[train_idx]
    for col in cat_cols:
        cat_tr = X_raw_train[col].iloc[train_idx]
        cat_va = X_raw_train[col].iloc[val_idx]
        cat_te = X_raw_test[col]

        stats = (
            pd.DataFrame({col: cat_tr, "target": y_tr})
            .groupby(col)["target"]
            .agg(["count", "mean", "std"])
        )
        stats["std"] = stats["std"].fillna(0)
        smoothed_mean = (
            stats["count"] * stats["mean"] + smooth_weight * global_mean
        ) / (stats["count"] + smooth_weight)

        te_cols_mean[col][val_idx] = (
            cat_va.map(smoothed_mean).fillna(global_mean).values
        )
        te_cols_std[col][val_idx] = cat_va.map(stats["std"]).fillna(0).values

        test_te_mean[col] += (
            cat_te.map(smoothed_mean).fillna(global_mean).values / n_splits
        )
        test_te_std[col] += cat_te.map(stats["std"]).fillna(0).values / n_splits

# Append target encodings to train and test
for col in cat_cols:
    X_raw_train[f"{col}_te_mean"] = te_cols_mean[col]
    X_raw_train[f"{col}_te_std"] = te_cols_std[col]
    X_raw_test[f"{col}_te_mean"] = test_te_mean[col]
    X_raw_test[f"{col}_te_std"] = test_te_std[col]

# Prepare datasets for CatBoost and LightGBM
cat_indices = [
    X_raw_train.columns.get_loc(c) for c in cat_cols if c in X_raw_train.columns
]

X_lgb_train = X_raw_train.copy()
X_lgb_test = X_raw_test.copy()
for col in cat_cols:
    X_lgb_train[col] = X_lgb_train[col].astype("category")
    X_lgb_test[col] = X_lgb_test[col].astype("category")

cb_oof = np.zeros(n_train)
cb_test = np.zeros(n_test)
lgb_oof = np.zeros(n_train)
lgb_test = np.zeros(n_test)

cb_params = {
    "iterations": 2000,
    "learning_rate": 0.035,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "verbose": False,
}

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.035,
    "num_leaves": 31,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_raw_train, y)):
    y_tr, y_va = y[train_idx], y[val_idx]

    # 1. CatBoost
    X_tr_cb = X_raw_train.iloc[train_idx]
    X_va_cb = X_raw_train.iloc[val_idx]

    cb_train_pool = Pool(X_tr_cb, y_tr, cat_features=cat_indices)
    cb_val_pool = Pool(X_va_cb, y_va, cat_features=cat_indices)
    cb_test_pool = Pool(X_raw_test, cat_features=cat_indices)

    cb_model = CatBoostRegressor(**cb_params)
    cb_model.fit(
        cb_train_pool,
        eval_set=cb_val_pool,
        early_stopping_rounds=100,
        verbose=False,
    )

    cb_oof[val_idx] = cb_model.predict(cb_val_pool)
    cb_test += cb_model.predict(cb_test_pool) / n_splits

    # 2. LightGBM
    X_tr_lgb = X_lgb_train.iloc[train_idx]
    X_va_lgb = X_lgb_train.iloc[val_idx]

    lgb_train_data = lgb.Dataset(X_tr_lgb, label=y_tr)
    lgb_val_data = lgb.Dataset(X_va_lgb, label=y_va, reference=lgb_train_data)

    lgb_model = lgb.train(
        lgb_params,
        lgb_train_data,
        num_boost_round=2000,
        valid_sets=[lgb_val_data],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)],
    )

    lgb_oof[val_idx] = lgb_model.predict(
        X_va_lgb, num_iteration=lgb_model.best_iteration
    )
    lgb_test += (
        lgb_model.predict(X_lgb_test, num_iteration=lgb_model.best_iteration) / n_splits
    )

cb_rmse = root_mean_squared_error(y, cb_oof)
lgb_rmse = root_mean_squared_error(y, lgb_oof)
print(f"CatBoost OOF RMSE: {cb_rmse:.5f}")
print(f"LightGBM OOF RMSE: {lgb_rmse:.5f}")


# Optimize blending weights
def loss_func(weights):
    w1, w2 = weights
    pred = w1 * cb_oof + w2 * lgb_oof
    return root_mean_squared_error(y, pred)


opt_res = minimize(
    loss_func,
    [0.5, 0.5],
    bounds=[(0, 1), (0, 1)],
    constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0},
)
w_cb, w_lgb = opt_res.x
ensemble_oof = w_cb * cb_oof + w_lgb * lgb_oof
ensemble_rmse = root_mean_squared_error(y, ensemble_oof)
print(f"Optimized Weights: CatBoost = {w_cb:.4f}, LightGBM = {w_lgb:.4f}")
print(f"Validation Metric (5-Fold Ensemble OOF RMSE): {ensemble_rmse:.5f}")

# Final ensemble test predictions
final_test_preds = w_cb * cb_test + w_lgb * lgb_test

# Save submission
submission = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "entityPayment": final_test_preds}
)
submission.to_csv(os.path.join(output_dir, "submission.csv"), index=False)
print("Submission saved successfully.")
