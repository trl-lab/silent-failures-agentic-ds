import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from scipy.optimize import minimize
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb

# Load datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Overall"
y = train_df[target_col].values

drop_cols = [target_col]
if "id" in train_df.columns:
    drop_cols.append("id")

feature_cols = [c for c in train_df.columns if c not in drop_cols]
X = train_df[feature_cols].copy()
X_test = test_df[feature_cols].copy()

# Identify categorical and numerical columns
cat_cols = []
num_cols = []
for col in feature_cols:
    if (
        X[col].dtype == "object"
        or pd.api.types.is_categorical_dtype(X[col])
        or pd.api.types.is_string_dtype(X[col])
    ):
        cat_cols.append(col)
        X[col] = X[col].fillna("__MISSING__").astype(str)
        X_test[col] = X_test[col].fillna("__MISSING__").astype(str)
    else:
        num_cols.append(col)
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

# Feature Engineering 1: Row-wise numerical summary statistics
if len(num_cols) > 1:
    for df in [X, X_test]:
        num_values = df[num_cols].values
        df["row_mean"] = np.nanmean(num_values, axis=1)
        df["row_std"] = np.nanstd(num_values, axis=1)
        df["row_min"] = np.nanmin(num_values, axis=1)
        df["row_max"] = np.nanmax(num_values, axis=1)
        df["row_n_missing"] = np.isnan(num_values).sum(axis=1)

# Feature Engineering 2: Frequency encoding for categorical columns
for col in cat_cols:
    freq = pd.concat([X[col], X_test[col]]).value_counts()
    X[f"{col}_freq"] = X[col].map(freq).fillna(0).astype(float)
    X_test[f"{col}_freq"] = X_test[col].map(freq).fillna(0).astype(float)

# Feature Engineering 3: Categorical group aggregations on numeric features
if len(num_cols) > 0 and len(cat_cols) > 0:
    combined_df = pd.concat([X, X_test], ignore_index=True)
    top_num_cols = num_cols[: min(5, len(num_cols))]
    for c_col in cat_cols:
        for n_col in top_num_cols:
            grp_mean = combined_df.groupby(c_col)[n_col].transform("mean")
            grp_std = combined_df.groupby(c_col)[n_col].transform("std").fillna(0)

            X[f"{c_col}_{n_col}_grp_mean"] = grp_mean.iloc[: len(X)].values
            X[f"{c_col}_{n_col}_grp_std"] = grp_std.iloc[: len(X)].values
            X_test[f"{c_col}_{n_col}_grp_mean"] = grp_mean.iloc[len(X) :].values
            X_test[f"{c_col}_{n_col}_grp_std"] = grp_std.iloc[len(X) :].values

# Feature Engineering 4: Out-of-fold Target Encoding with smoothing
kf = KFold(n_splits=5, shuffle=True, random_state=42)
smoothing = 10.0
global_mean = np.mean(y)

for col in cat_cols:
    te_oof = np.zeros(len(X))
    te_test_folds = np.zeros((kf.n_splits, len(X_test)))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        tr_cat = X.iloc[train_idx][col]
        tr_y = y[train_idx]

        # Calculate smoothed target mean on train fold
        cat_stats = (
            pd.DataFrame({"cat": tr_cat, "target": tr_y})
            .groupby("cat")["target"]
            .agg(["count", "mean"])
        )
        smoothed = (
            cat_stats["count"] * cat_stats["mean"] + smoothing * global_mean
        ) / (cat_stats["count"] + smoothing)

        te_oof[val_idx] = X.iloc[val_idx][col].map(smoothed).fillna(global_mean).values
        te_test_folds[fold] = X_test[col].map(smoothed).fillna(global_mean).values

    X[f"{col}_te"] = te_oof
    X_test[f"{col}_te"] = te_test_folds.mean(axis=0)

# Prepare categorical dtypes for LightGBM and XGBoost
X_lgb = X.copy()
X_test_lgb = X_test.copy()
for col in cat_cols:
    X_lgb[col] = X_lgb[col].astype("category")
    X_test_lgb[col] = X_test_lgb[col].astype("category")

cb_oof = np.zeros(len(train_df))
cb_test = np.zeros(len(test_df))

lgb_oof = np.zeros(len(train_df))
lgb_test = np.zeros(len(test_df))

xgb_oof = np.zeros(len(train_df))
xgb_test = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr_cb, y_tr = X.iloc[train_idx], y[train_idx]
    X_va_cb, y_va = X.iloc[val_idx], y[val_idx]

    # 1. CatBoost
    cb_model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42 + fold,
        cat_features=cat_cols if len(cat_cols) > 0 else None,
        verbose=0,
    )
    cb_model.fit(
        X_tr_cb,
        y_tr,
        eval_set=(X_va_cb, y_va),
        early_stopping_rounds=50,
        verbose=False,
    )
    cb_oof[val_idx] = cb_model.predict(X_va_cb)
    cb_test += cb_model.predict(X_test) / kf.n_splits

    # 2. LightGBM
    X_tr_lgb, X_va_lgb = X_lgb.iloc[train_idx], X_lgb.iloc[val_idx]
    lgb_model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        objective="regression",
        metric="rmse",
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr_lgb,
        y_tr,
        eval_set=[(X_va_lgb, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    lgb_oof[val_idx] = lgb_model.predict(X_va_lgb)
    lgb_test += lgb_model.predict(X_test_lgb) / kf.n_splits

    # 3. XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        tree_method="hist",
        random_state=42 + fold,
        eval_metric="rmse",
        early_stopping_rounds=50,
        n_jobs=-1,
    )
    xgb_model.fit(
        X_tr_lgb,
        y_tr,
        eval_set=[(X_va_lgb, y_va)],
        verbose=False,
    )
    xgb_oof[val_idx] = xgb_model.predict(X_va_lgb)
    xgb_test += xgb_model.predict(X_test_lgb) / kf.n_splits


# Optimize ensemble weights across CatBoost, LightGBM, and XGBoost
def blend_objective(weights):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    pred = weights[0] * cb_oof + weights[1] * lgb_oof + weights[2] * xgb_oof
    return root_mean_squared_error(y, pred)


init_weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
constraints = {"type": "eq", "fun": lambda w: sum(w) - 1.0}

opt_res = minimize(
    blend_objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
weights = opt_res.x / np.sum(opt_res.x)

oof_predictions = weights[0] * cb_oof + weights[1] * lgb_oof + weights[2] * xgb_oof
test_predictions = weights[0] * cb_test + weights[1] * lgb_test + weights[2] * xgb_test

cv_rmse = root_mean_squared_error(y, oof_predictions)
print(f"Optimal Blend Weights [CB, LGB, XGB]: {weights.round(4)}")
print(f"Validation RMSE: {cv_rmse:.5f}")

# Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_df)), "Overall": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
