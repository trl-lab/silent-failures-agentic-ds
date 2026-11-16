import gc
import json
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_error

# Create directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


# =========================================================================
# 1. EVALUATION METRIC FUNCTIONS
# =========================================================================
def compute_rmsle(y_true_raw: np.ndarray, y_pred_raw: np.ndarray) -> float:
    """Computes the exact Root Mean Squared Logarithmic Error (RMSLE) on raw sales values.

    RMSLE = sqrt(mean((log(y_pred + 1) - log(y_true + 1))^2))
    """
    y_true_clipped = np.maximum(0.0, y_true_raw)
    y_pred_clipped = np.maximum(0.0, y_pred_raw)
    log_true = np.log1p(y_true_clipped)
    log_pred = np.log1p(y_pred_clipped)
    return float(np.sqrt(np.mean((log_pred - log_true) ** 2)))


def compute_rmse_log(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """Computes Root Mean Squared Error on log1p-transformed sales."""
    y_pred_clipped = np.maximum(0.0, y_pred_log)
    return float(root_mean_squared_error(y_true_log, y_pred_clipped))


# =========================================================================
# 2. DATA LOADING & PREPROCESSING
# =========================================================================
print("Loading primary datasets...")
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

train_df["date"] = pd.to_datetime(train_df["date"])
test_df["date"] = pd.to_datetime(test_df["date"])

train_df = train_df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
test_df = test_df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

# Load supplemental metadata if available
stores_df = None
oil_df = None
holidays_df = None

for base_dir in ["./input", "./input/original"]:
    s_path = os.path.join(base_dir, "stores.csv")
    o_path = os.path.join(base_dir, "oil.csv")
    h_path = os.path.join(base_dir, "holidays_events.csv")

    if os.path.exists(s_path) and stores_df is None:
        stores_df = pd.read_csv(s_path)
    if os.path.exists(o_path) and oil_df is None:
        oil_df = pd.read_csv(o_path)
        oil_df["date"] = pd.to_datetime(oil_df["date"])
    if os.path.exists(h_path) and holidays_df is None:
        holidays_df = pd.read_csv(h_path)
        holidays_df["date"] = pd.to_datetime(holidays_df["date"])

if stores_df is not None:
    train_df = train_df.merge(stores_df, on="store_nbr", how="left")
    test_df = test_df.merge(stores_df, on="store_nbr", how="left")

if oil_df is not None:
    all_dates = pd.date_range(
        start=min(train_df["date"].min(), oil_df["date"].min()),
        end=max(test_df["date"].max(), oil_df["date"].max()),
    )
    oil_full = pd.DataFrame({"date": all_dates}).merge(oil_df, on="date", how="left")
    oil_full["dcoilwtico"] = (
        oil_full["dcoilwtico"].interpolate(method="linear").bfill().ffill()
    )
    oil_full["oil_rolling_mean_7"] = (
        oil_full["dcoilwtico"].rolling(7, min_periods=1).mean()
    )
    oil_full["oil_rolling_mean_30"] = (
        oil_full["dcoilwtico"].rolling(30, min_periods=1).mean()
    )
    train_df = train_df.merge(oil_full, on="date", how="left")
    test_df = test_df.merge(oil_full, on="date", how="left")

if holidays_df is not None:
    holidays_valid = holidays_df[~holidays_df["transferred"]].copy()
    holidays_valid = holidays_valid.drop_duplicates(subset=["date"])
    holidays_valid["is_holiday"] = 1
    holidays_valid = holidays_valid[["date", "is_holiday", "type", "locale"]].rename(
        columns={"type": "holiday_type", "locale": "holiday_locale"}
    )
    train_df = train_df.merge(holidays_valid, on="date", how="left")
    test_df = test_df.merge(holidays_valid, on="date", how="left")
    train_df["is_holiday"] = train_df["is_holiday"].fillna(0).astype(np.int8)
    test_df["is_holiday"] = test_df["is_holiday"].fillna(0).astype(np.int8)
    train_df["holiday_type"] = train_df["holiday_type"].fillna("None")
    test_df["holiday_type"] = test_df["holiday_type"].fillna("None")
    train_df["holiday_locale"] = train_df["holiday_locale"].fillna("None")
    test_df["holiday_locale"] = test_df["holiday_locale"].fillna("None")

train_df["is_test"] = 0
test_df["is_test"] = 1
test_df["sales"] = np.nan

full_df = pd.concat([train_df, test_df], ignore_index=True)
full_df = full_df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

# Target transformation: log1p
full_df["target_log"] = np.log1p(np.maximum(0.0, full_df["sales"]))

# =========================================================================
# 3. FEATURE ENGINEERING
# =========================================================================
print("Engineering calendar, payday, and horizon-aligned lag features...")

full_df["year"] = full_df["date"].dt.year.astype(np.int16)
full_df["month"] = full_df["date"].dt.month.astype(np.int8)
full_df["day"] = full_df["date"].dt.day.astype(np.int8)
full_df["dayofweek"] = full_df["date"].dt.dayofweek.astype(np.int8)
full_df["dayofyear"] = full_df["date"].dt.dayofyear.astype(np.int16)
full_df["weekofyear"] = full_df["date"].dt.isocalendar().week.astype(np.int8)
full_df["quarter"] = full_df["date"].dt.quarter.astype(np.int8)
full_df["is_weekend"] = full_df["dayofweek"].isin([5, 6]).astype(np.int8)
full_df["is_month_start"] = full_df["date"].dt.is_month_start.astype(np.int8)
full_df["is_month_end"] = full_df["date"].dt.is_month_end.astype(np.int8)

# Ecuadorian Paydays: 15th and month-end
full_df["is_payday"] = ((full_df["day"] == 15) | (full_df["is_month_end"] == 1)).astype(
    np.int8
)
full_df["days_to_payday"] = np.minimum(
    np.abs(full_df["day"] - 15),
    np.abs(full_df["day"] - full_df["date"].dt.days_in_month),
).astype(np.int8)

# Cyclical Transformations
full_df["sin_dayofweek"] = np.sin(2 * np.pi * full_df["dayofweek"] / 7.0).astype(
    np.float32
)
full_df["cos_dayofweek"] = np.cos(2 * np.pi * full_df["dayofweek"] / 7.0).astype(
    np.float32
)
full_df["sin_month"] = np.sin(2 * np.pi * full_df["month"] / 12.0).astype(np.float32)
full_df["cos_month"] = np.cos(2 * np.pi * full_df["month"] / 12.0).astype(np.float32)
full_df["sin_dayofyear"] = np.sin(2 * np.pi * full_df["dayofyear"] / 365.25).astype(
    np.float32
)
full_df["cos_dayofyear"] = np.cos(2 * np.pi * full_df["dayofyear"] / 365.25).astype(
    np.float32
)

# Promotions
full_df["onpromotion"] = full_df["onpromotion"].fillna(0).astype(np.float32)
full_df["has_promotion"] = (full_df["onpromotion"] > 0).astype(np.int8)
full_df["log_onpromotion"] = np.log1p(full_df["onpromotion"]).astype(np.float32)

# Horizon-Aligned Lags & Rolling Statistics (lag >= 16 to support direct 16-day forecasting)
grouped = full_df.groupby(["store_nbr", "family"])

lag_days = [16, 17, 18, 19, 20, 21, 22, 28, 35, 42, 56, 84]
for lag in lag_days:
    full_df[f"sales_lag_{lag}"] = grouped["target_log"].shift(lag).astype(np.float32)

lag16_series = full_df.groupby(["store_nbr", "family"])["target_log"].shift(16)
full_df["_temp_lag16"] = lag16_series
grouped_lag16 = full_df.groupby(["store_nbr", "family"])["_temp_lag16"]

for window in [7, 14, 30, 60]:
    full_df[f"sales_rolling_mean_{window}"] = (
        grouped_lag16.rolling(window, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
        .astype(np.float32)
    )
    full_df[f"sales_rolling_std_{window}"] = (
        grouped_lag16.rolling(window, min_periods=1)
        .std()
        .reset_index(level=[0, 1], drop=True)
        .fillna(0)
        .astype(np.float32)
    )
    full_df[f"sales_rolling_max_{window}"] = (
        grouped_lag16.rolling(window, min_periods=1)
        .max()
        .reset_index(level=[0, 1], drop=True)
        .astype(np.float32)
    )
    full_df[f"sales_rolling_min_{window}"] = (
        grouped_lag16.rolling(window, min_periods=1)
        .min()
        .reset_index(level=[0, 1], drop=True)
        .astype(np.float32)
    )

full_df["sales_mean_ratio_7_30"] = (
    full_df["sales_rolling_mean_7"] / (full_df["sales_rolling_mean_30"] + 1e-4)
).astype(np.float32)
full_df["sales_mean_ratio_14_60"] = (
    full_df["sales_rolling_mean_14"] / (full_df["sales_rolling_mean_60"] + 1e-4)
).astype(np.float32)

full_df["promo_lag_16"] = grouped["onpromotion"].shift(16).astype(np.float32)
full_df["promo_rolling_mean_7"] = (
    grouped["onpromotion"]
    .rolling(7, min_periods=1)
    .mean()
    .reset_index(level=[0, 1], drop=True)
    .astype(np.float32)
)
full_df["promo_rolling_mean_14"] = (
    grouped["onpromotion"]
    .rolling(14, min_periods=1)
    .mean()
    .reset_index(level=[0, 1], drop=True)
    .astype(np.float32)
)

full_df.drop(columns=["_temp_lag16"], inplace=True)

# Categorical Label Encoding
categorical_cols = [
    "family",
    "city",
    "state",
    "type",
    "holiday_type",
    "holiday_locale",
]
for col in categorical_cols:
    if col in full_df.columns:
        full_df[col] = full_df[col].astype("category").cat.codes.astype(np.int16)

full_df["store_nbr"] = full_df["store_nbr"].astype(np.int16)
if "cluster" in full_df.columns:
    full_df["cluster"] = full_df["cluster"].astype(np.int16)

# =========================================================================
# 4. STRICT TIME-SERIES SPLITTING & LEAK-FREE TARGET ENCODING
# =========================================================================
val_end_date = train_df["date"].max()
val_start_date = val_end_date - pd.Timedelta(days=15)
train_cutoff_date = pd.to_datetime("2015-01-01")

print(f"Validation Period: {val_start_date.date()} to {val_end_date.date()} (16 days)")

train_mask_for_enc = (
    (full_df["is_test"] == 0)
    & (full_df["date"] < val_start_date)
    & (full_df["date"] >= train_cutoff_date)
)
train_sub = full_df[train_mask_for_enc]

store_mean = (
    train_sub.groupby("store_nbr")["target_log"].mean().rename("enc_store_mean")
)
family_mean = train_sub.groupby("family")["target_log"].mean().rename("enc_family_mean")
store_family_mean = (
    train_sub.groupby(["store_nbr", "family"])["target_log"]
    .mean()
    .rename("enc_store_family_mean")
)
fam_dow_mean = (
    train_sub.groupby(["family", "dayofweek"])["target_log"]
    .mean()
    .rename("enc_fam_dow_mean")
)

full_df = full_df.merge(store_mean, on="store_nbr", how="left")
full_df = full_df.merge(family_mean, on="family", how="left")
full_df = full_df.merge(store_family_mean, on=["store_nbr", "family"], how="left")
full_df = full_df.merge(fam_dow_mean, on=["family", "dayofweek"], how="left")

global_mean = train_sub["target_log"].mean()
for enc_col in [
    "enc_store_mean",
    "enc_family_mean",
    "enc_store_family_mean",
    "enc_fam_dow_mean",
]:
    full_df[enc_col] = full_df[enc_col].fillna(global_mean).astype(np.float32)

train_final = full_df[
    (full_df["is_test"] == 0)
    & (full_df["date"] < val_start_date)
    & (full_df["date"] >= train_cutoff_date)
].copy()
val_final = full_df[
    (full_df["is_test"] == 0)
    & (full_df["date"] >= val_start_date)
    & (full_df["date"] <= val_end_date)
].copy()
test_final = full_df[full_df["is_test"] == 1].copy()

test_final = test_final.sort_values("id").reset_index(drop=True)
train_final = train_final.reset_index(drop=True)
val_final = val_final.reset_index(drop=True)

exclude_cols = ["id", "date", "sales", "target_log", "is_test"]
feature_cols = [c for c in train_final.columns if c not in exclude_cols]

candidate_cats = [
    "family",
    "store_nbr",
    "city",
    "state",
    "type",
    "cluster",
    "holiday_type",
    "holiday_locale",
]
cat_features = [c for c in candidate_cats if c in feature_cols]

X_train = train_final[feature_cols]
y_train = train_final["target_log"].values

X_val = val_final[feature_cols]
y_val = val_final["target_log"].values
y_val_raw = val_final["sales"].values

X_test = test_final[feature_cols]
test_ids = test_final["id"].values

print(f"Features count: {len(feature_cols)}")
print(
    f"Train samples: {len(X_train):,}, Validation samples: {len(X_val):,}, Test samples: {len(X_test):,}"
)

# Clean up memory
del full_df, train_sub, train_df, test_df
gc.collect()

# =========================================================================
# 5. MODEL 1: LIGHTGBM (Leaf-wise GBDT)
# =========================================================================
print("\n--- Training LightGBM Model 1 ---")
lgb_params_1 = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 127,
    "max_depth": 9,
    "min_child_samples": 30,
    "subsample": 0.80,
    "subsample_freq": 1,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

dtrain = lgb.Dataset(
    X_train, label=y_train, categorical_feature=cat_features, free_raw_data=False
)
dval = lgb.Dataset(
    X_val,
    label=y_val,
    reference=dtrain,
    categorical_feature=cat_features,
    free_raw_data=False,
)

lgb_model_1 = lgb.train(
    lgb_params_1,
    dtrain,
    num_boost_round=2500,
    valid_sets=[dtrain, dval],
    valid_names=["train", "val"],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
)

val_preds_log_lgb1 = np.maximum(
    0.0, lgb_model_1.predict(X_val, num_iteration=lgb_model_1.best_iteration)
)
val_preds_raw_lgb1 = np.maximum(0.0, np.expm1(val_preds_log_lgb1))
rmsle_lgb1 = compute_rmsle(y_val_raw, val_preds_raw_lgb1)
print(
    f"LightGBM 1 Best Iteration: {lgb_model_1.best_iteration} | Val RMSLE: {rmsle_lgb1:.5f}"
)

# =========================================================================
# 6. MODEL 2: LIGHTGBM (Deep Capacity & Alternate Subsample)
# =========================================================================
print("\n--- Training LightGBM Model 2 ---")
lgb_params_2 = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.025,
    "num_leaves": 255,
    "max_depth": 11,
    "min_child_samples": 40,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.65,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "random_state": 2024,
    "n_jobs": -1,
    "verbose": -1,
}

lgb_model_2 = lgb.train(
    lgb_params_2,
    dtrain,
    num_boost_round=2500,
    valid_sets=[dtrain, dval],
    valid_names=["train", "val"],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
)

val_preds_log_lgb2 = np.maximum(
    0.0, lgb_model_2.predict(X_val, num_iteration=lgb_model_2.best_iteration)
)
val_preds_raw_lgb2 = np.maximum(0.0, np.expm1(val_preds_log_lgb2))
rmsle_lgb2 = compute_rmsle(y_val_raw, val_preds_raw_lgb2)
print(
    f"LightGBM 2 Best Iteration: {lgb_model_2.best_iteration} | Val RMSLE: {rmsle_lgb2:.5f}"
)

# =========================================================================
# 7. MODEL 3: CATBOOST REGRESSOR (Oblivious Trees)
# =========================================================================
print("\n--- Training CatBoost Regressor ---")
cb_params = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 1800,
    "learning_rate": 0.04,
    "depth": 8,
    "l2_leaf_reg": 4.0,
    "random_seed": 42,
    "verbose": False,
    "thread_count": -1,
}

cb_model = CatBoostRegressor(**cb_params)
cb_model.fit(
    X_train,
    y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=50,
    verbose=False,
)

val_preds_log_cb = np.maximum(
    0.0, cb_model.predict(X_val, ntree_end=cb_model.get_best_iteration())
)
val_preds_raw_cb = np.maximum(0.0, np.expm1(val_preds_log_cb))
rmsle_cb = compute_rmsle(y_val_raw, val_preds_raw_cb)
print(
    f"CatBoost Best Iteration: {cb_model.get_best_iteration()} | Val RMSLE: {rmsle_cb:.5f}"
)

# =========================================================================
# 8. ENSEMBLE EVALUATION
# =========================================================================
print("\n--- Evaluating Diverse Ensemble ---")
w_lgb1, w_lgb2, w_cb = 0.40, 0.35, 0.25

val_preds_log_ens = (
    w_lgb1 * val_preds_log_lgb1 + w_lgb2 * val_preds_log_lgb2 + w_cb * val_preds_log_cb
)
val_preds_raw_ens = np.maximum(0.0, np.expm1(val_preds_log_ens))
ensemble_rmsle = compute_rmsle(y_val_raw, val_preds_raw_ens)
print(f"Ensemble Holdout Validation RMSLE: {ensemble_rmsle:.5f}")

# =========================================================================
# 9. TEST SET INFERENCE & SUBMISSION GENERATION
# =========================================================================
print("\nGenerating final test predictions...")
test_pred_log_lgb1 = np.maximum(
    0.0, lgb_model_1.predict(X_test, num_iteration=lgb_model_1.best_iteration)
)
test_pred_log_lgb2 = np.maximum(
    0.0, lgb_model_2.predict(X_test, num_iteration=lgb_model_2.best_iteration)
)
test_pred_log_cb = np.maximum(
    0.0, cb_model.predict(X_test, ntree_end=cb_model.get_best_iteration())
)

test_pred_log_ens = (
    w_lgb1 * test_pred_log_lgb1 + w_lgb2 * test_pred_log_lgb2 + w_cb * test_pred_log_cb
)
test_preds_sales = np.maximum(0.0, np.expm1(test_pred_log_ens))

submission_df = pd.DataFrame(
    {"id": test_ids.astype(np.int64), "sales": test_preds_sales}
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission saved to {submission_path} with shape {submission_df.shape}")
print(f"Final Validation Score: {ensemble_rmsle}")
