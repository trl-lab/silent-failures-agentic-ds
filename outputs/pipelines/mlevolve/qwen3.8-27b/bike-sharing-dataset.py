import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Train date range: {train['dteday'].min()} to {train['dteday'].max()}")
print(f"Test date range: {test['dteday'].min()} to {test['dteday'].max()}")

# Parse dates
train["dteday"] = pd.to_datetime(train["dteday"])
test["dteday"] = pd.to_datetime(test["dteday"])

# Sort by date to ensure temporal order
train = train.sort_values("dteday").reset_index(drop=True)
test = test.sort_values("dteday").reset_index(drop=True)

# ============================================================
# 2. CREATE FULL TIMELINE FOR LAG FEATURES
# ============================================================
train["is_train"] = 1
test["is_train"] = 0

full = pd.concat([train, test], axis=0, ignore_index=True)
full["dteday"] = pd.to_datetime(full["dteday"])
full = full.sort_values("dteday").reset_index(drop=True)

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

# --- 3a. Temporal Cyclical Features ---
full["hour_sin"] = np.sin(2 * np.pi * full["hr"] / 24)
full["hour_cos"] = np.cos(2 * np.pi * full["hr"] / 24)
full["dow_sin"] = np.sin(2 * np.pi * full["weekday"] / 7)
full["dow_cos"] = np.cos(2 * np.pi * full["weekday"] / 7)
full["month_sin"] = np.sin(2 * np.pi * full["mnth"] / 12)
full["month_cos"] = np.cos(2 * np.pi * full["mnth"] / 12)
full["doy_sin"] = np.sin(2 * np.pi * full["dteday"].dt.dayofyear / 365.25)
full["doy_cos"] = np.cos(2 * np.pi * full["dteday"].dt.dayofyear / 365.25)
full["week_of_year"] = full["dteday"].dt.isocalendar().week.astype(int)
full["day_of_year"] = full["dteday"].dt.dayofyear
full["is_weekend"] = (full["weekday"] >= 5).astype(int)

# --- 3b. Weather Interaction Features ---
full["temp_atemp"] = full["temp"] * full["atemp"]
full["temp_hum"] = full["temp"] * full["hum"]
full["temp_wind"] = full["temp"] * full["windspeed"]
full["atemp_hum"] = full["atemp"] * full["hum"]
full["atemp_wind"] = full["atemp"] * full["windspeed"]
full["hum_wind"] = full["hum"] * full["windspeed"]
full["temp_sq"] = full["temp"] ** 2
full["atemp_sq"] = full["atemp"] ** 2
full["hum_sq"] = full["hum"] ** 2
full["wind_sq"] = full["windspeed"] ** 2
full["comfort"] = full["temp"] - 0.5555 * (full["hum"] / 100.0 * 100 - 60) / 100.0
full["feels_like"] = full["atemp"] * (1 - 0.3 * full["windspeed"])

# --- 3c. Weather × Time Interactions ---
full["temp_x_hour_sin"] = full["temp"] * full["hour_sin"]
full["temp_x_hour_cos"] = full["temp"] * full["hour_cos"]
full["temp_x_dow_sin"] = full["temp"] * full["dow_sin"]
full["temp_x_dow_cos"] = full["temp"] * full["dow_cos"]
full["atemp_x_hour_sin"] = full["atemp"] * full["hour_sin"]
full["atemp_x_hour_cos"] = full["atemp"] * full["hour_cos"]
full["hum_x_hour_sin"] = full["hum"] * full["hour_sin"]
full["wind_x_hour_sin"] = full["windspeed"] * full["hour_sin"]
full["wind_x_hour_cos"] = full["windspeed"] * full["hour_cos"]

# --- 3d. Lag Features (using target - safe: uses strictly past values) ---
# Iteratively fill NaN cnt in test region using lag-1 from prior rows
# so that lag/rolling features are properly computed for test data
cnt_filled = full["cnt"].copy()
for i in range(len(cnt_filled)):
    if pd.isna(cnt_filled.iloc[i]):
        cnt_filled.iloc[i] = cnt_filled.iloc[i - 1] if i > 0 else 0

full["cnt_lag_1"] = cnt_filled.shift(1)
full["cnt_lag_24"] = cnt_filled.shift(24)
full["cnt_lag_48"] = cnt_filled.shift(48)
full["cnt_lag_168"] = cnt_filled.shift(168)
full["cnt_lag_336"] = cnt_filled.shift(336)

# Rolling statistics
full["cnt_roll_7d"] = cnt_filled.rolling(168, min_periods=1).mean()
full["cnt_roll_30d"] = cnt_filled.rolling(720, min_periods=1).mean()
full["cnt_roll_7d_std"] = cnt_filled.rolling(168, min_periods=1).std()
full["cnt_roll_30d_std"] = cnt_filled.rolling(720, min_periods=1).std()

# Weather deviation from recent
full["temp_diff_7d"] = full["temp"] - full["temp"].rolling(168, min_periods=1).mean()
full["hum_diff_7d"] = full["hum"] - full["hum"].rolling(168, min_periods=1).mean()
full["wind_diff_7d"] = (
    full["windspeed"] - full["windspeed"].rolling(168, min_periods=1).mean()
)

# --- 3e. Target Transformation ---
full["log_cnt"] = np.log1p(full["cnt"])

# ============================================================
# 4. SPLIT BACK AND CREATE VALIDATION
# ============================================================
train_full = full[full["is_train"] == 1].copy().reset_index(drop=True)
test_processed = full[full["is_train"] == 0].copy().reset_index(drop=True)

# Time-based validation split: last 10% of training data
val_size = int(len(train_full) * 0.10)
train_processed = train_full.iloc[:-val_size].copy().reset_index(drop=True)
val_processed = train_full.iloc[-val_size:].copy().reset_index(drop=True)

print(f"\nTrain (for modeling): {train_processed.shape}")
print(f"Validation: {val_processed.shape}")
print(f"Test: {test_processed.shape}")

# ============================================================
# 5. DEFINE FEATURE COLUMNS
# ============================================================
feature_cols = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "week_of_year",
    "day_of_year",
    "is_weekend",
    "temp_atemp",
    "temp_hum",
    "temp_wind",
    "atemp_hum",
    "atemp_wind",
    "hum_wind",
    "temp_sq",
    "atemp_sq",
    "hum_sq",
    "wind_sq",
    "comfort",
    "feels_like",
    "temp_x_hour_sin",
    "temp_x_hour_cos",
    "temp_x_dow_sin",
    "temp_x_dow_cos",
    "atemp_x_hour_sin",
    "atemp_x_hour_cos",
    "hum_x_hour_sin",
    "wind_x_hour_sin",
    "wind_x_hour_cos",
    "cnt_lag_1",
    "cnt_lag_24",
    "cnt_lag_48",
    "cnt_lag_168",
    "cnt_lag_336",
    "cnt_roll_7d",
    "cnt_roll_30d",
    "cnt_roll_7d_std",
    "cnt_roll_30d_std",
    "temp_diff_7d",
    "hum_diff_7d",
    "wind_diff_7d",
]

# Handle any remaining NaN in features (should be minimal after forward-fill)
for col in feature_cols:
    train_processed[col] = train_processed[col].fillna(0)
    val_processed[col] = val_processed[col].fillna(0)
    test_processed[col] = test_processed[col].fillna(test_processed[col].median())

missing = [c for c in feature_cols if c not in train_processed.columns]
if missing:
    print(f"WARNING: Missing features: {missing}")
    feature_cols = [c for c in feature_cols if c in train_processed.columns]

print(f"Number of features: {len(feature_cols)}")

# ============================================================
# 6. PREPARE MATRICES
# ============================================================
X_train = train_processed[feature_cols].values
y_train = np.log1p(train_processed["cnt"].values)

X_val = val_processed[feature_cols].values
y_val = np.log1p(val_processed["cnt"].values)

X_test = test_processed[feature_cols].values

# ============================================================
# 7. TRAIN MODEL 1: LightGBM with MAE objective
# ============================================================
print("\n--- Training Model 1 (MAE) ---")

params_mae = {
    "objective": "mae",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "learning_rate": 0.03,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42,
}

dtrain1 = lgb.Dataset(
    X_train, label=y_train, feature_name=feature_cols, free_raw_data=False
)
dval1 = lgb.Dataset(
    X_val,
    label=y_val,
    feature_name=feature_cols,
    free_raw_data=False,
    reference=dtrain1,
)

model1 = lgb.train(
    params_mae,
    dtrain1,
    num_boost_round=5000,
    valid_sets=[dval1],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)],
)

pred_val_1 = model1.predict(X_val)
pred_test_1 = model1.predict(X_test)
rmse_1 = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(pred_val_1)))
print(f"Model 1 (MAE) val RMSE: {rmse_1:.4f}, best_iteration: {model1.best_iteration}")

# ============================================================
# 8. TRAIN MODEL 2: LightGBM with MSE (regression) objective
# ============================================================
print("\n--- Training Model 2 (MSE) ---")

params_mse = {
    "objective": "regression",
    "metric": "l2",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "learning_rate": 0.03,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "lambda_l1": 0.0,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42,
}

dtrain2 = lgb.Dataset(
    X_train, label=y_train, feature_name=feature_cols, free_raw_data=False
)
dval2 = lgb.Dataset(
    X_val,
    label=y_val,
    feature_name=feature_cols,
    free_raw_data=False,
    reference=dtrain2,
)

model2 = lgb.train(
    params_mse,
    dtrain2,
    num_boost_round=5000,
    valid_sets=[dval2],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)],
)

pred_val_2 = model2.predict(X_val)
pred_test_2 = model2.predict(X_test)
rmse_2 = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(pred_val_2)))
print(f"Model 2 (MSE) val RMSE: {rmse_2:.4f}, best_iteration: {model2.best_iteration}")

# ============================================================
# 9. ENSEMBLE: Average of both models
# ============================================================
pred_val_ens = (pred_val_1 + pred_val_2) / 2.0
pred_test_ens = (pred_test_1 + pred_test_2) / 2.0

# Inverse transform and compute RMSE on original scale (official metric)
val_true_original = np.expm1(y_val)
val_pred_original = np.expm1(pred_val_ens)
val_pred_original = np.clip(val_pred_original, 0, None)

ensemble_rmse = np.sqrt(mean_squared_error(val_true_original, val_pred_original))
print(f"\nEnsemble val RMSE: {ensemble_rmse:.4f}")

# ============================================================
# 10. GENERATE SUBMISSION
# ============================================================
test_pred_original = np.expm1(pred_test_ens)
test_pred_original = np.clip(test_pred_original, 0, None)

os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame(
    {"id": range(len(test_pred_original)), "cnt": test_pred_original}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")
print(
    f"Prediction stats: mean={test_pred_original.mean():.1f}, min={test_pred_original.min():.1f}, max={test_pred_original.max():.1f}"
)

# ============================================================
# FINAL VALIDATION SCORE
# ============================================================
score = ensemble_rmse
print(f"Final Validation Score: {score}")