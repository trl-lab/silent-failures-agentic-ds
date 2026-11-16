import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import os

# Create directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")


# ============================================================
# 2. PARSE DATE AND CREATE TIME-BASED FEATURES
# ============================================================
def create_time_features(df):
    """Extract temporal features from date column."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d%H:%M:%S")
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Cyclical encoding for hour and day_of_week
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


train_df = create_time_features(train_df)
test_df = create_time_features(test_df)


# ============================================================
# 3. PHYSICAL/ENGINEERING FEATURES
# ============================================================
def create_engineering_features(df):
    """Create domain-specific engineering features for building energy."""
    df = df.copy()

    # Temperature differentials (indoor vs outdoor)
    indoor_temps = [f"T{i}" for i in range(1, 10)]
    for i in range(1, 10):
        df[f"T_diff_{i}"] = df[f"T{i}"] - df["T_out"]

    # Mean and spread of indoor temperatures
    df["T_mean_indoor"] = df[indoor_temps].mean(axis=1)
    df["T_std_indoor"] = df[indoor_temps].std(axis=1)
    df["T_max_indoor"] = df[indoor_temps].max(axis=1)
    df["T_min_indoor"] = df[indoor_temps].min(axis=1)
    df["T_range_indoor"] = df["T_max_indoor"] - df["T_min_indoor"]

    # Mean temperature differential
    df["T_diff_mean"] = df["T_mean_indoor"] - df["T_out"]

    # Humidity features
    rh_cols = [f"RH_{i}" for i in range(1, 10)]
    df["RH_mean_indoor"] = df[rh_cols].mean(axis=1)
    df["RH_std_indoor"] = df[rh_cols].std(axis=1)

    # Dew point spread
    df["dewpoint_spread"] = df["T_out"] - df["Tdewpoint"]

    # Temperature-humidity interaction
    df["T1_RH1"] = df["T1"] * df["RH_1"]
    df["T_out_RH_out"] = df["T_out"] * df["RH_out"]

    # Wind and visibility interactions
    df["wind_visibility"] = df["Windspeed"] * df["Visibility"]
    df["wind_T_out"] = df["Windspeed"] * df["T_out"]

    # rv1 and rv2 interactions (residual features)
    df["rv1_abs"] = np.abs(df["rv1"])
    df["rv2_abs"] = np.abs(df["rv2"])
    df["rv1_sq"] = df["rv1"] ** 2
    df["rv2_sq"] = df["rv2"] ** 2
    df["rv1_rv2"] = df["rv1"] * df["rv2"]
    df["rv_sum"] = df["rv1"] + df["rv2"]
    df["rv_diff"] = df["rv1"] - df["rv2"]

    # Lights interaction with time
    df["lights_hour"] = df["lights"] * df["hour_sin"]
    df["lights_dow"] = df["lights"] * df["dow_sin"]

    # Lights interaction with temperature differentials
    df["lights_T_diff_mean"] = df["lights"] * df["T_diff_mean"]
    df["lights_T_diff_1"] = df["lights"] * df["T_diff_1"]
    df["lights_T_diff_5"] = df["lights"] * df["T_diff_5"]
    df["lights_T_range"] = df["lights"] * df["T_range_indoor"]

    return df


train_df = create_engineering_features(train_df)
test_df = create_engineering_features(test_df)

# ============================================================
# 4. TIME-BASED SPLIT (respect temporal ordering)
# ============================================================
train_df = train_df.sort_values("date").reset_index(drop=True)

split_idx = int(len(train_df) * 0.85)
train_split = train_df.iloc[:split_idx].reset_index(drop=True)
val_split = train_df.iloc[split_idx:].reset_index(drop=True)

print(f"Train split: {len(train_split)} samples")
print(f"Validation split: {len(val_split)} samples")
print(f"Train date range: {train_split['date'].min()} to {train_split['date'].max()}")
print(f"Val date range: {val_split['date'].min()} to {val_split['date'].max()}")

# ============================================================
# 5. ADD LAG AND ROLLING FEATURES (no future leakage)
# ============================================================
# Extended lag windows to capture different temporal scales
LAG_WINDOWS = [1, 2, 3, 4, 6, 10, 12, 14, 24, 36, 48, 60, 144]
ROLL_WINDOWS = [12, 24]

# Build a continuous Appliances series aligned with the full timeline
full_appliances = train_df["Appliances"].values
full_dates = train_df["date"].values

# For train_split: lags are straightforward (shift within the series)
for lag in LAG_WINDOWS:
    train_split[f"Appliances_lag_{lag}"] = train_split["Appliances"].shift(lag).values

# For val_split: we need to handle the boundary
# The first `lag` values of val_split should use the tail of train_split
val_appliances = val_split["Appliances"].values
train_appliances_tail = train_split["Appliances"].values

for lag in LAG_WINDOWS:
    if lag <= len(train_appliances_tail):
        # Pad with tail of train
        pad_values = train_appliances_tail[-lag:]
        val_lag = np.concatenate([pad_values, val_appliances[:-lag]]) if lag < len(val_appliances) else train_appliances_tail[-lag:]
    else:
        val_lag = np.full(len(val_appliances), np.nan)
    val_split[f"Appliances_lag_{lag}"] = val_lag

# For test: we will fill lag/rolling features later via recursive prediction
# Initialize with NaN - will be filled in the recursive prediction loop
for lag in LAG_WINDOWS:
    test_df[f"Appliances_lag_{lag}"] = np.nan

for window in ROLL_WINDOWS:
    test_df[f"Appliances_roll_mean_{window}"] = np.nan
    test_df[f"Appliances_roll_std_{window}"] = np.nan

# Rolling mean and std features (window=12, 24)
for window in ROLL_WINDOWS:
    # Train
    train_split[f"Appliances_roll_mean_{window}"] = train_split["Appliances"].rolling(window, min_periods=1).mean().values
    train_split[f"Appliances_roll_std_{window}"] = train_split["Appliances"].rolling(window, min_periods=1).std().fillna(0).values

    # Val: use combined tail of train + val for rolling
    pad_len = window - 1
    combined = np.concatenate([train_appliances_tail[-pad_len:], val_appliances])
    roll_mean_full = pd.Series(combined).rolling(window, min_periods=1).mean().values
    roll_std_full = pd.Series(combined).rolling(window, min_periods=1).std().fillna(0).values
    # Take only the last len(val_appliances) values (corresponding to val rows)
    val_split[f"Appliances_roll_mean_{window}"] = roll_mean_full[-len(val_appliances):]
    val_split[f"Appliances_roll_std_{window}"] = roll_std_full[-len(val_appliances):]

# ============================================================
# 6. DEFINE FEATURE COLUMNS
# ============================================================
drop_cols = ["date", "Appliances"]
feature_cols = [c for c in train_split.columns if c not in drop_cols]
print(f"Number of features: {len(feature_cols)}")

# ============================================================
# 7. HANDLE ANY REMAINING NaN/INF VALUES (train and val only)
# ============================================================
for df in [train_split, val_split]:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[feature_cols] = df[feature_cols].fillna(0)

# ============================================================
# 8. PREPARE FEATURES AND TARGET
# ============================================================
X_train = train_split[feature_cols].values
y_train = train_split["Appliances"].values
X_val = val_split[feature_cols].values
y_val = val_split["Appliances"].values

# Test features will be filled row-by-row during recursive prediction
# (lag/rolling features are computed on the fly)

# ============================================================
# 9. TRAIN ENSEMBLE: LGB(base) + LGB(deep) + XGBoost + CatBoost
# ============================================================

# --- LightGBM (base) ---
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": 7,
    "learning_rate": 0.005,
    "n_estimators": 20000,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.01,
    "reg_lambda": 0.1,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

lgb_model = lgb.LGBMRegressor(**lgb_params)
early_stop_lgb = lgb.early_stopping(stopping_rounds=300, verbose=False)

lgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[early_stop_lgb],
)

print(f"LightGBM(base) best iteration: {lgb_model.best_iteration_}")
lgb_val_preds = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration_)
lgb_val_rmse = root_mean_squared_error(y_val, lgb_val_preds)
print(f"LightGBM(base) Validation RMSE: {lgb_val_rmse:.4f}")

# --- LightGBM (deep) ---
lgb_deep_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "max_depth": 9,
    "learning_rate": 0.005,
    "n_estimators": 20000,
    "min_child_samples": 30,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.05,
    "reg_lambda": 0.5,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

lgb_deep_model = lgb.LGBMRegressor(**lgb_deep_params)
early_stop_lgb_deep = lgb.early_stopping(stopping_rounds=300, verbose=False)

lgb_deep_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[early_stop_lgb_deep],
)

print(f"LightGBM(deep) best iteration: {lgb_deep_model.best_iteration_}")
lgb_deep_val_preds = lgb_deep_model.predict(X_val, num_iteration=lgb_deep_model.best_iteration_)
lgb_deep_val_rmse = root_mean_squared_error(y_val, lgb_deep_val_preds)
print(f"LightGBM(deep) Validation RMSE: {lgb_deep_val_rmse:.4f}")

# --- XGBoost ---
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    max_depth=6,
    learning_rate=0.005,
    n_estimators=20000,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=50,
    reg_alpha=0.01,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    early_stopping_rounds=300,
)

xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

print(f"XGBoost best iteration: {xgb_model.best_iteration}")
xgb_val_preds = xgb_model.predict(X_val, iteration_range=(0, xgb_model.best_iteration + 1))
xgb_val_rmse = root_mean_squared_error(y_val, xgb_val_preds)
print(f"XGBoost Validation RMSE: {xgb_val_rmse:.4f}")

# --- CatBoost ---
cat_model = CatBoostRegressor(
    iterations=20000,
    learning_rate=0.005,
    depth=6,
    loss_function="RMSE",
    random_seed=42,
    verbose=0,
    l2_leaf_reg=3,
)

cat_model.fit(
    X_train,
    y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=300,
    verbose=0,
)

cat_best_iter = cat_model.get_best_iteration()
print(f"CatBoost best iteration: {cat_best_iter}")
cat_val_preds = cat_model.predict(X_val)
cat_val_rmse = root_mean_squared_error(y_val, cat_val_preds)
print(f"CatBoost Validation RMSE: {cat_val_rmse:.4f}")

# ============================================================
# 10. OPTIMAL ENSEMBLE WEIGHTS VIA GRID SEARCH
# ============================================================
val_preds_array = np.column_stack([lgb_val_preds, lgb_deep_val_preds, xgb_val_preds, cat_val_preds])
n_models = 4

# Grid search over weight combinations (each weight in 0.05-0.5, step 0.05)
best_rmse = np.inf
best_weights = np.array([0.25, 0.25, 0.25, 0.25])

weight_options = np.arange(0.05, 0.51, 0.05)

for w1 in weight_options:
    for w2 in weight_options:
        for w3 in weight_options:
            w4 = 1.0 - w1 - w2 - w3
            if w4 < 0.05 or w4 > 0.5:
                continue
            w = np.array([w1, w2, w3, w4])
            ensemble_pred = val_preds_array @ w
            rmse = root_mean_squared_error(y_val, ensemble_pred)
            if rmse < best_rmse:
                best_rmse = rmse
                best_weights = w

print(f"\nOptimal ensemble weights: LGB={best_weights[0]:.2f}, LGB_deep={best_weights[1]:.2f}, XGB={best_weights[2]:.2f}, CAT={best_weights[3]:.2f}")
ensemble_val_preds = val_preds_array @ best_weights
ensemble_val_rmse = root_mean_squared_error(y_val, ensemble_val_preds)
print(f"Ensemble Validation RMSE: {ensemble_val_rmse:.4f}")

# ============================================================
# 11. TEST PREDICTIONS (recursive prediction with lag/rolling features)
# ============================================================
# Recursive prediction: for each test sample, compute lag/rolling features
# using the model's own previous predictions as pseudo-targets

# Get the base test features (without lag/rolling columns)
lag_roll_cols = [c for c in feature_cols if "Appliances_lag" in c or "Appliances_roll" in c]
non_lag_roll_cols = [c for c in feature_cols if "Appliances_lag" not in c and "Appliances_roll" not in c]

# Build the non-lag/rolling part of test features (these are static per row)
X_test_base = test_df[non_lag_roll_cols].values

# Buffer of recent predictions for computing lags and rolling stats
# Initialize with the last values from training
buffer = list(full_appliances[-max(LAG_WINDOWS):])

test_preds = np.zeros(len(test_df))

for i in range(len(test_df)):
    # Compute lag features from buffer
    lag_values = {}
    for lag in LAG_WINDOWS:
        if len(buffer) >= lag:
            lag_values[f"Appliances_lag_{lag}"] = buffer[-lag]
        else:
            lag_values[f"Appliances_lag_{lag}"] = buffer[0] if len(buffer) > 0 else 0.0

    # Compute rolling features from buffer
    roll_values = {}
    for window in ROLL_WINDOWS:
        if len(buffer) >= 1:
            recent = buffer[-window:] if len(buffer) >= window else buffer
            roll_values[f"Appliances_roll_mean_{window}"] = np.mean(recent)
            roll_values[f"Appliances_roll_std_{window}"] = np.std(recent) if len(recent) > 1 else 0.0
        else:
            roll_values[f"Appliances_roll_mean_{window}"] = 0.0
            roll_values[f"Appliances_roll_std_{window}"] = 0.0

    # Build the full feature row for this test sample
    # Combine non-lag/roll features with computed lag/roll features
    row_non_lag = X_test_base[i]
    row_lag_roll = np.array([lag_values.get(c, 0.0) if "lag" in c else roll_values.get(c, 0.0) for c in lag_roll_cols])

    # Reconstruct full feature vector in correct order
    X_row = np.zeros(len(feature_cols))
    for j, col in enumerate(feature_cols):
        if col in non_lag_roll_cols:
            X_row[j] = row_non_lag[non_lag_roll_cols.index(col)]
        else:
            X_row[j] = lag_values.get(col, 0.0) if "lag" in col else roll_values.get(col, 0.0)

    # Predict using all 4 models
    pred_lgb = lgb_model.predict(X_row.reshape(1, -1), num_iteration=lgb_model.best_iteration_)[0]
    pred_lgb_deep = lgb_deep_model.predict(X_row.reshape(1, -1), num_iteration=lgb_deep_model.best_iteration_)[0]
    pred_xgb = xgb_model.predict(X_row.reshape(1, -1), iteration_range=(0, xgb_model.best_iteration + 1))[0]
    pred_cat = cat_model.predict(X_row.reshape(1, -1))[0]

    # Weighted ensemble prediction
    test_preds[i] = (
        best_weights[0] * pred_lgb
        + best_weights[1] * pred_lgb_deep
        + best_weights[2] * pred_xgb
        + best_weights[3] * pred_cat
    )

    # Update buffer with this prediction for next iteration
    buffer.append(test_preds[i])

# ============================================================
# 12. SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame({"id": range(len(test_preds)), "Appliances": test_preds})
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")
print(
    f"Prediction stats: mean={test_preds.mean():.2f}, std={test_preds.std():.2f}, min={test_preds.min():.2f}, max={test_preds.max():.2f}"
)

# ============================================================
# 13. FEATURE IMPORTANCE (top 10)
# ============================================================
importance = pd.Series(lgb_model.feature_importances_, index=feature_cols).sort_values(
    ascending=False
)
print("\nTop 10 features (LightGBM):")
print(importance.head(10).to_string())

# ============================================================
# 14. FINAL SCORE (official metric: RMSE)
# ============================================================
score = ensemble_val_rmse
print(f"Final Validation Score: {score}")