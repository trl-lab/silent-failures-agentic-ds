import pandas as pd
import numpy as np
import os
import lightgbm as lgb

# ============================================================
# SETUP
# ============================================================
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============================================================
# STEP 1: DATA LOADING AND FEATURE ENGINEERING
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Parse datetime
train["datetime"] = pd.to_datetime(train["datetime"])
test["datetime"] = pd.to_datetime(test["datetime"])

# Sort by datetime to ensure temporal ordering
train = train.sort_values("datetime").reset_index(drop=True)
test = test.sort_values("datetime").reset_index(drop=True)

print(f"Train datetime range: {train['datetime'].min()} to {train['datetime'].max()}")
print(f"Test datetime range: {test['datetime'].min()} to {test['datetime'].max()}")

# Time-based split: last 15% as validation
split_idx = int(len(train) * 0.85)
train_split = train.iloc[:split_idx].copy()
val_split = train.iloc[split_idx:].copy()

print(f"Train split: {len(train_split)} samples")
print(f"Validation split: {len(val_split)} samples")


# Feature engineering function
def engineer_features(df):
    """Engineer features from raw data."""
    df = df.copy()

    # Extract temporal features
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["year"] = df["datetime"].dt.year
    df["weekofyear"] = df["datetime"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["datetime"].dt.quarter
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["dayofyear"] = df["datetime"].dt.dayofyear

    # Cyclic encoding for periodic features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

    # Temperature features
    df["temp_normalized"] = df["temp"] / 50.0
    df["atemp_normalized"] = df["atemp"] / 50.0
    df["temp_diff"] = df["atemp"] - df["temp"]
    df["temp_humidity"] = df["temp"] * df["humidity"] / 100.0
    df["apparent_temp"] = df["atemp"] - 0.55 * (1 - df["humidity"] / 100.0) * (
        df["atemp"] - 10
    )

    # Weather interactions
    df["weather_temp"] = df["weather"] * df["temp"]
    df["weather_hour"] = df["weather"] * df["hour"]
    df["weather_season"] = df["weather"] * df["season"]
    df["bad_weather"] = (df["weather"] >= 3).astype(int)
    df["good_weather"] = (df["weather"] == 1).astype(int)

    # Season interactions
    df["season_hour"] = df["season"] * df["hour"]
    df["season_temp"] = df["season"] * df["temp"]
    df["season_dow"] = df["season"] * df["dayofweek"]

    # Working day interactions
    df["workingday_hour"] = df["workingday"] * df["hour"]
    df["workingday_dow"] = df["workingday"] * df["dayofweek"]
    df["holiday_workingday"] = df["holiday"] * df["workingday"]

    # Rush hour indicators
    df["morning_rush"] = ((df["hour"] >= 7) & (df["hour"] <= 9)).astype(int)
    df["evening_rush"] = ((df["hour"] >= 16) & (df["hour"] <= 19)).astype(int)
    df["midnight"] = ((df["hour"] >= 23) | (df["hour"] <= 5)).astype(int)
    df["afternoon"] = ((df["hour"] >= 12) & (df["hour"] <= 15)).astype(int)
    df["night"] = ((df["hour"] >= 20) | (df["hour"] <= 5)).astype(int)

    # Wind features
    df["wind_temp"] = df["windspeed"] * df["temp"]
    df["wind_chill"] = df["windspeed"] * (df["atemp"] - 10) / 10.0
    df["high_wind"] = (df["windspeed"] > 20).astype(int)

    # NOTE: casual and registered are components of count (count = casual + registered)
    # Using them as features would be data leakage, so we do NOT derive features from them.
    df["casual_ratio"] = 0.0
    df["registered_ratio"] = 0.0
    df["log_casual"] = 0.0
    df["log_registered"] = 0.0

    # Hour-based features for different day types
    df["hour_x_weekend"] = df["hour"] * df["is_weekend"]
    df["hour_x_workingday"] = df["hour"] * df["workingday"]

    # Temperature comfort zone
    df["temp_comfort"] = np.exp(-((df["temp"] - 22) ** 2) / (2 * 8**2))
    df["humidity_comfort"] = np.exp(-((df["humidity"] - 50) ** 2) / (2 * 25**2))

    # Drop original datetime column
    df = df.drop(columns=["datetime"])

    return df


# Apply feature engineering
train_feat = engineer_features(train_split)
val_feat = engineer_features(val_split)
test_feat = engineer_features(test)

# Define feature columns (exclude target)
target = "count"
feature_cols = [c for c in train_feat.columns if c != target]
print(f"Number of features: {len(feature_cols)}")

# Prepare target in log-space (since metric is RMSLE)
train_feat["log_count"] = np.log1p(train_feat[target])
val_feat["log_count"] = np.log1p(val_feat[target])

# Fill any NaN values
for df_feat in [train_feat, val_feat, test_feat]:
    nan_counts = df_feat[feature_cols].isna().sum()
    if nan_counts.sum() > 0:
        df_feat[feature_cols] = df_feat[feature_cols].fillna(0)

print(f"Train features: {train_feat.shape[1]} columns")
print(f"Val features: {val_feat.shape[1]} columns")
print(f"Test features: {test_feat.shape[1]} columns")

# ============================================================
# STEP 2: MODEL DEFINITION (LightGBM Ensemble)
# ============================================================

# Define 3 diverse LightGBM model configurations
lgb_configs = [
    {
        "objective": "regression",
        "num_leaves": 63,
        "learning_rate": 0.01,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    },
    {
        "objective": "regression",
        "num_leaves": 31,
        "learning_rate": 0.005,
        "min_child_samples": 30,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "verbose": -1,
    },
    {
        "objective": "regression",
        "num_leaves": 127,
        "learning_rate": 0.02,
        "min_child_samples": 15,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.75,
        "bagging_freq": 5,
        "verbose": -1,
    },
]

N_ESTIMATORS = 5000
EARLY_STOPPING_ROUNDS = 50

print(f"LightGBM ensemble: {len(lgb_configs)} models, n_estimators={N_ESTIMATORS}, early_stopping={EARLY_STOPPING_ROUNDS}")

# ============================================================
# STEP 3: TRAINING AND EVALUATION (LightGBM)
# ============================================================

# Prepare numpy arrays
X_train = train_feat[feature_cols].values.astype(np.float64)
y_train = train_feat["log_count"].values.astype(np.float64)
X_val = val_feat[feature_cols].values.astype(np.float64)
y_val = val_feat["log_count"].values.astype(np.float64)
X_test = test_feat[feature_cols].values.astype(np.float64)

# Store original counts for RMSLE computation
val_counts = val_feat["count"].values.astype(np.float64)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"X_test: {X_test.shape}")


def compute_rmsle(y_true, y_pred):
    """Compute RMSLE: sqrt(mean((log1p(pred) - log1p(actual))^2))"""
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


# ============================================================
# PHASE 1: Training with early stopping
# ============================================================
print("\n" + "=" * 60)
print("PHASE 1: Training LightGBM ensemble with validation split")
print("=" * 60)

best_iterations = []
val_preds_log = np.zeros(len(X_val))

for i, params in enumerate(lgb_configs):
    print(f"\nTraining model {i+1}/{len(lgb_configs)}: num_leaves={params['num_leaves']}, lr={params['learning_rate']}")

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, free_raw_data=False)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=N_ESTIMATORS,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    best_iter = model.best_iteration
    best_iterations.append(best_iter)

    # Get validation predictions (in log-space since we trained on log1p(target))
    val_pred = model.predict(X_val, num_iteration=best_iter)
    val_preds_log += val_pred

    # Compute per-model RMSLE
    model_rmsle = np.sqrt(np.mean((val_pred - y_val) ** 2))
    print(f"  Best iteration: {best_iter}, Val RMSLE (log-space): {model_rmsle:.6f}")

# Average ensemble predictions in log-space
val_preds_log /= len(lgb_configs)

# Compute ensemble RMSLE
rmsle_val = np.sqrt(np.mean((val_preds_log - y_val) ** 2))
print(f"\nEnsemble Val RMSLE (Phase 1): {rmsle_val:.6f}")
print(f"Best iterations: {best_iterations}")

# ============================================================
# PHASE 2: Retrain on FULL training data for best iterations
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: Retraining on full data with best iterations")
print("=" * 60)

X_full = np.vstack([X_train, X_val])
y_full = np.concatenate([y_train, y_val])

full_models = []
for i, params in enumerate(lgb_configs):
    print(f"Retraining model {i+1}/{len(lgb_configs)} for {best_iterations[i]} iterations...")

    dtrain_full = lgb.Dataset(X_full, label=y_full, feature_name=feature_cols, free_raw_data=False)

    model = lgb.train(
        params,
        dtrain_full,
        num_boost_round=best_iterations[i],
    )
    full_models.append(model)
    print(f"  Model {i+1} retrained.")

# ============================================================
# VALIDATION SCORE (using Phase 1 ensemble for honest evaluation)
# ============================================================
print(f"\nValidation RMSLE (Phase 1 ensemble): {rmsle_val:.6f}")

# ============================================================
# TEST INFERENCE (using full-data models)
# ============================================================
test_pred_log = np.zeros(len(X_test))
for model in full_models:
    test_pred_log += model.predict(X_test)

# Average in log-space
test_pred_log /= len(full_models)

# Convert from log-space to count-space
test_pred_count = np.expm1(test_pred_log)
test_pred_count = np.clip(test_pred_count, 0, None)

# ============================================================
# SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame(
    {"id": np.arange(len(test_pred_count)), "count": test_pred_count}
)
submission.to_csv("./submission/submission.csv", index=False)

print(f"\nSubmission saved: {submission.shape}")
print(
    f"Predicted count range: [{test_pred_count.min():.2f}, {test_pred_count.max():.2f}]"
)
print(f"Predicted count mean: {test_pred_count.mean():.2f}")

# ============================================================
# FINAL SCORE
# ============================================================
score = rmsle_val
print(f"\n{'='*60}")
print(f"SUMMARY:")
print(f"  Phase 1 Ensemble Val RMSLE: {rmsle_val:.6f}")
print(f"  Best iterations per model: {best_iterations}")
print(f"  Phase 2: Full retrain with best iterations")
print(f"  Test predictions: {len(test_pred_count)} samples")
print(f"{'='*60}")
print(f"Final Validation Score: {score}")