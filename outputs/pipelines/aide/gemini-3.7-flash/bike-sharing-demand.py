import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")


# 2. Feature Engineering
def extract_features(df):
    df = df.copy()
    dt = pd.to_datetime(df["datetime"])

    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["dayofweek"] = dt.dt.dayofweek
    df["hour"] = dt.dt.hour
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Continuous time trend across dataset (captures growth over time)
    df["time_index"] = (
        (df["year"] - 2011) * 12 + df["month"] + df["day"] / 31.0 + df["hour"] / 744.0
    )

    # Cyclical representations
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)

    # Composite behavioral features
    df["workingday_hour"] = df["workingday"] * 24 + df["hour"]
    df["is_commute_morning"] = (
        (df["workingday"] == 1) & (df["hour"].between(7, 9))
    ).astype(int)
    df["is_commute_evening"] = (
        (df["workingday"] == 1) & (df["hour"].between(16, 19))
    ).astype(int)
    df["is_casual_peak"] = (
        (df["workingday"] == 0) & (df["hour"].between(10, 18))
    ).astype(int)

    # Meteorological interactions & indicators
    df["temp_diff"] = df["atemp"] - df["temp"]
    df["humidity_temp_ratio"] = df["humidity"] / (df["temp"] + 1.0)
    df["windspeed_humidity_prod"] = df["windspeed"] * df["humidity"]
    df["apparent_comfort"] = df["atemp"] - 0.2 * df["humidity"] - 0.1 * df["windspeed"]
    df["dew_point"] = df["temp"] - ((100.0 - df["humidity"]) / 5.0)
    df["is_zero_wind"] = (df["windspeed"] == 0).astype(int)
    df["is_bad_weather"] = (df["weather"] >= 3).astype(int)

    return df


train_features = extract_features(train_df)
test_features = extract_features(test_df)

feature_cols = [
    "season",
    "holiday",
    "workingday",
    "weather",
    "temp",
    "atemp",
    "humidity",
    "windspeed",
    "year",
    "month",
    "day",
    "dayofweek",
    "hour",
    "is_weekend",
    "time_index",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "workingday_hour",
    "is_commute_morning",
    "is_commute_evening",
    "is_casual_peak",
    "temp_diff",
    "humidity_temp_ratio",
    "windspeed_humidity_prod",
    "apparent_comfort",
    "dew_point",
    "is_zero_wind",
    "is_bad_weather",
]

X = train_features[feature_cols]
X_test = test_features[feature_cols]

# Target transformations: log(1 + y)
y_casual = np.log1p(train_df["casual"].values)
y_registered = np.log1p(train_df["registered"].values)
y_true_count = train_df["count"].values

# 3. 5-Fold Cross Validation Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_casual = np.zeros(len(train_df))
oof_registered = np.zeros(len(train_df))
test_casual_preds = np.zeros(len(test_df))
test_registered_preds = np.zeros(len(test_df))

cb_params = {
    "iterations": 1500,
    "learning_rate": 0.04,
    "depth": 6,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "verbose": False,
}

lgb_params = {
    "n_estimators": 1500,
    "learning_rate": 0.04,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, y_tr_cas, y_tr_reg = (
        X.iloc[train_idx],
        y_casual[train_idx],
        y_registered[train_idx],
    )
    X_va, y_va_cas, y_va_reg = X.iloc[val_idx], y_casual[val_idx], y_registered[val_idx]

    # --- Casual Models ---
    # CatBoost
    cb_cas = CatBoostRegressor(**cb_params)
    cb_cas.fit(
        X_tr,
        y_tr_cas,
        eval_set=(X_va, y_va_cas),
        early_stopping_rounds=50,
        verbose=False,
    )
    cb_val_cas = cb_cas.predict(X_va)
    cb_test_cas = cb_cas.predict(X_test)

    # LightGBM
    lgb_cas = lgb.LGBMRegressor(**lgb_params)
    lgb_cas.fit(
        X_tr,
        y_tr_cas,
        eval_set=[(X_va, y_va_cas)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    lgb_val_cas = lgb_cas.predict(X_va)
    lgb_test_cas = lgb_cas.predict(X_test)

    # Ensemble Casual (50/50 blend)
    oof_casual[val_idx] = 0.5 * cb_val_cas + 0.5 * lgb_val_cas
    test_casual_preds += (0.5 * cb_test_cas + 0.5 * lgb_test_cas) / kf.n_splits

    # --- Registered Models ---
    # CatBoost
    cb_reg = CatBoostRegressor(**cb_params)
    cb_reg.fit(
        X_tr,
        y_tr_reg,
        eval_set=(X_va, y_va_reg),
        early_stopping_rounds=50,
        verbose=False,
    )
    cb_val_reg = cb_reg.predict(X_va)
    cb_test_reg = cb_reg.predict(X_test)

    # LightGBM
    lgb_reg = lgb.LGBMRegressor(**lgb_params)
    lgb_reg.fit(
        X_tr,
        y_tr_reg,
        eval_set=[(X_va, y_va_reg)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    lgb_val_reg = lgb_reg.predict(X_va)
    lgb_test_reg = lgb_reg.predict(X_test)

    # Ensemble Registered (50/50 blend)
    oof_registered[val_idx] = 0.5 * cb_val_reg + 0.5 * lgb_val_reg
    test_registered_preds += (0.5 * cb_test_reg + 0.5 * lgb_test_reg) / kf.n_splits

# 4. Reconstruct count predictions: count = casual + registered
oof_casual_counts = np.clip(np.expm1(oof_casual), 0, None)
oof_registered_counts = np.clip(np.expm1(oof_registered), 0, None)
oof_total_counts = oof_casual_counts + oof_registered_counts

# Calculate Overall RMSLE
cv_rmsle = np.sqrt(
    mean_squared_error(np.log1p(y_true_count), np.log1p(oof_total_counts))
)
print(f"5-Fold CV RMSLE: {cv_rmsle:.5f}")

# 5. Generate Test Predictions and Save Submission
test_casual_counts = np.clip(np.expm1(test_casual_preds), 0, None)
test_registered_counts = np.clip(np.expm1(test_registered_preds), 0, None)
test_total_counts = test_casual_counts + test_registered_counts

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_df)), "count": test_total_counts})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
