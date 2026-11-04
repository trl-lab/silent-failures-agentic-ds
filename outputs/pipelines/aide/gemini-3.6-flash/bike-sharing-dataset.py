import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor

    HAS_CATBOOST = False

# 1. Ensure working directory exists
os.makedirs("./working", exist_ok=True)

# 2. Load datasets
train_raw = pd.read_parquet("./input/train.parquet")
test_raw = pd.read_parquet("./input/test.parquet")


# 3. Feature Engineering Pipeline
def process_features(df):
    data = df.copy()
    data["dteday"] = pd.to_datetime(data["dteday"])

    # Calendar & Date features
    data["year"] = data["dteday"].dt.year
    data["month"] = data["dteday"].dt.month
    data["day"] = data["dteday"].dt.day
    data["dayofweek"] = data["dteday"].dt.dayofweek
    data["dayofyear"] = data["dteday"].dt.dayofyear
    data["quarter"] = data["dteday"].dt.quarter
    data["is_weekend"] = (data["dayofweek"] >= 5).astype(int)

    # Cyclical Transformations
    data["hr_sin"] = np.sin(2 * np.pi * data["hr"] / 24.0)
    data["hr_cos"] = np.cos(2 * np.pi * data["hr"] / 24.0)
    data["mnth_sin"] = np.sin(2 * np.pi * data["mnth"] / 12.0)
    data["mnth_cos"] = np.cos(2 * np.pi * data["mnth"] / 12.0)
    data["weekday_sin"] = np.sin(2 * np.pi * data["weekday"] / 7.0)
    data["weekday_cos"] = np.cos(2 * np.pi * data["weekday"] / 7.0)
    data["dayofyear_sin"] = np.sin(2 * np.pi * data["dayofyear"] / 365.25)
    data["dayofyear_cos"] = np.cos(2 * np.pi * data["dayofyear"] / 365.25)

    # Weather & Physical Interactions
    data["temp_diff"] = data["atemp"] - data["temp"]
    data["temp_hum_ratio"] = data["temp"] / (data["hum"] + 0.01)
    data["temp_hum_prod"] = data["temp"] * data["hum"]
    data["wind_temp_prod"] = data["windspeed"] * data["temp"]
    data["wind_hum_prod"] = data["windspeed"] * data["hum"]

    # Daily Weather Context Aggregations
    daily_stats = data.groupby("dteday").agg(
        temp_daily_mean=("temp", "mean"),
        temp_daily_min=("temp", "min"),
        temp_daily_max=("temp", "max"),
        hum_daily_mean=("hum", "mean"),
        wind_daily_mean=("windspeed", "mean"),
    )
    daily_stats["temp_daily_range"] = (
        daily_stats["temp_daily_max"] - daily_stats["temp_daily_min"]
    )
    data = data.merge(daily_stats, on="dteday", how="left")

    data["temp_rel_daily"] = data["temp"] - data["temp_daily_mean"]
    data["hum_rel_daily"] = data["hum"] - data["hum_daily_mean"]

    # Peak Period Indicators
    data["is_morning_rush"] = (
        (data["hr"] >= 7) & (data["hr"] <= 9) & (data["workingday"] == 1)
    ).astype(int)
    data["is_evening_rush"] = (
        (data["hr"] >= 17) & (data["hr"] <= 19) & (data["workingday"] == 1)
    ).astype(int)
    data["is_weekend_peak"] = (
        (data["hr"] >= 11) & (data["hr"] <= 16) & (data["workingday"] == 0)
    ).astype(int)

    return data


train_df = process_features(train_raw)
test_df = process_features(test_raw)

# Group aggregations across hourly profile
all_data = pd.concat([train_df, test_df], axis=0, ignore_index=True)
grp_cols = ["hr", "workingday"]
agg_stats = all_data.groupby(grp_cols)[["temp", "hum", "windspeed"]].transform("mean")
agg_stats.columns = [f"{c}_grp_mean" for c in agg_stats.columns]

train_df = pd.concat([train_df, agg_stats.iloc[: len(train_df)]], axis=1)
test_df = pd.concat(
    [test_df, agg_stats.iloc[len(train_df) :].reset_index(drop=True)], axis=1
)

drop_cols = ["instant", "dteday", "casual", "registered", "cnt"]
feature_cols = [col for col in train_df.columns if col not in drop_cols]

X = train_df[feature_cols].copy()
y = train_df["cnt"].values
y_log = np.log1p(y)
X_test = test_df[feature_cols].copy()

# 4. Model Training & Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

lgb_params = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": -1,
    "verbose": -1,
}

xgb_params = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": -1,
    "early_stopping_rounds": 50,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
    y_tr_log, y_va_log = y_log[train_idx], y_log[val_idx]
    y_tr_raw, y_va_raw = y[train_idx], y[val_idx]

    # --- Models on Log Target ---
    # LightGBM Log
    lgb_log = lgb.LGBMRegressor(**lgb_params, random_state=42 + fold)
    lgb_log.fit(
        X_tr,
        y_tr_log,
        eval_set=[(X_va, y_va_log)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # XGBoost Log
    xgb_log = xgb.XGBRegressor(**xgb_params, random_state=42 + fold)
    xgb_log.fit(X_tr, y_tr_log, eval_set=[(X_va, y_va_log)], verbose=False)

    # --- Models on Direct Target ---
    # LightGBM Raw
    lgb_raw = lgb.LGBMRegressor(**lgb_params, random_state=100 + fold)
    lgb_raw.fit(
        X_tr,
        y_tr_raw,
        eval_set=[(X_va, y_va_raw)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # XGBoost Raw
    xgb_raw = xgb.XGBRegressor(**xgb_params, random_state=100 + fold)
    xgb_raw.fit(X_tr, y_tr_raw, eval_set=[(X_va, y_va_raw)], verbose=False)

    # CatBoost / HistGradientBoosting
    if HAS_CATBOOST:
        cat_log = CatBoostRegressor(
            iterations=2000,
            learning_rate=0.03,
            depth=7,
            verbose=0,
            random_seed=42 + fold,
            early_stopping_rounds=50,
        )
        cat_log.fit(X_tr, y_tr_log, eval_set=(X_va, y_va_log))

        cat_raw = CatBoostRegressor(
            iterations=2000,
            learning_rate=0.03,
            depth=7,
            verbose=0,
            random_seed=100 + fold,
            early_stopping_rounds=50,
        )
        cat_raw.fit(X_tr, y_tr_raw, eval_set=(X_va, y_va_raw))

        val_log_pred = (
            lgb_log.predict(X_va) + xgb_log.predict(X_va) + cat_log.predict(X_va)
        ) / 3.0
        val_raw_pred = (
            lgb_raw.predict(X_va) + xgb_raw.predict(X_va) + cat_raw.predict(X_va)
        ) / 3.0

        t_log_pred = (
            lgb_log.predict(X_test) + xgb_log.predict(X_test) + cat_log.predict(X_test)
        ) / 3.0
        t_raw_pred = (
            lgb_raw.predict(X_test) + xgb_raw.predict(X_test) + cat_raw.predict(X_test)
        ) / 3.0
    else:
        hgb_log = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.03, random_state=42 + fold
        )
        hgb_log.fit(X_tr, y_tr_log)

        hgb_raw = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.03, random_state=100 + fold
        )
        hgb_raw.fit(X_tr, y_tr_raw)

        val_log_pred = (
            lgb_log.predict(X_va) + xgb_log.predict(X_va) + hgb_log.predict(X_va)
        ) / 3.0
        val_raw_pred = (
            lgb_raw.predict(X_va) + xgb_raw.predict(X_va) + hgb_raw.predict(X_va)
        ) / 3.0

        t_log_pred = (
            lgb_log.predict(X_test) + xgb_log.predict(X_test) + hgb_log.predict(X_test)
        ) / 3.0
        t_raw_pred = (
            lgb_raw.predict(X_test) + xgb_raw.predict(X_test) + hgb_raw.predict(X_test)
        ) / 3.0

    # Combine log-scale and raw-scale predictions
    fold_val_pred = 0.5 * np.expm1(val_log_pred) + 0.5 * val_raw_pred
    oof_preds[val_idx] = np.clip(fold_val_pred, 0, None)

    fold_test_pred = 0.5 * np.expm1(t_log_pred) + 0.5 * t_raw_pred
    test_preds += np.clip(fold_test_pred, 0, None) / kf.n_splits

# 5. Evaluate overall CV RMSE
cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"Validation RMSE: {cv_rmse:.4f}")

# 6. Save final submission
submission = pd.DataFrame({"id": np.arange(len(test_df)), "cnt": test_preds})
submission.to_csv("./working/submission.csv", index=False)
