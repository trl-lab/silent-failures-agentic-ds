import os
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import xgboost as xgb


def calculate_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate exact Root Mean Squared Logarithmic Error (RMSLE)."""
    y_true_clipped = np.clip(y_true, 0, None)
    y_pred_clipped = np.clip(y_pred, 0, None)
    return float(
        np.sqrt(np.mean((np.log1p(y_pred_clipped) - np.log1p(y_true_clipped)) ** 2))
    )


def extract_features(
    df: pd.DataFrame, train_stats: dict = None
) -> tuple[pd.DataFrame, dict]:
    """Extract comprehensive temporal, meteorological, and interaction features

    from raw bike sharing data while strictly isolating statistics from training
    set.
    """
    df = df.copy()

    # Ensure datetime format
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df["datetime"] = pd.to_datetime(df["datetime"])

    # 1. Temporal feature extraction
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["dayofyear"] = df["datetime"].dt.dayofyear
    df["quarter"] = df["datetime"].dt.quarter
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Trend and continuous time indices
    base_year = 2011
    df["month_continuous"] = (df["year"] - base_year) * 12 + df["month"]
    df["hour_continuous"] = (
        (df["year"] - base_year) * 8760 + (df["dayofyear"] - 1) * 24 + df["hour"]
    )

    # Commuter vs Leisure behavioral indicators
    df["is_rush_morning"] = (
        (df["workingday"] == 1) & (df["hour"].isin([7, 8, 9]))
    ).astype(int)
    df["is_rush_evening"] = (
        (df["workingday"] == 1) & (df["hour"].isin([16, 17, 18, 19]))
    ).astype(int)
    df["is_rush_hour"] = df["is_rush_morning"] | df["is_rush_evening"]
    df["is_leisure_peak"] = (
        (df["workingday"] == 0) & (df["hour"].between(11, 16))
    ).astype(int)
    df["is_night"] = (df["hour"].between(0, 5)).astype(int)

    # 2. Cyclical trigonometric transformations & Fourier harmonics
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["sin_hour_h2"] = np.sin(4 * np.pi * df["hour"] / 24.0)
    df["cos_hour_h2"] = np.cos(4 * np.pi * df["hour"] / 24.0)
    df["sin_hour_h3"] = np.sin(6 * np.pi * df["hour"] / 24.0)
    df["cos_hour_h3"] = np.cos(6 * np.pi * df["hour"] / 24.0)
    df["sin_month"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)
    df["sin_dayofweek"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["cos_dayofweek"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
    df["sin_dayofyear"] = np.sin(2 * np.pi * (df["dayofyear"] - 1) / 365.25)
    df["cos_dayofyear"] = np.cos(2 * np.pi * (df["dayofyear"] - 1) / 365.25)

    # 3. Solar & Daylight Curve Approximations (Washington D.C. Lat ~ 38.9° N)
    lat_rad = 38.9072 * np.pi / 180.0
    declination_rad = 23.44 * np.pi / 180.0 * np.sin(2 * np.pi * (df["dayofyear"] - 81) / 365.25)
    hour_angle_rad = (df["hour"] - 12.0) * 15.0 * np.pi / 180.0
    sin_elevation = (
        np.sin(lat_rad) * np.sin(declination_rad)
        + np.cos(lat_rad) * np.cos(declination_rad) * np.cos(hour_angle_rad)
    )
    df["solar_elevation"] = np.clip(sin_elevation, 0, None)
    df["is_daylight"] = (df["solar_elevation"] > 0).astype(int)
    df["solar_energy_proxy"] = df["solar_elevation"] * np.clip(1.0 - (df["humidity"] / 100.0) * 0.5, 0, 1)

    # 4. Zero-windspeed contextual reconstruction
    imputer_cols = ["season", "weather", "temp", "atemp", "humidity", "hour", "month", "dayofweek"]
    if train_stats is None:
        train_stats = {}
        ws_non_zero = df[df["windspeed"] > 0]
        imputer = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        )
        imputer.fit(ws_non_zero[imputer_cols], ws_non_zero["windspeed"])
        train_stats["windspeed_imputer"] = imputer

    df["windspeed_is_zero"] = (df["windspeed"] == 0).astype(int)
    df["windspeed_reconstructed"] = df["windspeed"].copy()
    zero_ws_mask = df["windspeed"] == 0
    if zero_ws_mask.sum() > 0 and "windspeed_imputer" in train_stats:
        imputed_ws = train_stats["windspeed_imputer"].predict(df.loc[zero_ws_mask, imputer_cols])
        df.loc[zero_ws_mask, "windspeed_reconstructed"] = np.clip(imputed_ws, 0, None)

    # 5. Multi-step weather dynamics (1h, 2h, 3h lags, leads, diffs, and 3h rolling windows)
    day_group = df.groupby(["year", "month", "day"])
    for col in ["temp", "atemp", "humidity", "windspeed_reconstructed", "weather"]:
        for step in [1, 2, 3]:
            lag = day_group[col].shift(step).bfill()
            lead = day_group[col].shift(-step).ffill()
            df[f"{col}_lag{step}"] = lag
            df[f"{col}_lead{step}"] = lead
            df[f"{col}_diff_lag{step}"] = df[col] - lag
            df[f"{col}_diff_lead{step}"] = lead - df[col]

        df[f"{col}_roll3_mean"] = day_group[col].transform(
            lambda x: x.rolling(3, min_periods=1, center=True).mean()
        )
        df[f"{col}_roll3_std"] = day_group[col].transform(
            lambda x: x.rolling(3, min_periods=1, center=True).std()
        ).fillna(0)

    # 6. Biometeorological indices
    ws_kph = df["windspeed_reconstructed"]
    df["temp_diff"] = df["temp"] - df["atemp"]
    df["temp_ratio"] = df["temp"] / (df["atemp"] + 1e-3)
    df["dew_point"] = df["temp"] - ((100 - df["humidity"]) / 5.0)

    # Humidex discomfort index
    df["humidex"] = df["temp"] + 5 / 9 * (
        (6.11 * np.exp(5417.7530 * (1 / 273.16 - 1 / (273.15 + df["dew_point"])))) - 10
    )

    # Wind Chill index
    df["wind_chill"] = (
        13.12
        + 0.6215 * df["temp"]
        - 11.37 * np.power(np.maximum(ws_kph, 0.1), 0.16)
        + 0.3965 * df["temp"] * np.power(np.maximum(ws_kph, 0.1), 0.16)
    )

    # Steadman Heat Index proxy
    df["heat_index"] = (
        -8.784695
        + 1.611394 * df["temp"]
        + 2.338549 * df["humidity"]
        - 0.146116 * df["temp"] * df["humidity"]
        - 0.012308 * (df["temp"] ** 2)
        - 0.016425 * (df["humidity"] ** 2)
        + 0.002212 * (df["temp"] ** 2) * df["humidity"]
        + 0.000725 * df["temp"] * (df["humidity"] ** 2)
        - 0.00000358 * (df["temp"] ** 2) * (df["humidity"] ** 2)
    )

    # Weather comfort indicators
    df["is_bad_weather"] = (df["weather"] >= 3).astype(int)
    df["is_ideal_weather"] = (
        (df["weather"] == 1)
        & (df["temp"].between(18, 30))
        & (df["humidity"].between(30, 70))
        & (ws_kph < 20)
    ).astype(int)

    # 7. Specialized Casual vs Registered behavioral interactions
    df["casual_solar_weekend"] = df["solar_elevation"] * df["is_weekend"]
    df["casual_temp_weekend"] = df["temp"] * df["is_weekend"]
    df["casual_leisure_comfort"] = df["is_leisure_peak"] * df["solar_elevation"] * (df["weather"] == 1).astype(int)
    df["casual_weather_penalty"] = df["is_weekend"] * df["is_bad_weather"]

    df["commuter_rush_workday"] = df["is_rush_hour"] * df["workingday"]
    df["commuter_morning_workday"] = df["is_rush_morning"] * df["workingday"]
    df["commuter_evening_workday"] = df["is_rush_evening"] * df["workingday"]
    df["commuter_cold_tolerance"] = df["workingday"] * (df["temp"] < 10).astype(int)
    df["commuter_rain_tolerance"] = df["workingday"] * (df["weather"] >= 2).astype(int)

    # Environmental interactions
    df["temp_humidity_interaction"] = df["temp"] * df["humidity"]
    df["temp_windspeed_interaction"] = df["temp"] * ws_kph
    df["humidity_windspeed_interaction"] = df["humidity"] * ws_kph

    # Categorical interaction codes
    df["hour_workingday"] = df["hour"] + df["workingday"] * 24
    df["season_weather"] = (df["season"] - 1) * 4 + (df["weather"] - 1)

    # 8. Target-independent aggregation statistics strictly isolated from training partition
    if "season_hour_stats" not in train_stats:
        stats_group = df.groupby(["season", "hour"])[
            ["temp", "humidity", "windspeed_reconstructed"]
        ].agg(["mean", "std"])
        stats_group.columns = ["_".join(col) for col in stats_group.columns]
        train_stats["season_hour_stats"] = stats_group.reset_index()

    # Merge aggregated statistics back into current partition
    df = df.merge(train_stats["season_hour_stats"], on=["season", "hour"], how="left")

    # Fill any potential NaNs in aggregated columns
    for col in train_stats["season_hour_stats"].columns:
        if col not in ["season", "hour"] and col in df.columns:
            df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)

    # Relative deviations from seasonal hourly normal
    df["temp_dev_season_hour"] = df["temp"] - df["temp_mean"]
    df["humidity_dev_season_hour"] = df["humidity"] - df["humidity_mean"]
    df["windspeed_dev_season_hour"] = df["windspeed_reconstructed"] - df["windspeed_reconstructed_mean"]

    df = df.fillna(0)

    return df, train_stats


def main():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # 1. Load raw data
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    train_df["datetime"] = pd.to_datetime(train_df["datetime"])
    test_df["datetime"] = pd.to_datetime(test_df["datetime"])

    # Prepare log-transformed target variables
    target_cols = ["count", "casual", "registered"]
    for col in target_cols:
        if col in train_df.columns:
            train_df[f"target_log_{col}"] = np.log1p(train_df[col])

    # Intra-monthly validation split mimicking competition horizon:
    # Training: days 1-14; Validation: days 15-19
    train_df["day_num"] = train_df["datetime"].dt.day
    train_split_raw = train_df[train_df["day_num"] <= 14].copy()
    val_split_raw = train_df[train_df["day_num"] > 14].copy()

    # Strict isolation feature extraction
    train_split_feat, fitted_stats = extract_features(train_split_raw, train_stats=None)
    val_split_feat, _ = extract_features(val_split_raw, train_stats=fitted_stats)

    full_train_feat, full_fitted_stats = extract_features(train_df, train_stats=None)
    test_feat, _ = extract_features(test_df, train_stats=full_fitted_stats)

    excluded_cols = set(
        [
            "datetime",
            "count",
            "casual",
            "registered",
            "day_num",
            "id",
        ]
        + [col for col in train_df.columns if col.startswith("target_")]
    )
    feature_cols = [col for col in train_split_feat.columns if col not in excluded_cols]

    X_train = train_split_feat[feature_cols]
    X_val = val_split_feat[feature_cols]
    X_full = full_train_feat[feature_cols]
    X_test = test_feat[feature_cols]

    y_train_casual = train_split_feat["target_log_casual"].values
    y_train_registered = train_split_feat["target_log_registered"].values
    y_train_count = train_split_feat["target_log_count"].values

    y_val_casual = val_split_feat["target_log_casual"].values
    y_val_registered = val_split_feat["target_log_registered"].values
    y_val_count = val_split_feat["target_log_count"].values
    y_val_true_count = val_split_feat["count"].values

    y_full_casual = full_train_feat["target_log_casual"].values
    y_full_registered = full_train_feat["target_log_registered"].values
    y_full_count = full_train_feat["target_log_count"].values

    # 2. Hyperparameter definitions
    lgb_casual_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 45,
        "max_depth": 7,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.2,
        "reg_lambda": 1.0,
        "n_estimators": 2500,
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
    }

    lgb_registered_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 15,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.1,
        "reg_lambda": 0.5,
        "n_estimators": 2500,
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
    }

    lgb_count_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 55,
        "max_depth": 8,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.15,
        "reg_lambda": 0.8,
        "n_estimators": 2500,
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
    }

    xgb_casual_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.2,
        "reg_lambda": 1.0,
        "n_estimators": 2500,
        "random_state": 42,
        "tree_method": "hist",
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    }

    xgb_registered_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 7,
        "min_child_weight": 2,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.1,
        "reg_lambda": 0.5,
        "n_estimators": 2500,
        "random_state": 42,
        "tree_method": "hist",
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    }

    xgb_count_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 7,
        "min_child_weight": 2,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.15,
        "reg_lambda": 0.8,
        "n_estimators": 2500,
        "random_state": 42,
        "tree_method": "hist",
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    }

    cb_casual_params = {
        "iterations": 2500,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": 0,
        "early_stopping_rounds": 50,
    }

    cb_registered_params = {
        "iterations": 2500,
        "learning_rate": 0.03,
        "depth": 7,
        "l2_leaf_reg": 2.0,
        "random_seed": 42,
        "verbose": 0,
        "early_stopping_rounds": 50,
    }

    cb_count_params = {
        "iterations": 2500,
        "learning_rate": 0.03,
        "depth": 7,
        "l2_leaf_reg": 2.5,
        "random_seed": 42,
        "verbose": 0,
        "early_stopping_rounds": 50,
    }

    # 3. Train LightGBM Validation Models
    lgb_callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]

    lgb_model_cas = lgb.LGBMRegressor(**lgb_casual_params)
    lgb_model_cas.fit(
        X_train,
        y_train_casual,
        eval_set=[(X_val, y_val_casual)],
        callbacks=lgb_callbacks,
    )

    lgb_model_reg = lgb.LGBMRegressor(**lgb_registered_params)
    lgb_model_reg.fit(
        X_train,
        y_train_registered,
        eval_set=[(X_val, y_val_registered)],
        callbacks=lgb_callbacks,
    )

    lgb_model_cnt = lgb.LGBMRegressor(**lgb_count_params)
    lgb_model_cnt.fit(
        X_train,
        y_train_count,
        eval_set=[(X_val, y_val_count)],
        callbacks=lgb_callbacks,
    )

    # 4. Train XGBoost Validation Models
    xgb_model_cas = xgb.XGBRegressor(**xgb_casual_params)
    xgb_model_cas.fit(
        X_train, y_train_casual, eval_set=[(X_val, y_val_casual)], verbose=False
    )

    xgb_model_reg = xgb.XGBRegressor(**xgb_registered_params)
    xgb_model_reg.fit(
        X_train,
        y_train_registered,
        eval_set=[(X_val, y_val_registered)],
        verbose=False,
    )

    xgb_model_cnt = xgb.XGBRegressor(**xgb_count_params)
    xgb_model_cnt.fit(
        X_train, y_train_count, eval_set=[(X_val, y_val_count)], verbose=False
    )

    # 5. Train CatBoost Validation Models
    cb_model_cas = CatBoostRegressor(**cb_casual_params)
    cb_model_cas.fit(
        X_train, y_train_casual, eval_set=(X_val, y_val_casual), verbose=False
    )

    cb_model_reg = CatBoostRegressor(**cb_registered_params)
    cb_model_reg.fit(
        X_train, y_train_registered, eval_set=(X_val, y_val_registered), verbose=False
    )

    cb_model_cnt = CatBoostRegressor(**cb_count_params)
    cb_model_cnt.fit(
        X_train, y_train_count, eval_set=(X_val, y_val_count), verbose=False
    )

    # 6. Validation Ensemble Inference
    val_cas_lgb = np.clip(np.expm1(lgb_model_cas.predict(X_val)), 0, None)
    val_reg_lgb = np.clip(np.expm1(lgb_model_reg.predict(X_val)), 0, None)
    val_cnt_lgb_dir = np.clip(np.expm1(lgb_model_cnt.predict(X_val)), 0, None)
    val_lgb_blend = 0.85 * (val_cas_lgb + val_reg_lgb) + 0.15 * val_cnt_lgb_dir

    val_cas_xgb = np.clip(np.expm1(xgb_model_cas.predict(X_val)), 0, None)
    val_reg_xgb = np.clip(np.expm1(xgb_model_reg.predict(X_val)), 0, None)
    val_cnt_xgb_dir = np.clip(np.expm1(xgb_model_cnt.predict(X_val)), 0, None)
    val_xgb_blend = 0.85 * (val_cas_xgb + val_reg_xgb) + 0.15 * val_cnt_xgb_dir

    val_cas_cb = np.clip(np.expm1(cb_model_cas.predict(X_val)), 0, None)
    val_reg_cb = np.clip(np.expm1(cb_model_reg.predict(X_val)), 0, None)
    val_cnt_cb_dir = np.clip(np.expm1(cb_model_cnt.predict(X_val)), 0, None)
    val_cb_blend = 0.85 * (val_cas_cb + val_reg_cb) + 0.15 * val_cnt_cb_dir

    val_ensemble = 0.40 * val_lgb_blend + 0.35 * val_xgb_blend + 0.25 * val_cb_blend
    validation_score = calculate_rmsle(y_val_true_count, val_ensemble)

    # 7. Retrain Models on Full Training Dataset
    expansion_factor = len(full_train_feat) / len(train_split_feat)

    lgb_full_cas = lgb.LGBMRegressor(
        **{
            **lgb_casual_params,
            "n_estimators": max(
                100, int(lgb_model_cas.best_iteration_ * expansion_factor)
            ),
        }
    )
    lgb_full_cas.fit(X_full, y_full_casual)

    lgb_full_reg = lgb.LGBMRegressor(
        **{
            **lgb_registered_params,
            "n_estimators": max(
                100, int(lgb_model_reg.best_iteration_ * expansion_factor)
            ),
        }
    )
    lgb_full_reg.fit(X_full, y_full_registered)

    lgb_full_cnt = lgb.LGBMRegressor(
        **{
            **lgb_count_params,
            "n_estimators": max(
                100, int(lgb_model_cnt.best_iteration_ * expansion_factor)
            ),
        }
    )
    lgb_full_cnt.fit(X_full, y_full_count)

    xgb_full_cas_params = {
        k: v for k, v in xgb_casual_params.items() if k != "early_stopping_rounds"
    }
    xgb_full_cas_params["n_estimators"] = max(
        100, int(xgb_model_cas.best_iteration * expansion_factor)
    )
    xgb_full_cas = xgb.XGBRegressor(**xgb_full_cas_params)
    xgb_full_cas.fit(X_full, y_full_casual, verbose=False)

    xgb_full_reg_params = {
        k: v for k, v in xgb_registered_params.items() if k != "early_stopping_rounds"
    }
    xgb_full_reg_params["n_estimators"] = max(
        100, int(xgb_model_reg.best_iteration * expansion_factor)
    )
    xgb_full_reg = xgb.XGBRegressor(**xgb_full_reg_params)
    xgb_full_reg.fit(X_full, y_full_registered, verbose=False)

    xgb_full_cnt_params = {
        k: v for k, v in xgb_count_params.items() if k != "early_stopping_rounds"
    }
    xgb_full_cnt_params["n_estimators"] = max(
        100, int(xgb_model_cnt.best_iteration * expansion_factor)
    )
    xgb_full_cnt = xgb.XGBRegressor(**xgb_full_cnt_params)
    xgb_full_cnt.fit(X_full, y_full_count, verbose=False)

    cb_full_cas_params = {
        k: v for k, v in cb_casual_params.items() if k != "early_stopping_rounds"
    }
    cb_full_cas_best_iter = cb_model_cas.get_best_iteration()
    cb_full_cas_params["iterations"] = max(
        100,
        int((cb_full_cas_best_iter if cb_full_cas_best_iter is not None else 1000) * expansion_factor),
    )
    cb_full_cas = CatBoostRegressor(**cb_full_cas_params)
    cb_full_cas.fit(X_full, y_full_casual, verbose=False)

    cb_full_reg_params = {
        k: v for k, v in cb_registered_params.items() if k != "early_stopping_rounds"
    }
    cb_full_reg_best_iter = cb_model_reg.get_best_iteration()
    cb_full_reg_params["iterations"] = max(
        100,
        int((cb_full_reg_best_iter if cb_full_reg_best_iter is not None else 1000) * expansion_factor),
    )
    cb_full_reg = CatBoostRegressor(**cb_full_reg_params)
    cb_full_reg.fit(X_full, y_full_registered, verbose=False)

    cb_full_cnt_params = {
        k: v for k, v in cb_count_params.items() if k != "early_stopping_rounds"
    }
    cb_full_cnt_best_iter = cb_model_cnt.get_best_iteration()
    cb_full_cnt_params["iterations"] = max(
        100,
        int((cb_full_cnt_best_iter if cb_full_cnt_best_iter is not None else 1000) * expansion_factor),
    )
    cb_full_cnt = CatBoostRegressor(**cb_full_cnt_params)
    cb_full_cnt.fit(X_full, y_full_count, verbose=False)

    # 8. Generate Test Predictions
    test_cas_lgb = np.clip(np.expm1(lgb_full_cas.predict(X_test)), 0, None)
    test_reg_lgb = np.clip(np.expm1(lgb_full_reg.predict(X_test)), 0, None)
    test_cnt_lgb_dir = np.clip(np.expm1(lgb_full_cnt.predict(X_test)), 0, None)
    test_lgb_blend = 0.85 * (test_cas_lgb + test_reg_lgb) + 0.15 * test_cnt_lgb_dir

    test_cas_xgb = np.clip(np.expm1(xgb_full_cas.predict(X_test)), 0, None)
    test_reg_xgb = np.clip(np.expm1(xgb_full_reg.predict(X_test)), 0, None)
    test_cnt_xgb_dir = np.clip(np.expm1(xgb_full_cnt.predict(X_test)), 0, None)
    test_xgb_blend = 0.85 * (test_cas_xgb + test_reg_xgb) + 0.15 * test_cnt_xgb_dir

    test_cas_cb = np.clip(np.expm1(cb_full_cas.predict(X_test)), 0, None)
    test_reg_cb = np.clip(np.expm1(cb_full_reg.predict(X_test)), 0, None)
    test_cnt_cb_dir = np.clip(np.expm1(cb_full_cnt.predict(X_test)), 0, None)
    test_cb_blend = 0.85 * (test_cas_cb + test_reg_cb) + 0.15 * test_cnt_cb_dir

    test_ensemble_preds = (
        0.40 * test_lgb_blend + 0.35 * test_xgb_blend + 0.25 * test_cb_blend
    )
    test_ensemble_preds = np.clip(test_ensemble_preds, 0, None)

    # 8. Save Submission
    test_ids = (
        test_df["id"].values if "id" in test_df.columns else np.arange(len(test_df))
    )
    submission_df = pd.DataFrame({"id": test_ids, "count": test_ensemble_preds})
    submission_df.to_csv("./submission/submission.csv", index=False)

    # 9. Output Final Validation Score
    print(f"Final Validation Score: {validation_score}")


if __name__ == "__main__":
    main()
