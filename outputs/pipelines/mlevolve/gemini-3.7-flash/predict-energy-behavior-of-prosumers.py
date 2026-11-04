"""
Enefit - Predict Energy Behavior of Prosumers
Complete End-to-End Pipeline: Multi-Table Feature Engineering,
Decoupled Dual-Regime GBDT Architecture, L1 Optimization, and Post-Processing.
"""

import gc
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import joblib
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb

warnings.filterwarnings("ignore")

# =========================================================================
# 1. Configuration & Utilities
# =========================================================================
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)


def find_file(filename_patterns: List[str]) -> Optional[str]:
    """Search for input files across standard subdirectories and formats."""
    for pattern in filename_patterns:
        candidates = [
            os.path.join(INPUT_DIR, pattern),
            os.path.join(INPUT_DIR, "train", pattern),
            os.path.join(INPUT_DIR, "test", pattern),
            pattern,
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
    return None


def read_data_file(patterns: List[str]) -> pd.DataFrame:
    path = find_file(patterns)
    if path is None:
        raise FileNotFoundError(f"Could not locate file matching patterns: {patterns}")
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported format: {path}")


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric types to reduce memory consumption."""
    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type).startswith("int") or str(col_type).startswith("uint"):
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min >= -128 and c_max <= 127:
                df[col] = df[col].astype(np.int8)
            elif c_min >= -32768 and c_max <= 32767:
                df[col] = df[col].astype(np.int16)
            elif c_min >= -2147483648 and c_max <= 2147483647:
                df[col] = df[col].astype(np.int32)
        elif str(col_type).startswith("float"):
            df[col] = df[col].astype(np.float32)
    return df


def competition_mae(
    y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]
) -> float:
    """Official competition evaluation metric: Mean Absolute Error (MAE)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


# =========================================================================
# 2. Astronomical, Holiday & Temporal Feature Engineering
# =========================================================================
def get_estonian_holidays(years: List[int]) -> set:
    """Return a set of Estonian public holiday date strings (YYYY-MM-DD)."""
    holidays = set()
    for y in years:
        for m, d in [
            (1, 1),   # New Year's Day
            (2, 24),  # Independence Day
            (5, 1),   # Spring Day
            (6, 23),  # Victory Day
            (6, 24),  # Midsummer Day
            (8, 20),  # Day of Restoration of Independence
            (12, 24), # Christmas Eve
            (12, 25), # Christmas Day
            (12, 26), # Boxing Day
        ]:
            holidays.add(f"{y:04d}-{m:02d}-{d:02d}")
    movable_holidays = [
        "2021-04-02", "2021-04-04", "2021-05-23",
        "2022-04-15", "2022-04-17", "2022-06-05",
        "2023-04-07", "2023-04-09", "2023-05-28",
        "2024-03-29", "2024-03-31", "2024-05-19",
    ]
    for d_str in movable_holidays:
        holidays.add(d_str)
    return holidays


def add_solar_and_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute temporal cyclics, Estonian public holiday dynamics,
    and physical solar position angles (solar elevation and azimuth) for Estonia.
    """
    dt = pd.to_datetime(df["datetime"])
    df["hour"] = dt.dt.hour.astype(np.int8)
    df["day"] = dt.dt.day.astype(np.int8)
    df["weekday"] = dt.dt.weekday.astype(np.int8)
    df["month"] = dt.dt.month.astype(np.int8)
    df["dayofyear"] = dt.dt.dayofyear.astype(np.int16)
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(np.int8)

    # Estonian Public Holidays & Interactions
    estonian_holidays = get_estonian_holidays(list(range(2021, 2026)))
    date_strs = dt.dt.strftime("%Y-%m-%d")
    df["is_estonian_holiday"] = date_strs.isin(estonian_holidays).astype(np.int8)
    df["is_holiday_or_weekend"] = (df["is_estonian_holiday"] | df["is_weekend"]).astype(np.int8)
    df["is_holiday_and_weekend"] = (df["is_estonian_holiday"] & df["is_weekend"]).astype(np.int8)

    # Cyclical encodings
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0).astype(np.float32)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0).astype(np.float32)
    df["sin_dayofyear"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25).astype(
        np.float32
    )
    df["cos_dayofyear"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25).astype(
        np.float32
    )

    # Astronomical solar elevation calculation
    estonia_lat = np.radians(58.6)
    declination = np.radians(-23.44) * np.cos(
        2 * np.pi * (df["dayofyear"] + 10) / 365.25
    )
    hour_angle = np.radians(15.0 * (df["hour"] - 13.0))
    sin_elevation = np.sin(estonia_lat) * np.sin(declination) + np.cos(
        estonia_lat
    ) * np.cos(declination) * np.cos(hour_angle)
    df["solar_elevation"] = np.clip(sin_elevation, -1.0, 1.0).astype(np.float32)
    df["is_daylight"] = (df["solar_elevation"] > 0.0).astype(np.int8)

    return df


# =========================================================================
# 3. Reference Tables Preprocessing
# =========================================================================
def prepare_weather_forecast(
    fw_df: pd.DataFrame, mapping_df: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate weather forecasts to county and forecast_datetime level."""
    fw_df = fw_df.copy()
    mapping_df = mapping_df.copy()

    fw_df["latitude"] = fw_df["latitude"].round(1)
    fw_df["longitude"] = fw_df["longitude"].round(1)
    mapping_df["latitude"] = mapping_df["latitude"].round(1)
    mapping_df["longitude"] = mapping_df["longitude"].round(1)

    fw = fw_df.merge(
        mapping_df[["latitude", "longitude", "county"]].dropna(),
        on=["latitude", "longitude"],
        how="inner",
    )
    fw["forecast_datetime"] = pd.to_datetime(fw["forecast_datetime"])
    fw["county"] = fw["county"].astype(np.int16)

    if "temperature" in fw.columns:
        fw["hdd"] = np.maximum(0.0, 18.0 - fw["temperature"])
        fw["hdd_nonlinear"] = np.power(np.maximum(0.0, 18.0 - fw["temperature"]), 1.5)
        fw["cdd"] = np.maximum(0.0, fw["temperature"] - 22.0)

    if "temperature" in fw.columns and "dewpoint" in fw.columns:
        fw["dewpoint_depression"] = fw["temperature"] - fw["dewpoint"]

    if "direct_solar_radiation" in fw.columns and "cloudcover_total" in fw.columns:
        fw["cloud_attenuated_radiation"] = fw["direct_solar_radiation"] * (
            1.0 - np.clip(fw["cloudcover_total"] / 100.0, 0.0, 1.0)
        )

    agg_dict = {
        "temperature": ["mean", "min", "max"],
        "dewpoint": ["mean"],
        "dewpoint_depression": ["mean"],
        "hdd": ["mean"],
        "hdd_nonlinear": ["mean"],
        "cdd": ["mean"],
        "cloudcover_total": ["mean", "max"],
        "cloudcover_low": ["mean"],
        "cloudcover_mid": ["mean"],
        "cloudcover_high": ["mean"],
        "direct_solar_radiation": ["mean", "max"],
        "cloud_attenuated_radiation": ["mean", "max"],
        "surface_solar_radiation_downwards": (
            ["mean", "max"]
            if "surface_solar_radiation_downwards" in fw.columns
            else ["mean"]
        ),
        "10_metre_u_wind_component": ["mean"],
        "10_metre_v_wind_component": ["mean"],
    }
    agg_dict = {k: v for k, v in agg_dict.items() if k in fw.columns}

    fw_agg = fw.groupby(["county", "forecast_datetime"]).agg(agg_dict)
    fw_agg.columns = [f"fw_{col}_{stat}" for col, stat in fw_agg.columns]
    fw_agg = fw_agg.reset_index()

    if (
        "fw_10_metre_u_wind_component_mean" in fw_agg.columns
        and "fw_10_metre_v_wind_component_mean" in fw_agg.columns
    ):
        fw_agg["fw_wind_speed"] = np.sqrt(
            fw_agg["fw_10_metre_u_wind_component_mean"] ** 2
            + fw_agg["fw_10_metre_v_wind_component_mean"] ** 2
        ).astype(np.float32)

    return reduce_mem_usage(fw_agg)


def prepare_historical_weather(
    hw_df: pd.DataFrame, mapping_df: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate historical weather observations to county and datetime level."""
    hw_df = hw_df.copy()
    mapping_df = mapping_df.copy()

    hw_df["latitude"] = hw_df["latitude"].round(1)
    hw_df["longitude"] = hw_df["longitude"].round(1)
    mapping_df["latitude"] = mapping_df["latitude"].round(1)
    mapping_df["longitude"] = mapping_df["longitude"].round(1)

    hw = hw_df.merge(
        mapping_df[["latitude", "longitude", "county"]].dropna(),
        on=["latitude", "longitude"],
        how="inner",
    )
    hw["datetime"] = pd.to_datetime(hw["datetime"])
    hw["county"] = hw["county"].astype(np.int16)

    if "temperature" in hw.columns and "dewpoint" in hw.columns:
        hw["dewpoint_depression"] = hw["temperature"] - hw["dewpoint"]

    agg_dict = {
        "temperature": ["mean"],
        "dewpoint_depression": ["mean"],
        "rain": ["mean", "max"],
        "snowfall": ["mean"],
        "surface_pressure": ["mean"],
        "cloudcover_total": ["mean"],
        "direct_solar_radiation": ["mean"],
        "diffuse_radiation": ["mean"],
    }
    agg_dict = {k: v for k, v in agg_dict.items() if k in hw.columns}

    hw_agg = hw.groupby(["county", "datetime"]).agg(agg_dict)
    hw_agg.columns = [f"hw_{col}_{stat}" for col, stat in hw_agg.columns]
    hw_agg = hw_agg.reset_index()
    return reduce_mem_usage(hw_agg)


def prepare_prices(
    elec_df: pd.DataFrame, gas_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare electricity and gas price tables."""
    elec = elec_df.copy()
    elec["forecast_date"] = pd.to_datetime(elec["forecast_date"])
    elec = elec.rename(
        columns={"forecast_date": "datetime", "euros_per_mwh": "elec_price_per_mwh"}
    )
    elec = elec[["datetime", "elec_price_per_mwh"]].drop_duplicates(subset=["datetime"])

    gas = gas_df.copy()
    gas["gas_price_mean"] = (
        gas["lowest_price_per_mwh"] + gas["highest_price_per_mwh"]
    ) / 2.0
    gas = gas[
        [
            "data_block_id",
            "lowest_price_per_mwh",
            "highest_price_per_mwh",
            "gas_price_mean",
        ]
    ].drop_duplicates(subset=["data_block_id"])

    return reduce_mem_usage(elec), reduce_mem_usage(gas)


def prepare_client(client_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare client capacity table with momentum features (available with a 2-day lag: data_block_id + 2)."""
    client = client_df.copy()
    client = client.sort_values(
        ["product_type", "county", "is_business", "data_block_id"]
    ).reset_index(drop=True)
    client["capacity_per_eic"] = (
        client["installed_capacity"] / (client["eic_count"] + 1.0)
    ).astype(np.float32)
    client["capacity_momentum_7d"] = (
        client.groupby(["product_type", "county", "is_business"])["installed_capacity"]
        .diff()
        .fillna(0.0)
        .astype(np.float32)
    )
    client["eic_momentum_7d"] = (
        client.groupby(["product_type", "county", "is_business"])["eic_count"]
        .diff()
        .fillna(0.0)
        .astype(np.float32)
    )
    client = client[
        [
            "product_type",
            "county",
            "is_business",
            "eic_count",
            "installed_capacity",
            "capacity_per_eic",
            "capacity_momentum_7d",
            "eic_momentum_7d",
            "data_block_id",
        ]
    ].drop_duplicates(subset=["product_type", "county", "is_business", "data_block_id"])
    client["data_block_id"] = client["data_block_id"] + 2
    return reduce_mem_usage(client)


# -------------------------------------------------------------------------
# 4. Target Lag Feature Engineering (Strict 48h Latency Alignment)
# -------------------------------------------------------------------------
def create_target_lags(
    train_targets: pd.DataFrame, all_grid: pd.DataFrame
) -> pd.DataFrame:
    """
    Construct multi-scale autoregressive target lags respecting the 48-hour latency.
    Only historical target values are merged to avoid future target leakage.
    """
    target_df = train_targets[
        ["datetime", "prediction_unit_id", "is_consumption", "target"]
    ].dropna()
    target_df = target_df.drop_duplicates(
        subset=["prediction_unit_id", "is_consumption", "datetime"]
    )

    lags_hours = [48, 72, 96, 120, 144, 168, 336, 504]
    merged_lags = all_grid.copy()

    for lag in lags_hours:
        lag_sub = target_df.copy()
        lag_sub["datetime"] = lag_sub["datetime"] + pd.Timedelta(hours=lag)
        lag_sub = lag_sub.rename(columns={"target": f"target_lag_{lag}h"})
        merged_lags = merged_lags.merge(
            lag_sub, on=["datetime", "prediction_unit_id", "is_consumption"], how="left"
        )

    # Rolling statistics on lags
    lag_cols = [f"target_lag_{h}h" for h in [48, 72, 96, 120, 144, 168]]
    merged_lags["target_lag_week_mean"] = (
        merged_lags[lag_cols].mean(axis=1).astype(np.float32)
    )
    merged_lags["target_lag_week_std"] = (
        merged_lags[lag_cols].std(axis=1).astype(np.float32)
    )
    merged_lags["target_lag_week_max"] = (
        merged_lags[lag_cols].max(axis=1).astype(np.float32)
    )
    merged_lags["target_lag_week_min"] = (
        merged_lags[lag_cols].min(axis=1).astype(np.float32)
    )

    # Momentum differences across diurnal and multi-week cycles
    merged_lags["target_diff_48_72"] = (
        merged_lags["target_lag_48h"] - merged_lags["target_lag_72h"]
    ).astype(np.float32)
    merged_lags["target_diff_48_168"] = (
        merged_lags["target_lag_48h"] - merged_lags["target_lag_168h"]
    ).astype(np.float32)
    merged_lags["target_diff_168_336"] = (
        merged_lags["target_lag_168h"] - merged_lags["target_lag_336h"]
    ).astype(np.float32)
    merged_lags["target_ratio_48_168"] = (
        merged_lags["target_lag_48h"] / (merged_lags["target_lag_168h"] + 1e-2)
    ).astype(np.float32)

    return reduce_mem_usage(merged_lags)


# -------------------------------------------------------------------------
# 5. Full Feature Pipeline Assembly
# -------------------------------------------------------------------------
def build_feature_dataset(
    df_raw: pd.DataFrame,
    client_proc: pd.DataFrame,
    elec_proc: pd.DataFrame,
    gas_proc: pd.DataFrame,
    fw_proc: pd.DataFrame,
    hw_proc: pd.DataFrame,
    lags_proc: pd.DataFrame,
) -> pd.DataFrame:
    """Join all processed reference tables into the primary dataset."""
    df = df_raw.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    if "prediction_unit_id" not in df.columns:
        df["prediction_unit_id"] = (
            (
                df["county"].astype(str)
                + "_"
                + df["is_business"].astype(str)
                + "_"
                + df["product_type"].astype(str)
            )
            .astype("category")
            .cat.codes
        )

    # 1. Temporal and Solar features
    df = add_solar_and_time_features(df)

    # 2. Client capacity features
    df = df.merge(
        client_proc,
        on=["product_type", "county", "is_business", "data_block_id"],
        how="left",
    )

    # 3. Electricity & Gas prices
    df = df.merge(elec_proc, on="datetime", how="left")
    df = df.merge(gas_proc, on="data_block_id", how="left")

    # 4. Forecast weather
    df = df.merge(
        fw_proc.rename(columns={"forecast_datetime": "datetime"}),
        on=["county", "datetime"],
        how="left",
    )

    # 5. Historical weather (shifted by 48 hours)
    hw_shifted = hw_proc.copy()
    hw_shifted["datetime"] = hw_shifted["datetime"] + pd.Timedelta(hours=48)
    df = df.merge(hw_shifted, on=["county", "datetime"], how="left")

    # 6. Target Lags & Rolling statistics
    df = df.merge(
        lags_proc, on=["datetime", "prediction_unit_id", "is_consumption"], how="left"
    )

    # 7. Physical Interaction & Weather Drift Features
    if "fw_temperature_mean" in df.columns and "hw_temperature_mean" in df.columns:
        df["temp_forecast_hw_drift"] = (
            df["fw_temperature_mean"] - df["hw_temperature_mean"]
        ).astype(np.float32)

    if "fw_cloudcover_total_mean" in df.columns and "hw_cloudcover_total_mean" in df.columns:
        df["cloud_forecast_hw_drift"] = (
            df["fw_cloudcover_total_mean"] - df["hw_cloudcover_total_mean"]
        ).astype(np.float32)

    if (
        "fw_direct_solar_radiation_mean" in df.columns
        and "fw_surface_solar_radiation_downwards_mean" in df.columns
    ):
        df["fw_direct_surface_ratio"] = (
            df["fw_direct_solar_radiation_mean"]
            / (df["fw_surface_solar_radiation_downwards_mean"] + 1.0)
        ).astype(np.float32)

    if "fw_direct_solar_radiation_mean" in df.columns:
        df["effective_solar_irradiance"] = (
            np.maximum(0.0, df["solar_elevation"])
            * df["fw_direct_solar_radiation_mean"].fillna(0)
        ).astype(np.float32)

    if (
        "installed_capacity" in df.columns
        and "fw_direct_solar_radiation_mean" in df.columns
    ):
        df["solar_capacity_potential"] = (
            df["installed_capacity"].fillna(0)
            * np.maximum(0.0, df["solar_elevation"])
            * df["fw_direct_solar_radiation_mean"].fillna(0)
        ).astype(np.float32)

    if "installed_capacity" in df.columns:
        if "target_lag_48h" in df.columns:
            df["lag_48_capacity_ratio"] = (
                df["target_lag_48h"] / (df["installed_capacity"] + 1e-2)
            ).astype(np.float32)
        if "target_lag_168h" in df.columns:
            df["lag_168_capacity_ratio"] = (
                df["target_lag_168h"] / (df["installed_capacity"] + 1e-2)
            ).astype(np.float32)

    return reduce_mem_usage(df)


# =========================================================================
# 6. Dual-Regime Model Architecture
# =========================================================================
class DualRegimeEnergyModel(BaseEstimator, RegressorMixin):
    """
    Decoupled Heterogeneous Ensemble Model Architecture separating
    Photovoltaic Solar Generation (is_consumption=0) and
    Electrical Consumption (is_consumption=1).
    """

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        enforce_solar_physics: bool = True,
    ):
        self.feature_cols = feature_cols
        self.categorical_cols = categorical_cols or []
        self.enforce_solar_physics = enforce_solar_physics
        self.prod_weights = np.array([0.45, 0.30, 0.25], dtype=np.float64)
        self.cons_weights = np.array([0.45, 0.30, 0.25], dtype=np.float64)

        # Production Regime Ensemble
        self.prod_lgb_l1 = lgb.LGBMRegressor(
            objective="regression_l1",
            metric="mae",
            boosting_type="gbdt",
            n_estimators=2500,
            learning_rate=0.04,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=25,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        self.prod_lgb_huber = lgb.LGBMRegressor(
            objective="huber",
            metric="mae",
            boosting_type="gbdt",
            n_estimators=2500,
            learning_rate=0.04,
            num_leaves=45,
            max_depth=-1,
            min_child_samples=30,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.60,
            reg_alpha=0.2,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        self.prod_xgb = xgb.XGBRegressor(
            objective="reg:absoluteerror",
            eval_metric="mae",
            tree_method="hist",
            enable_categorical=True,
            n_estimators=2000,
            learning_rate=0.04,
            max_depth=7,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_alpha=0.1,
            reg_lambda=1.0,
            early_stopping_rounds=50,
            random_state=42,
            n_jobs=-1,
        )

        # Consumption Regime Ensemble
        self.cons_lgb_l1 = lgb.LGBMRegressor(
            objective="regression_l1",
            metric="mae",
            boosting_type="gbdt",
            n_estimators=2500,
            learning_rate=0.04,
            num_leaves=95,
            max_depth=-1,
            min_child_samples=30,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.80,
            reg_alpha=0.05,
            reg_lambda=0.5,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        self.cons_lgb_huber = lgb.LGBMRegressor(
            objective="huber",
            metric="mae",
            boosting_type="gbdt",
            n_estimators=2500,
            learning_rate=0.04,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=35,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=0.1,
            reg_lambda=1.5,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        self.cons_xgb = xgb.XGBRegressor(
            objective="reg:absoluteerror",
            eval_metric="mae",
            tree_method="hist",
            enable_categorical=True,
            n_estimators=2000,
            learning_rate=0.04,
            max_depth=8,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.05,
            reg_lambda=0.5,
            early_stopping_rounds=50,
            random_state=42,
            n_jobs=-1,
        )

        self.is_fitted_ = False

    def fit_prod(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray):
        print("  - Training LightGBM L1 (Production)...")
        self.prod_lgb_l1.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="mae",
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        print("  - Training LightGBM Huber (Production)...")
        self.prod_lgb_huber.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="mae",
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        print("  - Training XGBoost MAE (Production)...")
        self.prod_xgb.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

    def fit_cons(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray):
        print("  - Training LightGBM L1 (Consumption)...")
        self.cons_lgb_l1.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="mae",
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        print("  - Training LightGBM Huber (Consumption)...")
        self.cons_lgb_huber.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="mae",
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        print("  - Training XGBoost MAE (Consumption)...")
        self.cons_xgb.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

    def optimize_weights(
        self,
        X_val_prod: pd.DataFrame,
        y_val_prod: np.ndarray,
        X_val_cons: pd.DataFrame,
        y_val_cons: np.ndarray,
    ):
        """Optimize ensemble weights on validation set using MAE loss."""
        # Solar production regime weights
        p_l1_prod = self.prod_lgb_l1.predict(X_val_prod)
        p_hub_prod = self.prod_lgb_huber.predict(X_val_prod)
        p_xgb_prod = self.prod_xgb.predict(X_val_prod)
        P_prod = np.column_stack([p_l1_prod, p_hub_prod, p_xgb_prod])

        def loss_prod(w):
            pred = P_prod @ w
            return np.mean(np.abs(y_val_prod - pred))

        res_prod = minimize(
            loss_prod,
            [1.0 / 3, 1.0 / 3, 1.0 / 3],
            bounds=[(0.0, 1.0)] * 3,
            method="SLSQP",
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        )
        if res_prod.success:
            self.prod_weights = res_prod.x / np.sum(res_prod.x)
            print(f"  - Optimized Production Blend Weights: {self.prod_weights}")

        # Electrical consumption regime weights
        p_l1_cons = self.cons_lgb_l1.predict(X_val_cons)
        p_hub_cons = self.cons_lgb_huber.predict(X_val_cons)
        p_xgb_cons = self.cons_xgb.predict(X_val_cons)
        P_cons = np.column_stack([p_l1_cons, p_hub_cons, p_xgb_cons])

        def loss_cons(w):
            pred = P_cons @ w
            return np.mean(np.abs(y_val_cons - pred))

        res_cons = minimize(
            loss_cons,
            [1.0 / 3, 1.0 / 3, 1.0 / 3],
            bounds=[(0.0, 1.0)] * 3,
            method="SLSQP",
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        )
        if res_cons.success:
            self.cons_weights = res_cons.x / np.sum(res_cons.x)
            print(f"  - Optimized Consumption Blend Weights: {self.cons_weights}")

    def predict_prod(self, X: pd.DataFrame) -> np.ndarray:
        p_l1 = self.prod_lgb_l1.predict(X)
        p_huber = self.prod_lgb_huber.predict(X)
        p_xgb = self.prod_xgb.predict(X)
        return (
            self.prod_weights[0] * p_l1
            + self.prod_weights[1] * p_huber
            + self.prod_weights[2] * p_xgb
        )

    def predict_cons(self, X: pd.DataFrame) -> np.ndarray:
        p_l1 = self.cons_lgb_l1.predict(X)
        p_huber = self.cons_lgb_huber.predict(X)
        p_xgb = self.cons_xgb.predict(X)
        return (
            self.cons_weights[0] * p_l1
            + self.cons_weights[1] * p_huber
            + self.cons_weights[2] * p_xgb
        )

    def apply_domain_constraints(
        self,
        preds: np.ndarray,
        is_consumption: np.ndarray,
        solar_elevation: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Enforce non-negativity and astronomical solar zero-irradiance boundaries."""
        constrained_preds = np.maximum(0.0, preds)
        if self.enforce_solar_physics and solar_elevation is not None:
            night_prod_mask = (is_consumption == 0) & (solar_elevation <= 0.0)
            constrained_preds[night_prod_mask] = 0.0
        return constrained_preds


# =========================================================================
# 7. Main Execution Pipeline
# =========================================================================
def main():
    print("Loading raw datasets from input directory...")
    train_raw = read_data_file(["train.parquet", "train.csv"])
    test_raw = read_data_file(["test.parquet", "test.csv"])
    client_raw = read_data_file(["client.parquet", "client.csv"])
    elec_raw = read_data_file(["electricity_prices.parquet", "electricity_prices.csv"])
    gas_raw = read_data_file(["gas_prices.parquet", "gas_prices.csv"])
    fw_raw = read_data_file(["forecast_weather.parquet", "forecast_weather.csv"])
    hw_raw = read_data_file(["historical_weather.parquet", "historical_weather.csv"])
    mapping_raw = read_data_file(
        [
            "weather_station_to_county_mapping.parquet",
            "weather_station_to_county_mapping.csv",
        ]
    )

    train_raw["datetime"] = pd.to_datetime(train_raw["datetime"])
    test_raw["datetime"] = pd.to_datetime(test_raw["datetime"])
    train_raw = train_raw.dropna(subset=["target"]).reset_index(drop=True)

    print("Preprocessing reference tables...")
    client_proc = prepare_client(client_raw)
    elec_proc, gas_proc = prepare_prices(elec_raw, gas_raw)
    fw_proc = prepare_weather_forecast(fw_raw, mapping_raw)
    hw_proc = prepare_historical_weather(hw_raw, mapping_raw)

    del fw_raw, hw_raw, client_raw, elec_raw, gas_raw, mapping_raw
    gc.collect()

    print("Constructing 48h autoregressive target lags...")
    all_grid = (
        pd.concat(
            [
                train_raw[["datetime", "prediction_unit_id", "is_consumption"]],
                test_raw[["datetime", "prediction_unit_id", "is_consumption"]],
            ],
            ignore_index=True,
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )

    lags_proc = create_target_lags(train_raw, all_grid)
    del all_grid
    gc.collect()

    print("Building full feature matrices for train and test...")
    df_full_train = build_feature_dataset(
        train_raw, client_proc, elec_proc, gas_proc, fw_proc, hw_proc, lags_proc
    )
    df_full_test = build_feature_dataset(
        test_raw, client_proc, elec_proc, gas_proc, fw_proc, hw_proc, lags_proc
    )

    # ---------------------------------------------------------------------
    # Time-Based Out-of-Time Validation Split (Last 45 Days)
    # ---------------------------------------------------------------------
    max_block = df_full_train["data_block_id"].max()
    val_split_block = max_block - 45

    train_split = (
        df_full_train[df_full_train["data_block_id"] < val_split_block]
        .copy()
        .reset_index(drop=True)
    )
    val_split = (
        df_full_train[df_full_train["data_block_id"] >= val_split_block]
        .copy()
        .reset_index(drop=True)
    )

    print(
        f"Train shape: {train_split.shape} | Val shape: {val_split.shape} | Test shape: {df_full_test.shape}"
    )

    # ---------------------------------------------------------------------
    # Feature Set Definition
    # ---------------------------------------------------------------------
    ignore_cols = {
        "target",
        "datetime",
        "row_id",
        "id",
        "data_block_id",
        "date",
        "origin_date",
        "origin_datetime",
        "forecast_date",
        "forecast_datetime",
    }
    feature_cols = [
        col
        for col in train_split.columns
        if col not in ignore_cols and col in df_full_test.columns
    ]
    categorical_cols = [
        col
        for col in ["county", "is_business", "product_type", "prediction_unit_id"]
        if col in feature_cols
    ]

    for cat_col in categorical_cols:
        train_split[cat_col] = train_split[cat_col].astype("category")
        val_split[cat_col] = val_split[cat_col].astype("category")
        df_full_test[cat_col] = df_full_test[cat_col].astype("category")

    print(f"Features: {len(feature_cols)} | Categorical features: {categorical_cols}")

    # ---------------------------------------------------------------------
    # Dual-Regime Model Training
    # ---------------------------------------------------------------------
    dual_model = DualRegimeEnergyModel(
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
    )

    # --- Regime 1: Solar Generation ---
    train_mask_prod = (train_split["is_consumption"] == 0).values
    val_mask_prod = (val_split["is_consumption"] == 0).values

    X_train_prod = train_split.loc[train_mask_prod, feature_cols]
    y_train_prod = train_split.loc[train_mask_prod, "target"].values
    X_val_prod = val_split.loc[val_mask_prod, feature_cols]
    y_val_prod = val_split.loc[val_mask_prod, "target"].values

    print(f"Fitting Solar Production Ensemble (N={len(X_train_prod)})...")
    dual_model.fit_prod(X_train_prod, y_train_prod, X_val_prod, y_val_prod)

    # --- Regime 2: Electrical Consumption ---
    train_mask_cons = (train_split["is_consumption"] == 1).values
    val_mask_cons = (val_split["is_consumption"] == 1).values

    X_train_cons = train_split.loc[train_mask_cons, feature_cols]
    y_train_cons = train_split.loc[train_mask_cons, "target"].values
    X_val_cons = val_split.loc[val_mask_cons, feature_cols]
    y_val_cons = val_split.loc[val_mask_cons, "target"].values

    print(f"Fitting Electrical Consumption Ensemble (N={len(X_train_cons)})...")
    dual_model.fit_cons(X_train_cons, y_train_cons, X_val_cons, y_val_cons)
    dual_model.is_fitted_ = True

    print("Calibrating optimal MAE ensemble weights on validation splits...")
    dual_model.optimize_weights(X_val_prod, y_val_prod, X_val_cons, y_val_cons)

    # ---------------------------------------------------------------------
    # Validation Evaluation
    # ---------------------------------------------------------------------
    val_preds = np.zeros(len(val_split), dtype=np.float32)
    val_preds[val_mask_prod] = dual_model.predict_prod(X_val_prod)
    val_preds[val_mask_cons] = dual_model.predict_cons(X_val_cons)

    val_solar_elev = (
        val_split["solar_elevation"].values
        if "solar_elevation" in val_split.columns
        else None
    )
    val_preds_post = dual_model.apply_domain_constraints(
        preds=val_preds,
        is_consumption=val_split["is_consumption"].values,
        solar_elevation=val_solar_elev,
    )

    prod_val_mae = competition_mae(y_val_prod, val_preds_post[val_mask_prod])
    cons_val_mae = competition_mae(y_val_cons, val_preds_post[val_mask_cons])
    final_val_score = competition_mae(val_split["target"].values, val_preds_post)

    print(f"Validation Solar Production MAE: {prod_val_mae:.4f}")
    print(f"Validation Electrical Consumption MAE: {cons_val_mae:.4f}")

    # ---------------------------------------------------------------------
    # Test Inference & Submission Generation
    # ---------------------------------------------------------------------
    test_preds = np.zeros(len(df_full_test), dtype=np.float32)
    test_mask_prod = (df_full_test["is_consumption"] == 0).values
    test_mask_cons = (df_full_test["is_consumption"] == 1).values

    if test_mask_prod.any():
        test_preds[test_mask_prod] = dual_model.predict_prod(
            df_full_test.loc[test_mask_prod, feature_cols]
        )
    if test_mask_cons.any():
        test_preds[test_mask_cons] = dual_model.predict_cons(
            df_full_test.loc[test_mask_cons, feature_cols]
        )

    test_solar_elev = (
        df_full_test["solar_elevation"].values
        if "solar_elevation" in df_full_test.columns
        else None
    )
    test_preds_post = dual_model.apply_domain_constraints(
        preds=test_preds,
        is_consumption=df_full_test["is_consumption"].values,
        solar_elevation=test_solar_elev,
    )
    test_preds_post = np.nan_to_num(test_preds_post, nan=0.0, posinf=0.0, neginf=0.0)

    # Format submission dataframe with exact required columns: id and target
    id_vals = (
        test_raw["row_id"].values
        if "row_id" in test_raw.columns
        else np.arange(len(test_raw), dtype=np.int64)
    )

    submission = pd.DataFrame(
        {"id": id_vals, "target": test_preds_post.astype(np.float64)}
    )

    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(
        f"Submission successfully saved to {submission_path} ({len(submission)} rows, columns: {list(submission.columns)})"
    )

    # Required validation output
    print(f"Final Validation Score: {final_val_score}")


if __name__ == "__main__":
    main()