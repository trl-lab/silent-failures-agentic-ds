import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# -----------------------------
# Paths and constants
# -----------------------------
INPUT_DIR = Path("./input")
WORK_DIR = Path("./working")
SUBMISSION_DIR = Path("./submission")

WORK_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RAW_PATH = INPUT_DIR / "train.parquet"
TEST_RAW_PATH = INPUT_DIR / "test.parquet"
MODEL_PATH = WORK_DIR / "lgbm_best.txt"
METADATA_PATH = WORK_DIR / "feature_metadata.json"

RANDOM_STATE = 42
SEED = 42
VAL_FRACTION = 0.10
SMOOTHING_M = 20.0

NYC_LAT_MIN, NYC_LAT_MAX = 40.4, 41.0
NYC_LON_MIN, NYC_LON_MAX = -74.3, -73.7


# -----------------------------
# Metric
# -----------------------------
def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Official RMSLE: sqrt(mean((log1p(y_true) - log1p(y_pred))^2))."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.maximum(y_true, 0.0)
    y_pred = np.maximum(y_pred, 0.0)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


# -----------------------------
# Geometry helpers
# -----------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in kilometers."""
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * 6371.0088 * np.arcsin(np.sqrt(a))


def initial_bearing(lat1, lon1, lat2, lon2):
    """Initial bearing in radians, normalized to [0, 2*pi)."""
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    dlon = np.radians(lon2 - lon1)

    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.arctan2(y, x)
    return (bearing + 2.0 * np.pi) % (2.0 * np.pi)


# -----------------------------
# Datetime helpers
# -----------------------------
def parse_datetime(series: pd.Series) -> pd.Series:
    """Parse datetime strings robustly."""
    if series is None:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(series, errors="coerce")


def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Parse a datetime column if present; otherwise set all NaT."""
    out = df.copy()
    if col in out.columns:
        out[col] = parse_datetime(out[col])
    else:
        out[col] = pd.NaT
    return out


# -----------------------------
# Feature engineering
# -----------------------------
def add_time_features(
    df: pd.DataFrame, pickup_col: str = "pickup_datetime"
) -> pd.DataFrame:
    """Add calendar and cyclical time features from pickup time."""
    out = df.copy()

    if pickup_col not in out.columns:
        out["pickup_hour"] = 12
        out["pickup_minute"] = 0
        out["pickup_second"] = 0
        out["pickup_day"] = 1
        out["pickup_month"] = 1
        out["pickup_year"] = 2016
        out["pickup_dayofweek"] = 3
        out["pickup_is_weekend"] = 0
        out["pickup_is_rush_hour"] = 0
        out["pickup_is_night"] = 0
        out["pickup_hour_sin"] = 0.0
        out["pickup_hour_cos"] = 0.0
        out["pickup_dayofweek_sin"] = 0.0
        out["pickup_dayofweek_cos"] = 0.0
        return out

    dt = out[pickup_col]

    out["pickup_hour"] = dt.dt.hour.fillna(12).astype(np.int8)
    out["pickup_minute"] = dt.dt.minute.fillna(0).astype(np.int8)
    out["pickup_second"] = dt.dt.second.fillna(0).astype(np.int8)
    out["pickup_day"] = dt.dt.day.fillna(1).astype(np.int8)
    out["pickup_month"] = dt.dt.month.fillna(1).astype(np.int8)
    out["pickup_year"] = dt.dt.year.fillna(2016).astype(np.int16)
    out["pickup_dayofweek"] = dt.dt.dayofweek.fillna(3).astype(np.int8)
    out["pickup_is_weekend"] = (out["pickup_dayofweek"] >= 5).astype(np.int8)

    hour = out["pickup_hour"].astype(np.float64)
    out["pickup_is_rush_hour"] = (
        ((hour >= 7) & (hour < 10)) | ((hour >= 16) & (hour < 19))
    ).astype(np.int8)
    out["pickup_is_night"] = ((hour < 5) | (hour >= 23)).astype(np.int8)

    out["pickup_hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["pickup_hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)

    dow = out["pickup_dayofweek"].astype(np.float64)
    out["pickup_dayofweek_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    out["pickup_dayofweek_cos"] = np.cos(2.0 * np.pi * dow / 7.0)

    return out


def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add distance, bearing, and coarse grid features."""
    out = df.copy()

    lat1 = out["pickup_latitude"].to_numpy(dtype=np.float64)
    lon1 = out["pickup_longitude"].to_numpy(dtype=np.float64)
    lat2 = out["dropoff_latitude"].to_numpy(dtype=np.float64)
    lon2 = out["dropoff_longitude"].to_numpy(dtype=np.float64)

    out["haversine_km"] = haversine_km(lat1, lon1, lat2, lon2)
    out["haversine_m"] = out["haversine_km"] * 1000.0

    mean_lat = np.radians((lat1 + lat2) / 2.0)
    out["manhattan_approx_km"] = np.abs(lat2 - lat1) * 111.32 + np.abs(
        lon2 - lon1
    ) * 111.32 * np.cos(mean_lat)

    bearing = initial_bearing(lat1, lon1, lat2, lon2)
    out["bearing"] = bearing
    out["bearing_sin"] = np.sin(bearing)
    out["bearing_cos"] = np.cos(bearing)

    # Corrected grid encoding: scale before casting to integer.
    # 2 decimals ~ 1 km, 3 decimals ~ 100 m.
    lat1_2 = np.round(lat1 * 100.0).astype(np.int64)
    lon1_2 = np.round(lon1 * 100.0).astype(np.int64)
    lat2_2 = np.round(lat2 * 100.0).astype(np.int64)
    lon2_2 = np.round(lon2 * 100.0).astype(np.int64)

    lat1_3 = np.round(lat1 * 1000.0).astype(np.int64)
    lon1_3 = np.round(lon1 * 1000.0).astype(np.int64)
    lat2_3 = np.round(lat2 * 1000.0).astype(np.int64)
    lon2_3 = np.round(lon2 * 1000.0).astype(np.int64)

    out["pickup_lat_grid_2"] = lat1_2 * 100_000 + lon1_2
    out["dropoff_lat_grid_2"] = lat2_2 * 100_000 + lon2_2

    out["pickup_lat_grid_3"] = lat1_3 * 1_000_000 + lon1_3
    out["dropoff_lat_grid_3"] = lat2_3 * 1_000_000 + lon2_3

    # Combined coarse corridor.
    out["corridor_grid_2"] = out["pickup_lat_grid_2"].astype(
        np.int64
    ) * 1_000_000_000 + out["dropoff_lat_grid_2"].astype(np.int64)

    return out


def clean_coordinates(df: pd.DataFrame, coord_stats: dict) -> pd.DataFrame:
    """Replace invalid/out-of-city coordinates with train-fit medians."""
    out = df.copy()

    def clean_latlon(lat_col: str, lon_col: str, prefix: str) -> None:
        lat = out[lat_col].to_numpy(dtype=np.float64)
        lon = out[lon_col].to_numpy(dtype=np.float64)

        valid = (
            np.isfinite(lat)
            & np.isfinite(lon)
            & (lat >= NYC_LAT_MIN)
            & (lat <= NYC_LAT_MAX)
            & (lon >= NYC_LON_MIN)
            & (lon <= NYC_LON_MAX)
        )

        lat_med = coord_stats[f"{prefix}_lat_median"]
        lon_med = coord_stats[f"{prefix}_lon_median"]

        lat[~valid] = lat_med
        lon[~valid] = lon_med

        out[lat_col] = lat
        out[lon_col] = lon
        out[f"{prefix}_valid"] = valid.astype(np.int8)

    clean_latlon("pickup_latitude", "pickup_longitude", "pickup")
    clean_latlon("dropoff_latitude", "dropoff_longitude", "dropoff")
    return out


def fit_coordinate_stats(train_df: pd.DataFrame) -> dict:
    """Fit coordinate medians only on the train split."""
    stats = {}

    for prefix, lat_col, lon_col in [
        ("pickup", "pickup_latitude", "pickup_longitude"),
        ("dropoff", "dropoff_latitude", "dropoff_longitude"),
    ]:
        lat = train_df[lat_col].to_numpy(dtype=np.float64)
        lon = train_df[lon_col].to_numpy(dtype=np.float64)

        valid = (
            np.isfinite(lat)
            & np.isfinite(lon)
            & (lat >= NYC_LAT_MIN)
            & (lat <= NYC_LAT_MAX)
            & (lon >= NYC_LON_MIN)
            & (lon <= NYC_LON_MAX)
        )

        if valid.sum() > 0:
            stats[f"{prefix}_lat_median"] = float(np.median(lat[valid]))
            stats[f"{prefix}_lon_median"] = float(np.median(lon[valid]))
        else:
            stats[f"{prefix}_lat_median"] = 40.75
            stats[f"{prefix}_lon_median"] = -73.98

    return stats


def prepare_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode simple categorical columns."""
    out = df.copy()

    if "store_and_fwd_flag" in out.columns:
        mapping = {"N": 0, "Y": 1}
        out["store_and_fwd"] = (
            out["store_and_fwd_flag"].map(mapping).fillna(0).astype(np.int8)
        )
    else:
        out["store_and_fwd"] = 0

    if "vendor_id" in out.columns:
        out["vendor_id"] = out["vendor_id"].fillna(0).astype(np.int8)
    else:
        out["vendor_id"] = 0

    if "passenger_count" in out.columns:
        out["passenger_count"] = out["passenger_count"].fillna(1).astype(np.int8)
    else:
        out["passenger_count"] = 1

    return out


def fit_target_encoders(train_df: pd.DataFrame, target_log: np.ndarray) -> dict:
    """
    Fit smoothed target encoders on the train split only.

    Uses log1p(trip_duration) as the encoding target because the final
    metric is RMSLE. This is a leak-safe feature, not a model.
    """
    global_mean = float(np.mean(target_log))
    encoders = {"global_mean": global_mean}

    cat_cols = [
        "vendor_id",
        "store_and_fwd",
        "passenger_count",
        "pickup_hour",
        "pickup_dayofweek",
        "pickup_is_weekend",
        "pickup_is_rush_hour",
        "pickup_is_night",
        "pickup_lat_grid_2",
        "dropoff_lat_grid_2",
        "corridor_grid_2",
    ]

    tmp = pd.DataFrame({"__target_log__": target_log})
    for col in cat_cols:
        if col not in train_df.columns:
            continue
        tmp[col] = train_df[col].to_numpy()

    for col in cat_cols:
        if col not in tmp.columns:
            continue

        grouped = tmp.groupby(col)["__target_log__"].agg(["mean", "count"])
        smoothed = (grouped["count"] * grouped["mean"] + SMOOTHING_M * global_mean) / (
            grouped["count"] + SMOOTHING_M
        )

        encoders[col] = {
            "values": smoothed.to_dict(),
            "default": global_mean,
        }

    return encoders


def apply_target_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Apply train-fit target encoders to any split."""
    out = df.copy()
    global_mean = encoders.get("global_mean", 0.0)

    for col, enc in encoders.items():
        if col == "global_mean" or col not in out.columns:
            continue

        values = enc["values"]
        default = enc["default"]

        out[f"te_{col}"] = out[col].map(values).fillna(default).astype(np.float32)

    return out


def add_dropoff_time_features(df: pd.DataFrame, has_dropoff: bool) -> pd.DataFrame:
    """
    Add dropoff-time-derived features only if dropoff_datetime is safely present
    in both train and test.
    """
    out = df.copy()

    if (
        not has_dropoff
        or "dropoff_datetime" not in out.columns
        or "pickup_datetime" not in out.columns
    ):
        out["has_dropoff_datetime"] = int(has_dropoff)
        out["dropoff_hour"] = 12 if has_dropoff else 12
        out["dropoff_hour_sin"] = 0.0
        out["dropoff_hour_cos"] = 0.0
        out["time_delta_seconds"] = np.nan
        out["time_delta_log"] = np.nan
        return out

    delta = (out["dropoff_datetime"] - out["pickup_datetime"]).dt.total_seconds()
    delta = delta.replace([np.inf, -np.inf], np.nan)

    # Negative or absurdly large deltas are treated as invalid.
    delta = delta.where((delta >= 0) & (delta <= 3_600_000), other=np.nan)

    out["has_dropoff_datetime"] = 1
    out["time_delta_seconds"] = delta.astype(np.float64)
    out["time_delta_log"] = np.log1p(out["time_delta_seconds"].clip(lower=0.0))

    dropoff_hour = out["dropoff_datetime"].dt.hour.fillna(12).astype(np.int8)
    out["dropoff_hour"] = dropoff_hour
    out["dropoff_hour_sin"] = np.sin(
        2.0 * np.pi * dropoff_hour.astype(np.float64) / 24.0
    )
    out["dropoff_hour_cos"] = np.cos(
        2.0 * np.pi * dropoff_hour.astype(np.float64) / 24.0
    )

    return out


def finalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final feature matrix.
    Keeps raw and engineered features. Does not include target.
    """
    out = df.copy()

    # Drop raw datetime columns from modeling features to avoid mixed dtypes.
    out = out.drop(columns=["pickup_datetime", "dropoff_datetime"], errors="ignore")

    # Keep only numeric columns plus id if present.
    # This prevents object columns such as store_and_fwd_flag from breaking
    # numeric conversion later.
    keep_cols = [
        c for c in out.columns if c == "id" or pd.api.types.is_numeric_dtype(out[c])
    ]
    out = out[keep_cols]

    # Ensure finite numeric values.
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Fill remaining numeric NaNs with 0 for robustness.
    out[numeric_cols] = out[numeric_cols].fillna(0.0)

    return out


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    # -----------------------------
    # Load raw data
    # -----------------------------
    train_raw = pd.read_parquet(TRAIN_RAW_PATH)
    test_raw = pd.read_parquet(TEST_RAW_PATH)

    print(f"Loaded train: {train_raw.shape}, test: {test_raw.shape}")

    required_train_cols = [
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
        "trip_duration",
    ]
    missing_train = [c for c in required_train_cols if c not in train_raw.columns]
    if missing_train:
        raise ValueError(f"Train is missing required columns: {missing_train}")

    required_test_cols = [
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ]
    missing_test = [c for c in required_test_cols if c not in test_raw.columns]
    if missing_test:
        raise ValueError(f"Test is missing required columns: {missing_test}")

    # Submission id must be 0-based row index, so reset test index.
    test_raw = test_raw.reset_index(drop=True)

    # -----------------------------
    # Decide whether dropoff_datetime can be used safely.
    # It must exist in both train and test to be a valid feature.
    # -----------------------------
    has_dropoff_datetime = (
        "dropoff_datetime" in train_raw.columns
        and "dropoff_datetime" in test_raw.columns
    )

    if not has_dropoff_datetime:
        train_raw = train_raw.drop(columns=["dropoff_datetime"], errors="ignore")
        test_raw = test_raw.drop(columns=["dropoff_datetime"], errors="ignore")

    # -----------------------------
    # Parse datetimes before split if needed for stratification.
    # -----------------------------
    train_raw = ensure_datetime(train_raw, "pickup_datetime")
    if has_dropoff_datetime:
        train_raw = ensure_datetime(train_raw, "dropoff_datetime")

    # -----------------------------
    # Create validation split FIRST.
    # Stratify by log-duration buckets to keep rare long trips represented.
    # -----------------------------
    target = train_raw["trip_duration"].astype(np.float64)
    target = target.where(target > 0, other=1.0)

    log_target = np.log1p(target)
    try:
        strat = pd.qcut(log_target, q=20, labels=False, duplicates="drop")
    except ValueError:
        strat = pd.Series(0, index=train_raw.index)

    train_part, val_part = train_test_split(
        train_raw,
        test_size=VAL_FRACTION,
        random_state=RANDOM_STATE,
        stratify=strat,
    )

    train_part = train_part.reset_index(drop=True)
    val_part = val_part.reset_index(drop=True)

    print(f"Split: train={len(train_part)}, val={len(val_part)}, test={len(test_raw)}")

    # -----------------------------
    # Fit coordinate statistics on TRAIN SPLIT ONLY.
    # -----------------------------
    coord_stats = fit_coordinate_stats(train_part)

    # -----------------------------
    # Basic preprocessing: clean coordinates, parse datetimes, add features.
    # -----------------------------
    train_part = clean_coordinates(train_part, coord_stats)
    val_part = clean_coordinates(val_part, coord_stats)
    test_raw = clean_coordinates(test_raw, coord_stats)

    # Ensure datetime columns are parsed in all splits.
    train_part = ensure_datetime(train_part, "pickup_datetime")
    val_part = ensure_datetime(val_part, "pickup_datetime")
    test_raw = ensure_datetime(test_raw, "pickup_datetime")

    if has_dropoff_datetime:
        train_part = ensure_datetime(train_part, "dropoff_datetime")
        val_part = ensure_datetime(val_part, "dropoff_datetime")
        test_raw = ensure_datetime(test_raw, "dropoff_datetime")

    # Add time features.
    train_part = add_time_features(train_part)
    val_part = add_time_features(val_part)
    test_raw = add_time_features(test_raw)

    # Add spatial features.
    train_part = add_spatial_features(train_part)
    val_part = add_spatial_features(val_part)
    test_raw = add_spatial_features(test_raw)

    # Add dropoff-time features only if safe.
    train_part = add_dropoff_time_features(train_part, has_dropoff_datetime)
    val_part = add_dropoff_time_features(val_part, has_dropoff_datetime)
    test_raw = add_dropoff_time_features(test_raw, has_dropoff_datetime)

    # Prepare categorical columns.
    train_part = prepare_categoricals(train_part)
    val_part = prepare_categoricals(val_part)
    test_raw = prepare_categoricals(test_raw)

    # -----------------------------
    # Target encoding: fit on TRAIN SPLIT ONLY.
    # -----------------------------
    train_target = (
        train_part["trip_duration"]
        .astype(np.float64)
        .where(train_part["trip_duration"] > 0, other=1.0)
    )
    train_target_log = np.log1p(train_target.to_numpy(dtype=np.float64))

    target_encoders = fit_target_encoders(train_part, train_target_log)

    train_part = apply_target_encoders(train_part, target_encoders)
    val_part = apply_target_encoders(val_part, target_encoders)
    test_raw = apply_target_encoders(test_raw, target_encoders)

    # -----------------------------
    # Finalize feature matrices.
    # -----------------------------
    train_features = finalize_features(train_part)
    val_features = finalize_features(val_part)
    test_features = finalize_features(test_raw)

    # -----------------------------
    # Targets:
    # - trip_duration: raw target for official RMSLE evaluation.
    # - trip_duration_log: log1p target used for modeling.
    # -----------------------------
    train_target_raw = train_features.index.map(train_part["trip_duration"]).astype(
        np.float64
    )
    val_target_raw = val_features.index.map(val_part["trip_duration"]).astype(
        np.float64
    )

    train_target_raw = train_target_raw.where(train_target_raw > 0, other=1.0)
    val_target_raw = val_target_raw.where(val_target_raw > 0, other=1.0)

    train_targets = pd.DataFrame(
        {
            "trip_duration": train_target_raw.to_numpy(),
            "trip_duration_log": np.log1p(train_target_raw.to_numpy()),
        }
    )

    val_targets = pd.DataFrame(
        {
            "trip_duration": val_target_raw.to_numpy(),
            "trip_duration_log": np.log1p(val_target_raw.to_numpy()),
        }
    )

    # -----------------------------
    # Feature columns: identical numeric columns across all splits.
    # -----------------------------
    common_cols = (
        set(train_features.columns)
        & set(val_features.columns)
        & set(test_features.columns)
    )
    feature_columns = [
        c
        for c in train_features.columns
        if c in common_cols
        and c not in {"id", "trip_duration"}
        and pd.api.types.is_numeric_dtype(train_features[c])
    ]

    print(f"Using {len(feature_columns)} numeric features for modeling.")

    X_train = train_features[feature_columns].to_numpy(dtype=np.float32)
    X_val = val_features[feature_columns].to_numpy(dtype=np.float32)
    X_test = test_features[feature_columns].to_numpy(dtype=np.float32)

    y_train_log = train_targets["trip_duration_log"].to_numpy(dtype=np.float64)
    y_val_raw = val_targets["trip_duration"].to_numpy(dtype=np.float64)
    y_val_log = val_targets["trip_duration_log"].to_numpy(dtype=np.float64)

    # -----------------------------
    # Model:
    # LightGBM regressor on log1p(trip_duration) with L2 loss.
    # This is mathematically aligned with minimizing RMSLE^2.
    # -----------------------------
    params = {
        "objective": "regression",
        "metric": "rmse",  # RMSE in log-space == RMSLE (same quantity)
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 255,
        "max_depth": -1,
        "min_child_samples": 40,
        "min_child_weight": 1e-3,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "n_jobs": -1,
        "seed": SEED,
        "verbosity": -1,
    }

    train_ds = lgb.Dataset(X_train, label=y_train_log, free_raw_data=False)
    val_ds = lgb.Dataset(
        X_val, label=y_val_log, free_raw_data=False, reference=train_ds
    )

    model = lgb.train(
        params,
        train_ds,
        num_boost_round=2500,
        valid_sets=[val_ds],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=150, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    best_iter = int(getattr(model, "best_iteration", 0) or 0)
    if best_iter <= 0:
        best_iter = int(model.current_iteration())

    best_score = model.best_score.get("val", {}).get("rmse", float("nan"))
    print(
        f"Training done. best_iteration={best_iter}, "
        f"best_score(log-RMSE)={best_score:.6f}"
    )

    # -----------------------------
    # Validation: official RMSLE on hold-out.
    # Same predict + post-processing as test.
    # -----------------------------
    val_pred_log = model.predict(X_val, num_iteration=best_iter)
    val_pred = np.expm1(val_pred_log)
    val_pred = np.clip(val_pred, 0.0, None)
    val_pred = np.nan_to_num(val_pred, nan=0.0, posinf=0.0, neginf=0.0)

    val_rmsle = rmsle(y_val_raw, val_pred)
    print(f"Validation RMSLE (official metric): {val_rmsle:.6f}")

    # -----------------------------
    # Persist best model for auditability.
    # -----------------------------
    model.save_model(MODEL_PATH)

    # -----------------------------
    # Test inference: IDENTICAL processing to validation.
    # -----------------------------
    test_pred_log = model.predict(X_test, num_iteration=best_iter)
    test_pred = np.expm1(test_pred_log)
    test_pred = np.clip(test_pred, 0.0, None)
    test_pred = np.nan_to_num(test_pred, nan=0.0, posinf=0.0, neginf=0.0)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_pred), dtype=np.int64),
            "trip_duration": test_pred.astype(np.float64),
        }
    )
    submission.to_csv(SUBMISSION_DIR / "submission.csv", index=False)

    print(
        f"Submission written: {SUBMISSION_DIR / 'submission.csv'} "
        f"rows={len(submission)}, mean={test_pred.mean():.2f}s, "
        f"median={np.median(test_pred):.2f}s"
    )

    # -----------------------------
    # Optional metadata for auditability.
    # -----------------------------
    metadata = {
        "random_state": RANDOM_STATE,
        "val_fraction": VAL_FRACTION,
        "has_dropoff_datetime": bool(has_dropoff_datetime),
        "n_train": int(len(train_features)),
        "n_val": int(len(val_features)),
        "n_test": int(len(test_features)),
        "feature_columns": feature_columns,
        "coordinate_stats": coord_stats,
        "target_encoder_columns": [
            k for k in target_encoders.keys() if k != "global_mean"
        ],
        "metric": "rmsle",
        "target_column": "trip_duration",
        "submission_id_policy": "0-based integer row index of test.parquet",
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    # -----------------------------
    # Required final line (official RMSLE on hold-out validation set)
    # -----------------------------
    score = val_rmsle
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()
