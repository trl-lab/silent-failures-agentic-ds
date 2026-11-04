import gc
import math
import os
from typing import Any, Dict

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Define key NYC Landmark Coordinates (Latitude, Longitude)
NYC_LANDMARKS = {
    "jfk": (40.6413, -73.7781),
    "lga": (40.7769, -73.8740),
    "ewr": (40.6895, -74.1745),
    "times_square": (40.7580, -73.9855),
    "grand_central": (40.7527, -73.9772),
    "financial_district": (40.7075, -74.0090),
    "brooklyn_downtown": (40.6932, -73.9855),
}

# NYC Manhattan street grid rotation angle relative to true North (approx -29 degrees)
GRID_ROTATION_RAD = math.radians(-29.0)


def haversine_np(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Calculate the great-circle distance between two points in kilometers."""
    r = 6371.0  # Earth radius in km
    lat1_r, lon1_r, lat2_r, lon2_r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(
        np.sqrt(np.clip(a, 0.0, 1.0)), np.sqrt(np.clip(1.0 - a, 0.0, 1.0))
    )
    return (r * c).astype(np.float32)


def calculate_bearing_np(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Calculate the compass bearing between two points in degrees."""
    lat1_r, lon1_r, lat2_r, lon2_r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon_r = lon2_r - lon1_r
    y = np.sin(dlon_r) * np.cos(lat2_r)
    x = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(
        dlon_r
    )
    bearing = np.degrees(np.arctan2(y, x))
    return ((bearing + 360.0) % 360.0).astype(np.float32)


def extract_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Vectorized feature engineering pipeline for NYC taxi fare prediction."""
    feats = pd.DataFrame(index=df.index)

    p_lat = df["pickup_latitude"].astype(np.float32).values
    p_lon = df["pickup_longitude"].astype(np.float32).values
    d_lat = df["dropoff_latitude"].astype(np.float32).values
    d_lon = df["dropoff_longitude"].astype(np.float32).values

    # 1. Coordinate differences & midpoints
    d_lat_diff = np.abs(d_lat - p_lat)
    d_lon_diff = np.abs(d_lon - p_lon)
    feats["abs_diff_lat"] = d_lat_diff
    feats["abs_diff_lon"] = d_lon_diff
    feats["pickup_longitude"] = p_lon
    feats["pickup_latitude"] = p_lat
    feats["dropoff_longitude"] = d_lon
    feats["dropoff_latitude"] = d_lat
    feats["mid_latitude"] = (p_lat + d_lat) / 2.0
    feats["mid_longitude"] = (p_lon + d_lon) / 2.0

    # 2. Geodesic & Manhattan distances
    feats["haversine_dist"] = haversine_np(p_lat, p_lon, d_lat, d_lon)
    feats["manhattan_dist_lat"] = np.abs(d_lat - p_lat) * 111.0
    feats["manhattan_dist_lon"] = (
        np.abs(d_lon - p_lon) * 111.0 * np.cos(np.radians(feats["mid_latitude"]))
    )
    feats["manhattan_dist_km"] = (
        feats["manhattan_dist_lat"] + feats["manhattan_dist_lon"]
    )
    feats["bearing"] = calculate_bearing_np(p_lat, p_lon, d_lat, d_lon)

    # 3. Manhattan street-grid rotated distances
    cos_rot = math.cos(GRID_ROTATION_RAD)
    sin_rot = math.sin(GRID_ROTATION_RAD)
    p_lon_rot = p_lon * cos_rot - p_lat * sin_rot
    p_lat_rot = p_lon * sin_rot + p_lat * cos_rot
    d_lon_rot = d_lon * cos_rot - d_lat * sin_rot
    d_lat_rot = d_lon * sin_rot + d_lat * cos_rot

    feats["rotated_diff_lon"] = np.abs(d_lon_rot - p_lon_rot)
    feats["rotated_diff_lat"] = np.abs(d_lat_rot - p_lat_rot)
    feats["rotated_grid_dist"] = feats["rotated_diff_lon"] + feats["rotated_diff_lat"]
    feats["dist_ratio"] = (
        feats["rotated_grid_dist"] / (feats["haversine_dist"] + 1e-4)
    ).astype(np.float32)

    # 4. Proximity to major NYC landmarks & airports
    for name, (l_lat, l_lon) in NYC_LANDMARKS.items():
        feats[f"pickup_to_{name}"] = haversine_np(
            p_lat, p_lon, np.full_like(p_lat, l_lat), np.full_like(p_lon, l_lon)
        )
        feats[f"dropoff_to_{name}"] = haversine_np(
            d_lat, d_lon, np.full_like(d_lat, l_lat), np.full_like(d_lon, l_lon)
        )
        feats[f"min_dist_{name}"] = np.minimum(
            feats[f"pickup_to_{name}"], feats[f"dropoff_to_{name}"]
        )

    # Airport corridor indicators
    feats["is_jfk_trip"] = (
        (feats["pickup_to_jfk"] < 2.5) | (feats["dropoff_to_jfk"] < 2.5)
    ).astype(np.int8)
    feats["is_lga_trip"] = (
        (feats["pickup_to_lga"] < 2.0) | (feats["dropoff_to_lga"] < 2.0)
    ).astype(np.int8)
    feats["is_ewr_trip"] = (
        (feats["pickup_to_ewr"] < 2.5) | (feats["dropoff_to_ewr"] < 2.5)
    ).astype(np.int8)

    # 5. Temporal features
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    feats["year"] = dt.dt.year.astype(np.int16)
    feats["month"] = dt.dt.month.astype(np.int8)
    feats["day"] = dt.dt.day.astype(np.int8)
    feats["dayofweek"] = dt.dt.dayofweek.astype(np.int8)
    feats["hour"] = dt.dt.hour.astype(np.int8)
    feats["minute"] = dt.dt.minute.astype(np.int8)

    # Cyclical hour and day of week
    feats["hour_sin"] = np.sin(2.0 * np.pi * feats["hour"] / 24.0).astype(np.float32)
    feats["hour_cos"] = np.cos(2.0 * np.pi * feats["hour"] / 24.0).astype(np.float32)
    feats["dow_sin"] = np.sin(2.0 * np.pi * feats["dayofweek"] / 7.0).astype(np.float32)
    feats["dow_cos"] = np.cos(2.0 * np.pi * feats["dayofweek"] / 7.0).astype(np.float32)

    # NYC Surcharge Rules
    is_weekday = feats["dayofweek"] < 5
    feats["is_rush_hour"] = (
        is_weekday & (feats["hour"] >= 16) & (feats["hour"] < 20)
    ).astype(np.int8)
    feats["is_night"] = ((feats["hour"] >= 20) | (feats["hour"] < 6)).astype(np.int8)
    feats["is_weekend"] = (~is_weekday).astype(np.int8)
    feats["year_offset"] = (feats["year"] - 2009).astype(np.float32)

    # 6. Passenger count
    if "passenger_count" in df.columns:
        feats["passenger_count"] = (
            df["passenger_count"].fillna(1).clip(1, 8).astype(np.int8)
        )
    else:
        feats["passenger_count"] = 1

    # Target
    if is_train and "fare_amount" in df.columns:
        feats["fare_amount"] = df["fare_amount"].astype(np.float32)

    return feats


def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out invalid coordinates and target anomalies strictly from the train set."""
    lat_min, lat_max = 40.45, 41.80
    lon_min, lon_max = -74.50, -72.75

    valid_mask = (
        (df["fare_amount"] >= 2.50)
        & (df["fare_amount"] <= 500.0)
        & (df["pickup_latitude"].between(lat_min, lat_max))
        & (df["pickup_longitude"].between(lon_min, lon_max))
        & (df["dropoff_latitude"].between(lat_min, lat_max))
        & (df["dropoff_longitude"].between(lon_min, lon_max))
    )
    if "passenger_count" in df.columns:
        valid_mask = (
            valid_mask & (df["passenger_count"] >= 0) & (df["passenger_count"] <= 8)
        )

    return df[valid_mask].copy()


def get_lgbm_regressor_params(random_state: int = 42) -> Dict[str, Any]:
    """Returns optimal LightGBM hyperparameters for NYC Taxi Fare regression optimizing RMSE."""
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 4000,
        "learning_rate": 0.04,
        "num_leaves": 127,
        "max_depth": 10,
        "min_child_samples": 50,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_jobs": -1,
        "random_state": random_state,
        "verbose": -1,
    }


def create_lgbm_model(
    custom_params: Dict[str, Any] = None,
) -> lgb.LGBMRegressor:
    """Instantiates a LightGBM Regressor configured for RMSE optimization."""
    params = get_lgbm_regressor_params()
    if custom_params:
        params.update(custom_params)
    return lgb.LGBMRegressor(**params)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error faithfully matching competition metric."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    print("Loading raw training and test datasets...")
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")
    print(f"Raw Train shape: {train_df.shape}, Raw Test shape: {test_df.shape}")

    # Clean anomalous coordinates and targets
    train_df = clean_training_data(train_df)
    print(f"Cleaned Train shape: {train_df.shape}")

    # Hold-out validation split
    train_split, val_split = train_test_split(
        train_df, test_size=0.15, random_state=42, shuffle=True
    )
    del train_df
    gc.collect()

    print("Extracting features...")
    train_features = extract_features(train_split, is_train=True)
    del train_split
    gc.collect()

    val_features = extract_features(val_split, is_train=True)
    del val_split
    gc.collect()

    test_features = extract_features(test_df, is_train=False)
    print("Feature extraction complete.")

    ignore_cols = {"fare_amount", "id", "key", "pickup_datetime"}
    feature_cols = [c for c in train_features.columns if c not in ignore_cols]

    X_train = train_features[feature_cols]
    y_train = train_features["fare_amount"].values
    X_val = val_features[feature_cols]
    y_val = val_features["fare_amount"].values
    X_test = test_features[feature_cols]

    print(f"Features count: {len(feature_cols)}")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Model training
    model = create_lgbm_model()
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=250),
    ]

    print("Training LightGBM model...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_names=["val"],
        callbacks=callbacks,
    )

    # Save model artifact
    model_path = "./working/lgbm_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Validation evaluation
    val_preds = model.predict(X_val)
    val_preds = np.clip(val_preds, 2.50, None)
    score = calculate_rmse(y_val, val_preds)
    print(f"Validation RMSE: {score:.5f}")

    # Test set inference
    test_preds = model.predict(X_test)
    test_preds = np.clip(test_preds, 2.50, None)

    submission_id = np.arange(len(test_preds), dtype=np.int64)
    submission_df = pd.DataFrame(
        {"id": submission_id, "fare_amount": test_preds.astype(np.float32)}
    )

    submission_path = "./submission/submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}, Shape: {submission_df.shape}")

    # Final validation metric print
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()
