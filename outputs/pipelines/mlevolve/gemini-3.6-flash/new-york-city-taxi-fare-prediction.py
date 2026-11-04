import gc
import json
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Ensure output directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

print("Starting NYC Taxi Fare Prediction Pipeline...")

# Define NYC spatial bounding box & key landmarks
NYC_BOUNDS = {"min_lon": -76.0, "max_lon": -70.0, "min_lat": 38.0, "max_lat": 42.0}

LANDMARKS = {
    "JFK": (40.6413, -73.7781),
    "LGA": (40.7769, -73.8740),
    "EWR": (40.6895, -74.1745),
    "Times_Square": (40.7580, -73.9855),
    "Grand_Central": (40.7527, -73.9772),
    "Financial_District": (40.7074, -74.0113),
}


def haversine_np(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance in kilometers between array of coordinates."""
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def manhattan_distance_np(lat1, lon1, lat2, lon2):
    """Calculate Manhattan distance (sum of lateral and horizontal Haversine components)."""
    dist_lat = haversine_np(lat1, lon1, lat2, lon1)
    dist_lon = haversine_np(lat1, lon1, lat1, lon2)
    return dist_lat + dist_lon


def bearing_np(lat1, lon1, lat2, lon2):
    """Calculate bearing angle of trip direction in degrees."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.degrees(np.arctan2(x, y))


def engineer_features(df):
    """
    Construct high-impact domain features:
    - Spatial distances (Haversine, Manhattan, Bearing)
    - Distances to key NYC hubs (JFK, LGA, EWR, Times Square, etc.)
    - 45-degree rotated coordinates aligned with Manhattan grid
    - Temporal features (Year, Month, Hour, DayOfWeek, Cyclic transformations)
    - Coordinate delta metrics
    """
    features = pd.DataFrame(index=df.index)

    # Parse datetimes
    dt = pd.to_datetime(df["pickup_datetime"])
    features["year"] = dt.dt.year.astype("int16")
    features["month"] = dt.dt.month.astype("int8")
    features["day"] = dt.dt.day.astype("int8")
    features["dayofweek"] = dt.dt.dayofweek.astype("int8")
    features["hour"] = dt.dt.hour.astype("int8")
    features["is_weekend"] = (dt.dt.dayofweek >= 5).astype("int8")

    # Cyclical datetime encodings
    features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24.0).astype("float32")
    features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24.0).astype("float32")
    features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12.0).astype(
        "float32"
    )
    features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12.0).astype(
        "float32"
    )

    # Raw coordinates
    p_lat = df["pickup_latitude"].astype("float32").values
    p_lon = df["pickup_longitude"].astype("float32").values
    d_lat = df["dropoff_latitude"].astype("float32").values
    d_lon = df["dropoff_longitude"].astype("float32").values

    features["pickup_latitude"] = p_lat
    features["pickup_longitude"] = p_lon
    features["dropoff_latitude"] = d_lat
    features["dropoff_longitude"] = d_lon

    # Passenger count
    features["passenger_count"] = df["passenger_count"].fillna(1).astype("int8")

    # Coordinate differences
    features["abs_lat_diff"] = np.abs(d_lat - p_lat).astype("float32")
    features["abs_lon_diff"] = np.abs(d_lon - p_lon).astype("float32")

    # Spatial Distance Metrics
    features["distance_haversine"] = haversine_np(p_lat, p_lon, d_lat, d_lon).astype(
        "float32"
    )
    features["distance_manhattan"] = manhattan_distance_np(
        p_lat, p_lon, d_lat, d_lon
    ).astype("float32")
    features["bearing"] = bearing_np(p_lat, p_lon, d_lat, d_lon).astype("float32")

    # Manhattan grid rotation (45 degrees rotation)
    angle = np.radians(45.0)
    features["rot_pickup_x"] = (p_lon * np.cos(angle) - p_lat * np.sin(angle)).astype(
        "float32"
    )
    features["rot_pickup_y"] = (p_lon * np.sin(angle) + p_lat * np.cos(angle)).astype(
        "float32"
    )
    features["rot_dropoff_x"] = (d_lon * np.cos(angle) - d_lat * np.sin(angle)).astype(
        "float32"
    )
    features["rot_dropoff_y"] = (d_lon * np.sin(angle) + d_lat * np.cos(angle)).astype(
        "float32"
    )

    # Landmark Distances
    for name, (l_lat, l_lon) in LANDMARKS.items():
        features[f"pickup_dist_{name}"] = haversine_np(
            p_lat, p_lon, l_lat, l_lon
        ).astype("float32")
        features[f"dropoff_dist_{name}"] = haversine_np(
            d_lat, d_lon, l_lat, l_lon
        ).astype("float32")

    return features


# 1. Load Raw Datasets
print("Loading parquet files from ./input/...")
df_train_raw = pd.read_parquet("./input/train.parquet")
df_test_raw = pd.read_parquet("./input/test.parquet")

print(
    f"Loaded Raw Train shape: {df_train_raw.shape}, Raw Test shape: {df_test_raw.shape}"
)

# 2. Clean Training Data (Remove Target & Spatial Outliers)
print("Cleaning train set outliers...")
valid_mask = (
    (df_train_raw["fare_amount"] >= 2.5)
    & (df_train_raw["fare_amount"] <= 500.0)
    & (
        df_train_raw["pickup_latitude"].between(
            NYC_BOUNDS["min_lat"], NYC_BOUNDS["max_lat"]
        )
    )
    & (
        df_train_raw["dropoff_latitude"].between(
            NYC_BOUNDS["min_lat"], NYC_BOUNDS["max_lat"]
        )
    )
    & (
        df_train_raw["pickup_longitude"].between(
            NYC_BOUNDS["min_lon"], NYC_BOUNDS["max_lon"]
        )
    )
    & (
        df_train_raw["dropoff_longitude"].between(
            NYC_BOUNDS["min_lon"], NYC_BOUNDS["max_lon"]
        )
    )
    & (df_train_raw["passenger_count"] >= 0)
    & (df_train_raw["passenger_count"] <= 8)
)

df_train_clean = df_train_raw[valid_mask].reset_index(drop=True)
del df_train_raw
gc.collect()
print(f"Train shape after cleaning: {df_train_clean.shape}")

# 3. Train / Validation Split (85% Train / 15% Validation)
print("Creating train/validation split...")
train_df, val_df = train_test_split(df_train_clean, test_size=0.15, random_state=42)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
del df_train_clean
gc.collect()

# 4. Feature Engineering Pipeline
print("Engineering features for Train set...")
X_train = engineer_features(train_df)
y_train = train_df["fare_amount"].astype("float32").values
del train_df
gc.collect()

print("Engineering features for Validation set...")
X_val = engineer_features(val_df)
y_val = val_df["fare_amount"].astype("float32").values
del val_df
gc.collect()

print("Engineering features for Test set...")
X_test = engineer_features(df_test_raw)
test_ids = np.arange(len(df_test_raw), dtype=np.int64)
del df_test_raw
gc.collect()

feature_cols = list(X_train.columns)
print(f"Total features created: {len(feature_cols)}")

# Save feature name list for record
with open("./working/feature_names.json", "w") as f:
    json.dump(feature_cols, f)

# 5. Model Architecture & Hyperparameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.08,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}

# Construct LightGBM Datasets
dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)

# 6. Train LightGBM Regressor with Early Stopping
print("Training LightGBM Regressor...")
model = lgb.train(
    params,
    dtrain,
    num_boost_round=3000,
    valid_sets=[dtrain, dval],
    valid_names=["train", "val"],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)],
)

# Save Trained Model Checkpoint
model_path = "./working/lgb_model.txt"
model.save_model(model_path)
print(f"Model saved to {model_path}")

# 7. Local Validation Evaluation
print("Evaluating model on validation set...")
val_preds = model.predict(X_val, num_iteration=model.best_iteration)
val_preds = np.clip(
    val_preds, 2.5, None
)  # Enforce minimum NYC base fare threshold ($2.50)

val_rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
print(f"Calculated Holdout Validation RMSE: {val_rmse:.5f}")

# 8. Test Inference & Submission Generation
print("Generating test predictions...")
test_preds = model.predict(X_test, num_iteration=model.best_iteration)
test_preds = np.clip(test_preds, 2.5, None)

submission_df = pd.DataFrame({"id": test_ids, "fare_amount": test_preds})
sub_path = "./submission/submission.csv"
submission_df.to_csv(sub_path, index=False)
print(f"Submission successfully saved to {sub_path} with shape {submission_df.shape}")

# 9. Final Validation Score Output
print(f"Final Validation Score: {val_rmse}")
