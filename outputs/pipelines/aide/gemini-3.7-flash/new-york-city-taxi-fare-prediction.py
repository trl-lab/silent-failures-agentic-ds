import gc
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

# Set seeds for reproducibility
np.random.seed(42)

os.makedirs("./working", exist_ok=True)

print("Loading data...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Filter valid training records within reasonable NYC taxi bounds
train_df = train_df[
    (train_df["fare_amount"] >= 2.5)
    & (train_df["fare_amount"] <= 500.0)
    & (train_df["pickup_longitude"].between(-75.0, -72.5))
    & (train_df["dropoff_longitude"].between(-75.0, -72.5))
    & (train_df["pickup_latitude"].between(40.0, 42.0))
    & (train_df["dropoff_latitude"].between(40.0, 42.0))
    & (train_df["passenger_count"] >= 0)
    & (train_df["passenger_count"] <= 10)
].reset_index(drop=True)

# Sample train set if memory is constrained (using 2,500,000 rows if train is larger)
if len(train_df) > 2500000:
    train_df = train_df.sample(n=2500000, random_state=42).reset_index(drop=True)

print(f"Train shape after filtering/sampling: {train_df.shape}")


# Feature Engineering
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    km = 6371.0 * c
    return km


def bearing_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
    return np.degrees(np.arctan2(x, y))


def extract_features(df):
    feats = pd.DataFrame(index=df.index)

    # Coordinates
    p_lon = df["pickup_longitude"].values.astype(np.float32)
    p_lat = df["pickup_latitude"].values.astype(np.float32)
    d_lon = df["dropoff_longitude"].values.astype(np.float32)
    d_lat = df["dropoff_latitude"].values.astype(np.float32)

    feats["pickup_longitude"] = p_lon
    feats["pickup_latitude"] = p_lat
    feats["dropoff_longitude"] = d_lon
    feats["dropoff_latitude"] = d_lat
    feats["passenger_count"] = df["passenger_count"].fillna(1).astype(np.int8)

    # Coordinate differences
    feats["abs_diff_longitude"] = np.abs(d_lon - p_lon)
    feats["abs_diff_latitude"] = np.abs(d_lat - p_lat)

    # Distance metrics
    feats["haversine_dist"] = haversine_np(p_lon, p_lat, d_lon, d_lat).astype(
        np.float32
    )
    feats["manhattan_dist"] = (
        feats["abs_diff_longitude"] * 84.0 + feats["abs_diff_latitude"] * 111.0
    ).astype(np.float32)
    feats["euclidean_dist"] = np.sqrt(
        feats["abs_diff_longitude"] ** 2 + feats["abs_diff_latitude"] ** 2
    ).astype(np.float32)
    feats["bearing"] = bearing_np(p_lon, p_lat, d_lon, d_lat).astype(np.float32)

    # Key NYC landmarks / airports
    landmarks = {
        "JFK": (-73.7781, 40.6413),
        "LGA": (-73.8740, 40.7769),
        "EWR": (-74.1745, 40.6895),
        "TimesSquare": (-73.9855, 40.7580),
        "WallStreet": (-74.0090, 40.7050),
    }

    for name, (l_lon, l_lat) in landmarks.items():
        feats[f"p_dist_{name}"] = haversine_np(p_lon, p_lat, l_lon, l_lat).astype(
            np.float32
        )
        feats[f"d_dist_{name}"] = haversine_np(d_lon, d_lat, l_lon, l_lat).astype(
            np.float32
        )

    # Datetime features
    dt = pd.to_datetime(df["pickup_datetime"], utc=True)
    feats["year"] = dt.dt.year.astype(np.int16)
    feats["month"] = dt.dt.month.astype(np.int8)
    feats["day"] = dt.dt.day.astype(np.int8)
    feats["dayofweek"] = dt.dt.dayofweek.astype(np.int8)
    feats["hour"] = dt.dt.hour.astype(np.int8)
    feats["minute"] = dt.dt.minute.astype(np.int8)
    feats["is_weekend"] = (feats["dayofweek"] >= 5).astype(np.int8)

    # Cyclical hour
    feats["hour_sin"] = np.sin(2 * np.pi * feats["hour"] / 24.0).astype(np.float32)
    feats["hour_cos"] = np.cos(2 * np.pi * feats["hour"] / 24.0).astype(np.float32)

    return feats


print("Extracting features...")
X_train = extract_features(train_df)
y_train = train_df["fare_amount"].values.astype(np.float32)
X_test = extract_features(test_df)

del train_df
gc.collect()

feature_cols = list(X_train.columns)
print(f"Features ({len(feature_cols)}): {feature_cols}")

# Model Training with 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_train), dtype=np.float32)
test_preds = np.zeros(len(X_test), dtype=np.float32)

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.08,
    "num_leaves": 127,
    "max_depth": -1,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "min_child_samples": 50,
    "verbosity": -1,
    "n_jobs": -1,
    "random_state": 42,
}

print("Starting 5-Fold Cross-Validation...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

    model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=1200,
        valid_sets=[dtrain, dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    val_pred = model.predict(X_va, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred
    fold_rmse = root_mean_squared_error(y_va, val_pred)
    print(f"Fold {fold + 1} RMSE: {fold_rmse:.4f}")

    test_preds += (
        model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits
    )

overall_rmse = root_mean_squared_error(y_train, oof_preds)
print(f"\nOverall Out-of-Fold RMSE: {overall_rmse:.4f}")

# Post-processing: taxi fares cannot be lower than initial base fare
test_preds = np.clip(test_preds, a_min=2.5, a_max=None)

# Format and save submission
if "id" in test_df.columns:
    submission = pd.DataFrame({"id": test_df["id"], "fare_amount": test_preds})
else:
    submission = pd.DataFrame(
        {"id": np.arange(len(test_preds)), "fare_amount": test_preds}
    )

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
print(submission.head())
