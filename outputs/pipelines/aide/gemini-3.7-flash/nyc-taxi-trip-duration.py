import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

print(f"Loading data from {train_path} and {test_path}...")
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Filter out extreme outliers in training data (trip_duration > 24 hours or < 10 seconds)
train_df = train_df[
    (train_df["trip_duration"] >= 10) & (train_df["trip_duration"] <= 86400)
].reset_index(drop=True)


# 2. Feature Engineering Function
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    km = 6367.0 * c
    return km


def dummy_manhattan_distance(lat1, lon1, lat2, lon2):
    a = haversine_np(lon1, lat1, lon1, lat2)
    b = haversine_np(lon1, lat1, lon2, lat1)
    return a + b


def bearing_array(lat1, lng1, lat2, lng2):
    avg_earth_radius = 6371.0
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    diff_lng = lng2 - lng1
    x = np.sin(diff_lng) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(diff_lng))
    return np.degrees(np.arctan2(x, y))


def extract_features(df):
    feats = pd.DataFrame(index=df.index)

    # Datetime features
    pickup_dt = pd.to_datetime(df["pickup_datetime"])
    feats["month"] = pickup_dt.dt.month.astype(np.int8)
    feats["day"] = pickup_dt.dt.day.astype(np.int8)
    feats["dayofweek"] = pickup_dt.dt.dayofweek.astype(np.int8)
    feats["hour"] = pickup_dt.dt.hour.astype(np.int8)
    feats["minute"] = pickup_dt.dt.minute.astype(np.int8)
    feats["minute_of_day"] = (feats["hour"] * 60 + feats["minute"]).astype(np.int16)
    feats["dayofyear"] = pickup_dt.dt.dayofyear.astype(np.int16)
    feats["is_weekend"] = (feats["dayofweek"] >= 5).astype(np.int8)

    # Categorical / basic features
    feats["vendor_id"] = df["vendor_id"].astype(np.int8)
    feats["passenger_count"] = df["passenger_count"].astype(np.int8)
    if "store_and_fwd_flag" in df.columns:
        feats["store_and_fwd_flag"] = (df["store_and_fwd_flag"] == "Y").astype(np.int8)
    else:
        feats["store_and_fwd_flag"] = 0

    # Spatial coordinates
    lat1, lon1 = df["pickup_latitude"].values, df["pickup_longitude"].values
    lat2, lon2 = df["dropoff_latitude"].values, df["dropoff_longitude"].values

    feats["pickup_latitude"] = lat1
    feats["pickup_longitude"] = lon1
    feats["dropoff_latitude"] = lat2
    feats["dropoff_longitude"] = lon2

    # Distance and coordinate diffs
    feats["lat_diff"] = np.abs(lat2 - lat1)
    feats["lon_diff"] = np.abs(lon2 - lon1)
    feats["haversine_dist"] = haversine_np(lon1, lat1, lon2, lat2)
    feats["manhattan_dist"] = dummy_manhattan_distance(lat1, lon1, lat2, lon2)
    feats["bearing"] = bearing_array(lat1, lon1, lat2, lon2)
    feats["center_lat"] = (lat1 + lat2) / 2.0
    feats["center_lon"] = (lon1 + lon2) / 2.0

    # Rotated coordinates (NYC grid angle approx ~29 degrees)
    theta = np.radians(29)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    feats["pickup_rot_x"] = cos_t * lon1 - sin_t * lat1
    feats["pickup_rot_y"] = sin_t * lon1 + cos_t * lat1
    feats["dropoff_rot_x"] = cos_t * lon2 - sin_t * lat2
    feats["dropoff_rot_y"] = sin_t * lon2 + cos_t * lat2

    return feats


print("Engineering features...")
X_train = extract_features(train_df)
y_train = np.log1p(train_df["trip_duration"].values)

X_test = extract_features(test_df)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# 3. Model Training with 5-Fold Cross Validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

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
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}

print("Starting 5-fold cross-validation...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_val = X_train.iloc[val_idx], y_train[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_val, reference=train_data)

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1500,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)],
    )

    val_pred = model.predict(X_va, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred

    fold_rmsle = root_mean_squared_error(y_val, val_pred)
    print(f"Fold {fold + 1} RMSLE: {fold_rmsle:.5f}")

    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

cv_rmsle = root_mean_squared_error(y_train, oof_preds)
print(f"\nOverall Out-of-Fold RMSLE: {cv_rmsle:.5f}")

# 4. Generate Submission File
os.makedirs("./working", exist_ok=True)
sub_df = pd.DataFrame(
    {"id": np.arange(len(test_df)), "trip_duration": np.expm1(test_preds)}
)

submission_path = "./working/submission.csv"
sub_df.to_csv(submission_path, index=False)
print(f"Submission saved successfully to {submission_path} with shape {sub_df.shape}")
