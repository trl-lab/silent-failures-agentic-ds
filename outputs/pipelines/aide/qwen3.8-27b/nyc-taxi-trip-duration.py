import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
import os

os.makedirs("./working", exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")


def engineer_features(df):
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])

    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_dayofweek"] = df["pickup_datetime"].dt.dayofweek
    df["pickup_month"] = df["pickup_datetime"].dt.month
    df["pickup_is_weekend"] = (df["pickup_dayofweek"] >= 5).astype(int)

    # Haversine distance
    lat1 = np.radians(df["pickup_latitude"].values)
    lon1 = np.radians(df["pickup_longitude"].values)
    lat2 = np.radians(df["dropoff_latitude"].values)
    lon2 = np.radians(df["dropoff_longitude"].values)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    df["distance_km"] = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    # Euclidean distance
    df["euclidean_dist"] = np.sqrt(
        (df["dropoff_latitude"] - df["pickup_latitude"]) ** 2
        + (df["dropoff_longitude"] - df["pickup_longitude"]) ** 2
    )

    # Bearing
    df["bearing"] = np.arctan2(
        np.sin(dlon) * np.cos(lat2),
        np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon),
    )

    # Timestamp difference in seconds
    df["ts_diff"] = (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds()

    return df


train = engineer_features(train)
test = engineer_features(test)


# Filter geographic outliers (NYC bounds with margin)
def geo_mask(df):
    return (
        (
            (df["pickup_latitude"] > 40.0)
            & (df["pickup_latitude"] < 42.0)
            & (df["pickup_longitude"] > -75.5)
            & (df["pickup_longitude"] < -72.5)
            & (df["dropoff_latitude"] > 40.0)
            & (df["dropoff_latitude"] < 42.0)
            & (df["dropoff_longitude"] > -75.5)
            & (df["dropoff_longitude"] < -72.5)
            & (df["passenger_count"] >= 0)
            & (df["passenger_count"] <= 6)
            & (df["trip_duration"] > 0)
            & (df["trip_duration"] < 100000)
        )
        if "trip_duration" in df.columns
        else (
            (df["pickup_latitude"] > 40.0)
            & (df["pickup_latitude"] < 42.0)
            & (df["pickup_longitude"] > -75.5)
            & (df["pickup_longitude"] < -72.5)
            & (df["dropoff_latitude"] > 40.0)
            & (df["dropoff_latitude"] < 42.0)
            & (df["dropoff_longitude"] > -75.5)
            & (df["dropoff_longitude"] < -72.5)
            & (df["passenger_count"] >= 0)
            & (df["passenger_count"] <= 6)
        )
    )


train = train[geo_mask(train)].reset_index(drop=True)
test = test[geo_mask(test)].reset_index(drop=True)

train = pd.get_dummies(train, columns=["store_and_fwd_flag"], drop_first=True)
test = pd.get_dummies(test, columns=["store_and_fwd_flag"], drop_first=True)

feature_cols = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "passenger_count",
    "vendor_id",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "pickup_is_weekend",
    "distance_km",
    "euclidean_dist",
    "bearing",
    "ts_diff",
    "store_and_fwd_flag_Y",
]

y = np.log1p(train["trip_duration"].values)
X = train[feature_cols].values.astype(np.float32)
X_test = test[feature_cols].values.astype(np.float32)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmsle_scores = []
test_preds = np.zeros(len(X_test))

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 255,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "num_threads": -1,
    "seed": 42,
}

for fold, (trn_idx, val_idx) in enumerate(kf.split(X)):
    dtrain = lgb.Dataset(X[trn_idx], label=y[trn_idx])
    dval = lgb.Dataset(X[val_idx], label=y[val_idx], reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    val_pred_log = model.predict(X[val_idx])
    rmsle = np.sqrt(np.mean((val_pred_log - y[val_idx]) ** 2))
    rmsle_scores.append(rmsle)
    print(f"Fold {fold}: RMSLE = {rmsle:.6f}")

    test_preds += model.predict(X_test) / 5

mean_rmsle = np.mean(rmsle_scores)
std_rmsle = np.std(rmsle_scores)
print(f"\nMean RMSLE: {mean_rmsle:.6f} +/- {std_rmsle:.6f}")

test_preds = np.expm1(test_preds)
test_preds = np.maximum(test_preds, 1)

submission = pd.DataFrame(
    {"id": range(len(test_preds)), "trip_duration": test_preds.astype(int)}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
