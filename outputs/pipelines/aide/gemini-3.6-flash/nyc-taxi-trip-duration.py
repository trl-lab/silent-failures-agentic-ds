import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb

warnings.filterwarnings("ignore")

# 1. Load Data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Clean training outliers (unrealistic trip durations and coordinates outside NYC region)
train = train[
    (train["trip_duration"] >= 10) & (train["trip_duration"] <= 36000)
].reset_index(drop=True)
train = train[
    (train["pickup_latitude"] >= 40.0)
    & (train["pickup_latitude"] <= 42.0)
    & (train["pickup_longitude"] >= -75.0)
    & (train["pickup_longitude"] <= -72.0)
    & (train["dropoff_latitude"] >= 40.0)
    & (train["dropoff_latitude"] <= 42.0)
    & (train["dropoff_longitude"] >= -75.0)
    & (train["dropoff_longitude"] <= -72.0)
].reset_index(drop=True)

train["log_trip_duration"] = np.log1p(train["trip_duration"])

# Global target mean for target encoding smoothing
global_mean = train["log_trip_duration"].mean()

# Fit Spatial Representations (PCA & MiniBatchKMeans)
pickup_coords_train = train[["pickup_latitude", "pickup_longitude"]].values
dropoff_coords_train = train[["dropoff_latitude", "dropoff_longitude"]].values
pickup_coords_test = test[["pickup_latitude", "pickup_longitude"]].values
dropoff_coords_test = test[["dropoff_latitude", "dropoff_longitude"]].values

all_coords = np.vstack(
    [pickup_coords_train, dropoff_coords_train, pickup_coords_test, dropoff_coords_test]
)

pca = PCA(n_components=2, random_state=42)
pca.fit(all_coords)

kmeans = MiniBatchKMeans(n_clusters=80, random_state=42, batch_size=10000)
kmeans.fit(all_coords)


# Distance Helpers
def haversine_np(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def dummy_manhattan_np(lat1, lon1, lat2, lon2):
    a = haversine_np(lat1, lon1, lat1, lon2)
    b = haversine_np(lat1, lon1, lat2, lon1)
    return a + b


def bearing_np(lat1, lon1, lat2, lon2):
    lng_delta_rad = np.radians(lon2 - lon1)
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(
        lng_delta_rad
    )
    return np.degrees(np.arctan2(y, x))


landmarks = {
    "jfk": (40.6413, -73.7781),
    "lga": (40.7769, -73.8740),
    "ewr": (40.6895, -74.1745),
    "times_square": (40.7580, -73.9855),
    "grand_central": (40.7527, -73.9772),
    "financial": (40.7074, -74.0113),
}


def extract_features(df):
    dt = pd.to_datetime(df["pickup_datetime"])
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["dayofyear"] = dt.dt.dayofyear
    df["minute"] = dt.dt.minute
    df["minute_of_day"] = df["hour"] * 60 + df["minute"]
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Cyclical encodings
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["sin_dow"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)

    # Distances
    p_lat, p_lon = df["pickup_latitude"].values, df["pickup_longitude"].values
    d_lat, d_lon = df["dropoff_latitude"].values, df["dropoff_longitude"].values

    df["dist_haversine"] = haversine_np(p_lat, p_lon, d_lat, d_lon)
    df["dist_manhattan"] = dummy_manhattan_np(p_lat, p_lon, d_lat, d_lon)
    df["bearing"] = bearing_np(p_lat, p_lon, d_lat, d_lon)

    # Rotated Manhattan distance (~28.9 deg grid rotation)
    angle = np.radians(28.9)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot_p_x = p_lon * cos_a - p_lat * sin_a
    rot_p_y = p_lon * sin_a + p_lat * cos_a
    rot_d_x = d_lon * cos_a - d_lat * sin_a
    rot_d_y = d_lon * sin_a + d_lat * cos_a
    df["dist_rot_manhattan"] = np.abs(rot_d_x - rot_p_x) + np.abs(rot_d_y - rot_p_y)
    df["dist_ratio"] = df["dist_haversine"] / (df["dist_manhattan"] + 1e-4)

    # Landmark proximity
    for name, (l_lat, l_lon) in landmarks.items():
        df[f"dist_pick_{name}"] = haversine_np(p_lat, p_lon, l_lat, l_lon)
        df[f"dist_drop_{name}"] = haversine_np(d_lat, d_lon, l_lat, l_lon)

    # PCA feature space
    p_coords = np.column_stack([p_lat, p_lon])
    d_coords = np.column_stack([d_lat, d_lon])
    p_pca = pca.transform(p_coords)
    d_pca = pca.transform(d_coords)

    df["pca_pick_0"] = p_pca[:, 0]
    df["pca_pick_1"] = p_pca[:, 1]
    df["pca_drop_0"] = d_pca[:, 0]
    df["pca_drop_1"] = d_pca[:, 1]
    df["pca_dist"] = np.abs(d_pca[:, 0] - p_pca[:, 0]) + np.abs(
        d_pca[:, 1] - p_pca[:, 1]
    )

    # Cluster assignments
    df["pickup_cluster"] = kmeans.predict(p_coords)
    df["dropoff_cluster"] = kmeans.predict(d_coords)
    df["cluster_pair"] = df["pickup_cluster"] * 100 + df["dropoff_cluster"]

    # Categorical/binary features
    df["vendor_id"] = df["vendor_id"].astype(int)
    df["store_and_fwd_flag"] = (df["store_and_fwd_flag"] == "Y").astype(int)

    return df


train = extract_features(train)
test = extract_features(test)

# Log speed target for spatial-temporal congestion encoding
train["log_speed"] = (
    np.log(train["dist_haversine"] + 0.001) - train["log_trip_duration"]
)
global_speed_mean = train["log_speed"].mean()


# Target Encoding Helpers
def compute_te_single(df_src, df_dst, col, target_col, g_mean, m=10):
    stats = df_src.groupby(col)[target_col].agg(["mean", "count"])
    smooth = (stats["count"] * stats["mean"] + m * g_mean) / (stats["count"] + m)
    return df_dst[col].map(smooth).fillna(g_mean).values


def compute_te_multi(df_src, df_dst, cols, target_col, g_mean, m=10):
    stats = df_src.groupby(cols)[target_col].agg(["mean", "count"]).reset_index()
    stats["te"] = (stats["count"] * stats["mean"] + m * g_mean) / (stats["count"] + m)
    merged = df_dst[cols].merge(stats[cols + ["te"]], on=cols, how="left")
    return merged["te"].fillna(g_mean).values


base_feature_cols = (
    [
        "hour",
        "dayofweek",
        "month",
        "day",
        "dayofyear",
        "minute",
        "minute_of_day",
        "is_weekend",
        "sin_hour",
        "cos_hour",
        "sin_dow",
        "cos_dow",
        "passenger_count",
        "vendor_id",
        "store_and_fwd_flag",
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
        "dist_haversine",
        "dist_manhattan",
        "dist_rot_manhattan",
        "bearing",
        "dist_ratio",
        "pca_pick_0",
        "pca_pick_1",
        "pca_drop_0",
        "pca_drop_1",
        "pca_dist",
        "pickup_cluster",
        "dropoff_cluster",
        "cluster_pair",
    ]
    + [f"dist_pick_{k}" for k in landmarks]
    + [f"dist_drop_{k}" for k in landmarks]
)

# 5-Fold Cross Validation Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

for fold, (tr_idx, va_idx) in enumerate(kf.split(train)):
    train_fold = train.iloc[tr_idx]
    val_fold = train.iloc[va_idx]

    # Target encodings for duration
    tr_te_cp = compute_te_single(
        train_fold, train_fold, "cluster_pair", "log_trip_duration", global_mean
    )
    va_te_cp = compute_te_single(
        train_fold, val_fold, "cluster_pair", "log_trip_duration", global_mean
    )
    te_te_cp = compute_te_single(
        train_fold, test, "cluster_pair", "log_trip_duration", global_mean
    )

    tr_te_ph = compute_te_multi(
        train_fold,
        train_fold,
        ["pickup_cluster", "hour"],
        "log_trip_duration",
        global_mean,
    )
    va_te_ph = compute_te_multi(
        train_fold,
        val_fold,
        ["pickup_cluster", "hour"],
        "log_trip_duration",
        global_mean,
    )
    te_te_ph = compute_te_multi(
        train_fold, test, ["pickup_cluster", "hour"], "log_trip_duration", global_mean
    )

    # Target encodings for spatial-temporal speed
    tr_te_sp_cp = compute_te_single(
        train_fold, train_fold, "cluster_pair", "log_speed", global_speed_mean
    )
    va_te_sp_cp = compute_te_single(
        train_fold, val_fold, "cluster_pair", "log_speed", global_speed_mean
    )
    te_te_sp_cp = compute_te_single(
        train_fold, test, "cluster_pair", "log_speed", global_speed_mean
    )

    tr_te_sp_ph = compute_te_multi(
        train_fold,
        train_fold,
        ["pickup_cluster", "hour"],
        "log_speed",
        global_speed_mean,
    )
    va_te_sp_ph = compute_te_multi(
        train_fold,
        val_fold,
        ["pickup_cluster", "hour"],
        "log_speed",
        global_speed_mean,
    )
    te_te_sp_ph = compute_te_multi(
        train_fold, test, ["pickup_cluster", "hour"], "log_speed", global_speed_mean
    )

    tr_te_sp_dh = compute_te_multi(
        train_fold,
        train_fold,
        ["dropoff_cluster", "hour"],
        "log_speed",
        global_speed_mean,
    )
    va_te_sp_dh = compute_te_multi(
        train_fold,
        val_fold,
        ["dropoff_cluster", "hour"],
        "log_speed",
        global_speed_mean,
    )
    te_te_sp_dh = compute_te_multi(
        train_fold, test, ["dropoff_cluster", "hour"], "log_speed", global_speed_mean
    )

    X_tr = train_fold[base_feature_cols].copy()
    X_va = val_fold[base_feature_cols].copy()
    X_te = test[base_feature_cols].copy()

    X_tr["te_cluster_pair"] = tr_te_cp
    X_va["te_cluster_pair"] = va_te_cp
    X_te["te_cluster_pair"] = te_te_cp

    X_tr["te_pick_hour"] = tr_te_ph
    X_va["te_pick_hour"] = va_te_ph
    X_te["te_pick_hour"] = te_te_ph

    X_tr["te_speed_cp"] = tr_te_sp_cp
    X_va["te_speed_cp"] = va_te_sp_cp
    X_te["te_speed_cp"] = te_te_sp_cp

    X_tr["te_speed_ph"] = tr_te_sp_ph
    X_va["te_speed_ph"] = va_te_sp_ph
    X_te["te_speed_ph"] = te_te_sp_ph

    X_tr["te_speed_dh"] = tr_te_sp_dh
    X_va["te_speed_dh"] = va_te_sp_dh
    X_te["te_speed_dh"] = te_te_sp_dh

    # Explicit duration estimates based on spatial speed and distance
    X_tr["est_log_duration_cp"] = np.log(X_tr["dist_haversine"] + 0.001) - tr_te_sp_cp
    X_va["est_log_duration_cp"] = np.log(X_va["dist_haversine"] + 0.001) - va_te_sp_cp
    X_te["est_log_duration_cp"] = np.log(X_te["dist_haversine"] + 0.001) - te_te_sp_cp

    X_tr["est_log_duration_ph"] = np.log(X_tr["dist_haversine"] + 0.001) - tr_te_sp_ph
    X_va["est_log_duration_ph"] = np.log(X_va["dist_haversine"] + 0.001) - va_te_sp_ph
    X_te["est_log_duration_ph"] = np.log(X_te["dist_haversine"] + 0.001) - te_te_sp_ph

    y_tr = train_fold["log_trip_duration"].values
    y_va = val_fold["log_trip_duration"].values

    model = lgb.LGBMRegressor(
        n_estimators=1600,
        learning_rate=0.05,
        num_leaves=127,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
    )

    oof_preds[va_idx] = model.predict(X_va)
    test_preds += model.predict(X_te) / kf.n_splits

# Compute Metric
rmsle = root_mean_squared_error(train["log_trip_duration"], oof_preds)
print(f"Validation RMSLE: {rmsle:.5f}")

# Generate Submission File
os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame()

if "id" in test.columns:
    sub["id"] = test["id"]
else:
    sub["id"] = np.arange(len(test))

sub["trip_duration"] = np.expm1(test_preds)
sub["trip_duration"] = np.clip(sub["trip_duration"], a_min=1.0, a_max=None)
sub.to_csv("./working/submission.csv", index=False)
