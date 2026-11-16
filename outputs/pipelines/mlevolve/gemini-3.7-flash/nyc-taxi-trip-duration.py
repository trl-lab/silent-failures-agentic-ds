import os
import math
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import KFold


# ==============================================================================
# 1. SPATIAL & TEMPORAL FEATURE ENGINEERING FUNCTIONS
# ==============================================================================
def calculate_haversine(lat1, lon1, lat2, lon2):
    """Vectorized Haversine distance calculation in kilometers."""
    r = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return r * c


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Vectorized bearing calculation in degrees."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_lambda = np.radians(lon2 - lon1)

    y = np.sin(delta_lambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda)
    bearing = np.degrees(np.arctan2(y, x))
    return (bearing + 360.0) % 360.0


def rotate_coordinates(lat, lon, angle_deg=29.0):
    """Rotate coordinates to align with Manhattan orthogonal street grid."""
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    lat_rot = lat * cos_a - lon * sin_a
    lon_rot = lat * sin_a + lon * cos_a
    return lat_rot, lon_rot


def add_temporal_features(df):
    """Extract calendar, cyclical, and clock features from pickup_datetime."""
    dt = pd.to_datetime(df["pickup_datetime"])

    df["month"] = dt.dt.month.astype(np.int8)
    df["day"] = dt.dt.day.astype(np.int8)
    df["dayofweek"] = dt.dt.dayofweek.astype(np.int8)
    df["hour"] = dt.dt.hour.astype(np.int8)
    df["minute"] = dt.dt.minute.astype(np.int8)
    df["second_of_day"] = (
        df["hour"].astype(np.int32) * 3600
        + df["minute"].astype(np.int32) * 60
        + dt.dt.second.astype(np.int32)
    )
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)

    # Rush hour flags (weekday morning 7-10 and evening 16-20)
    is_weekday = df["dayofweek"] < 5
    df["is_morning_rush"] = (
        is_weekday & (df["hour"] >= 7) & (df["hour"] <= 10)
    ).astype(np.int8)
    df["is_evening_rush"] = (
        is_weekday & (df["hour"] >= 16) & (df["hour"] <= 20)
    ).astype(np.int8)
    df["is_rush_hour"] = (
        (df["is_morning_rush"] == 1) | (df["is_evening_rush"] == 1)
    ).astype(np.int8)

    # Cyclical encodings & minute-of-day
    df["minute_of_day"] = (
        df["hour"].astype(np.int32) * 60 + df["minute"].astype(np.int32)
    ).astype(np.int16)
    df["mod_sin"] = np.sin(2.0 * np.pi * df["minute_of_day"] / 1440.0).astype(
        np.float32
    )
    df["mod_cos"] = np.cos(2.0 * np.pi * df["minute_of_day"] / 1440.0).astype(
        np.float32
    )
    df["hour_sin"] = np.sin(2.0 * np.pi * df["hour"] / 24.0).astype(np.float32)
    df["hour_cos"] = np.cos(2.0 * np.pi * df["hour"] / 24.0).astype(np.float32)
    df["dow_sin"] = np.sin(2.0 * np.pi * df["dayofweek"] / 7.0).astype(np.float32)
    df["dow_cos"] = np.cos(2.0 * np.pi * df["dayofweek"] / 7.0).astype(np.float32)
    df["doy_sin"] = np.sin(2.0 * np.pi * dt.dt.dayofyear / 365.25).astype(np.float32)
    df["doy_cos"] = np.cos(2.0 * np.pi * dt.dt.dayofyear / 365.25).astype(np.float32)
    return df


def add_spatial_features(df):
    """Compute geometric, grid-aligned, and distance features."""
    p_lat = df["pickup_latitude"].values
    p_lon = df["pickup_longitude"].values
    d_lat = df["dropoff_latitude"].values
    d_lon = df["dropoff_longitude"].values

    # Raw coordinate deltas (absolute and signed)
    df["delta_lat"] = np.abs(d_lat - p_lat).astype(np.float32)
    df["delta_lon"] = np.abs(d_lon - p_lon).astype(np.float32)
    df["signed_delta_lat"] = (d_lat - p_lat).astype(np.float32)
    df["signed_delta_lon"] = (d_lon - p_lon).astype(np.float32)

    # Haversine distance & bearing
    df["haversine_dist_km"] = calculate_haversine(p_lat, p_lon, d_lat, d_lon).astype(
        np.float32
    )
    df["bearing"] = calculate_bearing(p_lat, p_lon, d_lat, d_lon).astype(np.float32)

    # Rotated grid coordinates (Manhattan grid tilt ~29 degrees)
    p_lat_rot, p_lon_rot = rotate_coordinates(p_lat, p_lon, 29.0)
    d_lat_rot, d_lon_rot = rotate_coordinates(d_lat, d_lon, 29.0)

    df["p_lat_rot"] = p_lat_rot.astype(np.float32)
    df["p_lon_rot"] = p_lon_rot.astype(np.float32)
    df["d_lat_rot"] = d_lat_rot.astype(np.float32)
    df["d_lon_rot"] = d_lon_rot.astype(np.float32)

    # Manhattan grid street & avenue distances (signed and absolute in km)
    signed_street_deg = d_lat_rot - p_lat_rot
    signed_avenue_deg = d_lon_rot - p_lon_rot
    df["signed_street_dist_km"] = (signed_street_deg * 111.0).astype(np.float32)
    df["signed_avenue_dist_km"] = (
        signed_avenue_deg * 111.0 * np.cos(np.radians(40.75))
    ).astype(np.float32)
    df["street_dist_km"] = np.abs(df["signed_street_dist_km"]).astype(np.float32)
    df["avenue_dist_km"] = np.abs(df["signed_avenue_dist_km"]).astype(np.float32)
    df["manhattan_dist_km"] = (df["street_dist_km"] + df["avenue_dist_km"]).astype(
        np.float32
    )

    # Directional quadrant on rotated Manhattan grid
    df["dir_quadrant"] = (
        (signed_street_deg >= 0).astype(np.int8) * 2
        + (signed_avenue_deg >= 0).astype(np.int8)
    ).astype(np.int8)

    # Route detour / straightness ratio
    df["route_directness"] = np.where(
        df["manhattan_dist_km"] > 1e-4,
        df["haversine_dist_km"] / df["manhattan_dist_km"],
        1.0,
    ).astype(np.float32)

    # Distances to major NYC transit hubs & landmarks
    landmarks = {
        "jfk": (40.6413, -73.7781),
        "lga": (40.7769, -73.8740),
        "ewr": (40.6895, -74.1745),
        "times_sq": (40.7580, -73.9855),
        "wall_st": (40.7075, -74.0090),
        "grand_central": (40.7527, -73.9772),
    }

    for name, (l_lat, l_lon) in landmarks.items():
        df[f"p_dist_{name}"] = calculate_haversine(p_lat, p_lon, l_lat, l_lon).astype(
            np.float32
        )
        df[f"d_dist_{name}"] = calculate_haversine(d_lat, d_lon, l_lat, l_lon).astype(
            np.float32
        )

    # Minimum distance to any airport
    df["p_min_airport_dist"] = (
        df[["p_dist_jfk", "p_dist_lga", "p_dist_ewr"]].min(axis=1).astype(np.float32)
    )
    df["d_min_airport_dist"] = (
        df[["d_dist_jfk", "d_dist_lga", "d_dist_ewr"]].min(axis=1).astype(np.float32)
    )

    return df


# ==============================================================================
# 2. EVALUATION METRIC & MODEL HYPERPARAMETERS
# ==============================================================================
def compute_rmsle(y_true, y_pred):
    """Official competition metric: Root Mean Squared Logarithmic Error (RMSLE)."""
    y_true_clipped = np.clip(y_true, 0, None)
    y_pred_clipped = np.clip(y_pred, 0, None)
    return float(
        np.sqrt(np.mean((np.log1p(y_true_clipped) - np.log1p(y_pred_clipped)) ** 2))
    )


def get_lgb_params(custom_overrides=None):
    """LightGBM hyperparameter specification optimized for NYC spatio-temporal features."""
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 150,
        "max_depth": 11,
        "min_child_samples": 40,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 4000,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if custom_overrides:
        params.update(custom_overrides)
    return params


# ==============================================================================
# 3. MAIN PIPELINE EXECUTION
# ==============================================================================
def main():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    print("Loading datasets...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # 1. Clean train data outliers strictly on train set
    train_clean_mask = (
        (train_df["trip_duration"] >= 10)
        & (train_df["trip_duration"] <= 86400)
        & (train_df["pickup_longitude"] >= -74.3)
        & (train_df["pickup_longitude"] <= -73.6)
        & (train_df["pickup_latitude"] >= 40.5)
        & (train_df["pickup_latitude"] <= 41.0)
        & (train_df["dropoff_longitude"] >= -74.3)
        & (train_df["dropoff_longitude"] <= -73.6)
        & (train_df["dropoff_latitude"] >= 40.5)
        & (train_df["dropoff_latitude"] <= 41.0)
        & (train_df["passenger_count"] >= 0)
        & (train_df["passenger_count"] <= 9)
    )
    train_df = train_df[train_clean_mask].reset_index(drop=True)
    train_df["log_trip_duration"] = np.log1p(train_df["trip_duration"].values).astype(
        np.float32
    )

    # 2. Binary and categorical basic encodings
    for df in [train_df, test_df]:
        df["store_and_fwd_flag"] = (df["store_and_fwd_flag"] == "Y").astype(np.int8)
        df["vendor_id"] = df["vendor_id"].astype(np.int8)
        df["passenger_count"] = df["passenger_count"].astype(np.int8)
        df["is_solo_passenger"] = (df["passenger_count"] == 1).astype(np.int8)

    # 3. Add temporal and spatial features
    train_df = add_temporal_features(train_df)
    test_df = add_temporal_features(test_df)

    train_df = add_spatial_features(train_df)
    test_df = add_spatial_features(test_df)

    # 4. Multi-Scale Spatial Clustering (Coarse k=30, Fine k=120)
    coord_samples = np.vstack(
        [
            train_df[["pickup_latitude", "pickup_longitude"]].values,
            train_df[["dropoff_latitude", "dropoff_longitude"]].values,
        ]
    )

    kmeans_coarse = MiniBatchKMeans(
        n_clusters=30, batch_size=10000, random_state=42, n_init=3
    )
    kmeans_coarse.fit(coord_samples)

    kmeans_fine = MiniBatchKMeans(
        n_clusters=120, batch_size=10000, random_state=42, n_init=3
    )
    kmeans_fine.fit(coord_samples)

    # Apply coarse clustering
    train_df["pickup_cluster_coarse"] = kmeans_coarse.predict(
        train_df[["pickup_latitude", "pickup_longitude"]].values
    ).astype(np.int16)
    train_df["dropoff_cluster_coarse"] = kmeans_coarse.predict(
        train_df[["dropoff_latitude", "dropoff_longitude"]].values
    ).astype(np.int16)
    test_df["pickup_cluster_coarse"] = kmeans_coarse.predict(
        test_df[["pickup_latitude", "pickup_longitude"]].values
    ).astype(np.int16)
    test_df["dropoff_cluster_coarse"] = kmeans_coarse.predict(
        test_df[["dropoff_latitude", "dropoff_longitude"]].values
    ).astype(np.int16)

    # Apply fine clustering
    train_df["pickup_cluster_fine"] = kmeans_fine.predict(
        train_df[["pickup_latitude", "pickup_longitude"]].values
    ).astype(np.int16)
    train_df["dropoff_cluster_fine"] = kmeans_fine.predict(
        train_df[["dropoff_latitude", "dropoff_longitude"]].values
    ).astype(np.int16)
    test_df["pickup_cluster_fine"] = kmeans_fine.predict(
        test_df[["pickup_latitude", "pickup_longitude"]].values
    ).astype(np.int16)
    test_df["dropoff_cluster_fine"] = kmeans_fine.predict(
        test_df[["dropoff_latitude", "dropoff_longitude"]].values
    ).astype(np.int16)

    # Interaction OD pairs & Rush regime stratification
    for df in [train_df, test_df]:
        df["rush_regime"] = np.where(
            df["is_morning_rush"] == 1,
            1,
            np.where(df["is_evening_rush"] == 1, 2, 0),
        ).astype(np.int8)

    train_df["od_pair_coarse"] = (
        train_df["pickup_cluster_coarse"].astype(np.int32) * 100
        + train_df["dropoff_cluster_coarse"].astype(np.int32)
    ).astype(np.int32)
    test_df["od_pair_coarse"] = (
        test_df["pickup_cluster_coarse"].astype(np.int32) * 100
        + test_df["dropoff_cluster_coarse"].astype(np.int32)
    ).astype(np.int32)

    train_df["od_pair_fine"] = (
        train_df["pickup_cluster_fine"].astype(np.int32) * 1000
        + train_df["dropoff_cluster_fine"].astype(np.int32)
    ).astype(np.int32)
    test_df["od_pair_fine"] = (
        test_df["pickup_cluster_fine"].astype(np.int32) * 1000
        + test_df["dropoff_cluster_fine"].astype(np.int32)
    ).astype(np.int32)

    # Local traffic activity / density indicators
    traffic_density = (
        train_df.groupby(["pickup_cluster_fine", "dayofweek", "hour"])
        .size()
        .reset_index(name="pickup_traffic_density")
    )
    traffic_density["pickup_traffic_density"] = np.log1p(
        traffic_density["pickup_traffic_density"]
    ).astype(np.float32)

    train_df = train_df.merge(
        traffic_density, on=["pickup_cluster_fine", "dayofweek", "hour"], how="left"
    )
    train_df["pickup_traffic_density"] = (
        train_df["pickup_traffic_density"].fillna(0.0).astype(np.float32)
    )

    test_df = test_df.merge(
        traffic_density, on=["pickup_cluster_fine", "dayofweek", "hour"], how="left"
    )
    test_df["pickup_traffic_density"] = (
        test_df["pickup_traffic_density"].fillna(0.0).astype(np.float32)
    )

    # 5. Out-of-fold Hierarchical Bayesian Target Encodings (Strictly Leakage-Free)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_df["fold"] = -1

    train_df["trip_speed_kmh"] = np.where(
        train_df["trip_duration"] > 0,
        (train_df["haversine_dist_km"] / (train_df["trip_duration"] / 3600.0)).clip(
            1.0, 120.0
        ),
        20.0,
    )
    global_mean_log_dur = float(train_df["log_trip_duration"].mean())
    global_mean_speed = float(train_df["trip_speed_kmh"].mean())

    def get_smooth_te(trn_df, val_df, group_cols, target_col, global_mean, weight=20.0):
        stats = (
            trn_df.groupby(group_cols)[target_col]
            .agg(["count", "mean"])
            .reset_index()
        )
        stats["smooth_val"] = (
            stats["count"] * stats["mean"] + weight * global_mean
        ) / (stats["count"] + weight)
        keep_cols = group_cols + ["smooth_val"]
        merged = val_df[group_cols].merge(stats[keep_cols], on=group_cols, how="left")
        return merged["smooth_val"].fillna(global_mean).values.astype(np.float32)

    # Target encodings columns to populate (including rush-hour stratified OD speeds)
    te_configs = [
        (["pickup_cluster_fine", "dropoff_cluster_fine"], "oof_od_fine_log_dur", "log_trip_duration", global_mean_log_dur, 15.0),
        (["pickup_cluster_fine", "dropoff_cluster_fine"], "oof_od_fine_speed", "trip_speed_kmh", global_mean_speed, 15.0),
        (["pickup_cluster_fine", "dropoff_cluster_fine", "rush_regime"], "oof_od_rush_speed", "trip_speed_kmh", global_mean_speed, 15.0),
        (["pickup_cluster_fine", "dropoff_cluster_fine", "rush_regime"], "oof_od_rush_log_dur", "log_trip_duration", global_mean_log_dur, 15.0),
        (["pickup_cluster_fine", "hour"], "oof_pcluster_hour_log_dur", "log_trip_duration", global_mean_log_dur, 10.0),
        (["pickup_cluster_fine", "hour"], "oof_pcluster_hour_speed", "trip_speed_kmh", global_mean_speed, 10.0),
        (["pickup_cluster_coarse", "dropoff_cluster_coarse"], "oof_od_coarse_log_dur", "log_trip_duration", global_mean_log_dur, 25.0),
        (["pickup_cluster_coarse", "dropoff_cluster_coarse"], "oof_od_coarse_speed", "trip_speed_kmh", global_mean_speed, 25.0),
    ]

    for _, col_name, _, _, _ in te_configs:
        train_df[col_name] = np.nan

    for fold, (trn_idx, val_idx) in enumerate(kf.split(train_df)):
        train_df.loc[val_idx, "fold"] = fold
        trn_data = train_df.iloc[trn_idx]
        val_data = train_df.iloc[val_idx]

        for group_cols, col_name, target_col, g_mean, w in te_configs:
            train_df.loc[val_idx, col_name] = get_smooth_te(
                trn_data, val_data, group_cols, target_col, g_mean, weight=w
            )

    # Test set Bayesian target encodings using entire train dataset
    for group_cols, col_name, target_col, g_mean, w in te_configs:
        test_df[col_name] = get_smooth_te(
            train_df, test_df, group_cols, target_col, g_mean, weight=w
        )

    # Derived physics features: estimated expected log duration from predicted OD speeds
    train_df["expected_dur_from_od_speed"] = np.log1p(
        train_df["haversine_dist_km"]
        / np.clip(train_df["oof_od_fine_speed"], 1.0, 120.0)
        * 3600.0
    ).astype(np.float32)
    test_df["expected_dur_from_od_speed"] = np.log1p(
        test_df["haversine_dist_km"]
        / np.clip(test_df["oof_od_fine_speed"], 1.0, 120.0)
        * 3600.0
    ).astype(np.float32)

    train_df["expected_dur_from_rush_speed"] = np.log1p(
        train_df["haversine_dist_km"]
        / np.clip(train_df["oof_od_rush_speed"], 1.0, 120.0)
        * 3600.0
    ).astype(np.float32)
    test_df["expected_dur_from_rush_speed"] = np.log1p(
        test_df["haversine_dist_km"]
        / np.clip(test_df["oof_od_rush_speed"], 1.0, 120.0)
        * 3600.0
    ).astype(np.float32)

    train_df = train_df.drop(columns=["trip_speed_kmh"])

    # 6. Prepare Feature Sets for Modeling
    ignore_cols = {
        "id",
        "pickup_datetime",
        "dropoff_datetime",
        "trip_duration",
        "log_trip_duration",
        "fold",
    }
    feature_cols = [c for c in train_df.columns if c not in ignore_cols]
    categorical_cols = [
        c
        for c in [
            "vendor_id",
            "store_and_fwd_flag",
            "pickup_cluster_coarse",
            "dropoff_cluster_coarse",
            "pickup_cluster_fine",
            "dropoff_cluster_fine",
            "od_pair_coarse",
            "od_pair_fine",
            "month",
            "dayofweek",
            "hour",
            "is_weekend",
            "is_morning_rush",
            "is_evening_rush",
            "is_rush_hour",
            "rush_regime",
            "dir_quadrant",
        ]
        if c in feature_cols
    ]

    for col in categorical_cols:
        train_df[col] = train_df[col].astype("category")
        test_df[col] = test_df[col].astype("category")

    X_test_lgb = test_df[feature_cols].copy()
    X_test_cb = test_df[feature_cols].copy()
    X_test_xgb = test_df[feature_cols].copy()
    for col in categorical_cols:
        X_test_cb[col] = X_test_cb[col].astype(np.int32)

    n_splits = 5
    oof_preds_lgb = np.zeros(len(train_df), dtype=np.float32)
    test_preds_lgb = np.zeros(len(test_df), dtype=np.float64)

    oof_preds_cb = np.zeros(len(train_df), dtype=np.float32)
    test_preds_cb = np.zeros(len(test_df), dtype=np.float64)

    oof_preds_xgb = np.zeros(len(train_df), dtype=np.float32)
    test_preds_xgb = np.zeros(len(test_df), dtype=np.float64)

    lgb_params = get_lgb_params()

    cb_params = {
        "iterations": 3500,
        "learning_rate": 0.05,
        "depth": 8,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
        "early_stopping_rounds": 50,
    }

    xgb_params = {
        "tree_method": "hist",
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": 42,
        "n_estimators": 3500,
        "n_jobs": -1,
        "enable_categorical": True,
        "early_stopping_rounds": 50,
    }

    print(
        f"Starting 5-fold training on {len(train_df)} samples with {len(feature_cols)} features..."
    )

    for fold in range(n_splits):
        trn_mask = train_df["fold"] != fold
        val_mask = train_df["fold"] == fold

        # Data split
        X_train_fold = train_df.loc[trn_mask, feature_cols]
        y_train_fold = train_df.loc[trn_mask, "log_trip_duration"].values

        X_val_fold = train_df.loc[val_mask, feature_cols]
        y_val_fold = train_df.loc[val_mask, "log_trip_duration"].values

        # 1. Train LightGBM
        trn_ds = lgb.Dataset(
            X_train_fold,
            label=y_train_fold,
            categorical_feature=categorical_cols,
            free_raw_data=False,
        )
        val_ds = lgb.Dataset(
            X_val_fold,
            label=y_val_fold,
            reference=trn_ds,
            categorical_feature=categorical_cols,
            free_raw_data=False,
        )

        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        model_lgb = lgb.train(
            params=lgb_params,
            train_set=trn_ds,
            valid_sets=[val_ds],
            callbacks=callbacks,
        )

        val_preds_lgb = model_lgb.predict(
            X_val_fold, num_iteration=model_lgb.best_iteration
        )
        oof_preds_lgb[val_mask] = val_preds_lgb
        test_preds_lgb += (
            model_lgb.predict(X_test_lgb, num_iteration=model_lgb.best_iteration)
            / n_splits
        )

        # 2. Train CatBoost
        X_train_cb = X_train_fold.copy()
        X_val_cb = X_val_fold.copy()
        for col in categorical_cols:
            X_train_cb[col] = X_train_cb[col].astype(np.int32)
            X_val_cb[col] = X_val_cb[col].astype(np.int32)

        model_cb = CatBoostRegressor(**cb_params)
        model_cb.fit(
            X_train_cb,
            y_train_fold,
            eval_set=(X_val_cb, y_val_fold),
            use_best_model=True,
        )

        val_preds_cb = model_cb.predict(X_val_cb)
        oof_preds_cb[val_mask] = val_preds_cb
        test_preds_cb += model_cb.predict(X_test_cb) / n_splits

        # 3. Train Hist-XGBoost
        model_xgb = xgb.XGBRegressor(**xgb_params)
        model_xgb.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=False,
        )

        val_preds_xgb = model_xgb.predict(X_val_fold)
        oof_preds_xgb[val_mask] = val_preds_xgb
        test_preds_xgb += model_xgb.predict(X_test_xgb) / n_splits

        # Fold evaluation
        fold_rmsle_lgb = compute_rmsle(
            np.expm1(y_val_fold), np.clip(np.expm1(val_preds_lgb), 0, None)
        )
        fold_rmsle_cb = compute_rmsle(
            np.expm1(y_val_fold), np.clip(np.expm1(val_preds_cb), 0, None)
        )
        fold_rmsle_xgb = compute_rmsle(
            np.expm1(y_val_fold), np.clip(np.expm1(val_preds_xgb), 0, None)
        )
        print(
            f"Fold {fold} | LGB: {fold_rmsle_lgb:.5f} (iter {model_lgb.best_iteration}) | "
            f"CB: {fold_rmsle_cb:.5f} (iter {model_cb.get_best_iteration()}) | "
            f"XGB: {fold_rmsle_xgb:.5f} (iter {model_xgb.best_iteration})"
        )

    # 7. Tripartite Optimal Blending & Overall Evaluation
    y_true_sec = train_df["trip_duration"].values

    def blend_objective(weights):
        w1, w2, w3 = weights
        w_sum = w1 + w2 + w3
        if w_sum <= 0:
            return 1e9
        w1, w2, w3 = w1 / w_sum, w2 / w_sum, w3 / w_sum
        blended_log = w1 * oof_preds_lgb + w2 * oof_preds_cb + w3 * oof_preds_xgb
        blended_sec = np.clip(np.expm1(blended_log), 0, None)
        return compute_rmsle(y_true_sec, blended_sec)

    opt_result = minimize(
        blend_objective,
        x0=[0.25, 0.50, 0.25],
        bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        method="Nelder-Mead",
    )
    raw_w = np.clip(opt_result.x, 0.0, 1.0)
    w_norm = raw_w / np.sum(raw_w)
    w_lgb, w_cb, w_xgb = float(w_norm[0]), float(w_norm[1]), float(w_norm[2])

    print(
        f"Optimal Blend Weights -> LightGBM: {w_lgb:.4f}, CatBoost: {w_cb:.4f}, XGBoost: {w_xgb:.4f}"
    )

    oof_preds_log = (
        w_lgb * oof_preds_lgb + w_cb * oof_preds_cb + w_xgb * oof_preds_xgb
    )
    oof_preds_sec = np.clip(np.expm1(oof_preds_log), 0, None)
    overall_rmsle = compute_rmsle(y_true_sec, oof_preds_sec)

    # 8. Test Prediction & Submission File Generation
    test_preds_log = (
        w_lgb * test_preds_lgb + w_cb * test_preds_cb + w_xgb * test_preds_xgb
    )
    test_preds_sec = np.clip(np.expm1(test_preds_log), 1.0, None)

    if "id" in test_df.columns:
        sub_id = test_df["id"]
    else:
        sub_id = np.arange(len(test_df), dtype=np.int64)

    submission_df = pd.DataFrame({"id": sub_id, "trip_duration": test_preds_sec})

    submission_path = "./submission/submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path} with {len(submission_df)} rows.")

    print(f"Final Validation Score: {overall_rmsle}")


if __name__ == "__main__":
    main()
