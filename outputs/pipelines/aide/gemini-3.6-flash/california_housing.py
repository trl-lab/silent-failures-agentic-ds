import os
import warnings
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb

warnings.filterwarnings("ignore")
np.random.seed(42)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

n_train = len(train_df)
n_test = len(test_df)

y_true = train_df["medianHouseValue"].values
log_y = np.log1p(y_true)
y_min, y_max = y_true.min(), y_true.max()

train_features = train_df.drop(columns=["medianHouseValue"]).copy()
test_features = test_df.copy()
full_df = pd.concat([train_features, test_features], axis=0).reset_index(drop=True)


# Haversine distance function
def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# Base Feature Engineering
def add_static_features(df):
    data = df.copy()

    # Domain ratios
    data["rooms_per_household"] = data["totalRooms"] / (data["households"] + 1e-5)
    data["bedrooms_per_household"] = data["totalBedrooms"] / (data["households"] + 1e-5)
    data["bedrooms_per_room"] = data["totalBedrooms"] / (data["totalRooms"] + 1e-5)
    data["population_per_household"] = data["population"] / (data["households"] + 1e-5)
    data["rooms_per_person"] = data["totalRooms"] / (data["population"] + 1e-5)
    data["bedrooms_per_person"] = data["totalBedrooms"] / (data["population"] + 1e-5)
    data["income_per_household"] = data["medianIncome"] / (data["households"] + 1e-5)
    data["income_per_person"] = data["medianIncome"] / (data["population"] + 1e-5)
    data["non_bedroom_rooms"] = data["totalRooms"] - data["totalBedrooms"]
    data["non_bedroom_per_household"] = data["non_bedroom_rooms"] / (
        data["households"] + 1e-5
    )

    # Non-linear Income and Age features
    data["income_sq"] = data["medianIncome"] ** 2
    data["income_log"] = np.log1p(data["medianIncome"])
    data["income_age_interaction"] = data["medianIncome"] * data["housingMedianAge"]

    # Coordinate Rotations
    for angle in [15, 30, 45, 60, 75]:
        rad = np.radians(angle)
        data[f"rot_{angle}_x"] = data["latitude"] * np.cos(rad) + data[
            "longitude"
        ] * np.sin(rad)
        data[f"rot_{angle}_y"] = -data["latitude"] * np.sin(rad) + data[
            "longitude"
        ] * np.cos(rad)

    # Distances to Major California Cities
    cities = {
        "LA": (34.0522, -118.2437),
        "SF": (37.7749, -122.4194),
        "SJ": (37.3382, -121.8863),
        "SD": (32.7157, -117.1611),
        "SAC": (38.5816, -121.4944),
        "FRES": (36.7468, -119.7726),
        "BAK": (35.3733, -119.0187),
        "OAK": (37.8044, -122.2712),
        "SV": (37.3861, -122.0839),
    }

    for city_name, (clat, clon) in cities.items():
        data[f"dist_{city_name}"] = haversine_distance(
            data["latitude"].values, data["longitude"].values, clat, clon
        )

    data["min_dist_city"] = data[[f"dist_{c}" for c in cities.keys()]].min(axis=1)

    # Coastline Distance Approximation
    coast_points = np.array(
        [
            [42.0, -124.2],
            [41.0, -124.1],
            [40.0, -124.1],
            [39.0, -123.7],
            [38.0, -123.0],
            [37.0, -122.4],
            [36.0, -121.6],
            [35.0, -120.6],
            [34.0, -118.8],
            [33.5, -117.8],
            [32.5, -117.1],
        ]
    )

    coast_dists = []
    coords = data[["latitude", "longitude"]].values
    for lat, lon in coords:
        dists = [
            haversine_distance(lat, lon, clat, clon) for clat, clon in coast_points
        ]
        coast_dists.append(min(dists))
    data["dist_coast"] = coast_dists

    return data


full_df_feat = add_static_features(full_df)

# Multi-resolution KMeans Clustering
coords = full_df_feat[["latitude", "longitude"]].values
for k in [30, 100]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    full_df_feat[f"kmeans_cluster_{k}"] = kmeans.fit_predict(coords)

X_train_base = full_df_feat.iloc[:n_train].reset_index(drop=True)
X_test_base = full_df_feat.iloc[n_train:].reset_index(drop=True)

# 5-Fold Cross Validation Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Models to train
model_names = ["lgb", "xgb", "cat", "hgb"]
oof_preds = {m: np.zeros(n_train) for m in model_names}
test_preds = {m: np.zeros(n_test) for m in model_names}

# Define KNN spatial aggregation columns
knn_cols_to_agg = [
    "medianIncome",
    "housingMedianAge",
    "rooms_per_household",
    "bedrooms_per_room",
]

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_base)):
    X_tr = X_train_base.iloc[train_idx].copy()
    X_va = X_train_base.iloc[val_idx].copy()
    X_te = X_test_base.copy()

    y_tr_log = log_y[train_idx]
    y_va_log = log_y[val_idx]

    # Convert coordinates to radians for exact Haversine metric in NearestNeighbors
    train_coords_rad = np.radians(X_tr[["latitude", "longitude"]].values)
    val_coords_rad = np.radians(X_va[["latitude", "longitude"]].values)
    test_coords_rad = np.radians(X_te[["latitude", "longitude"]].values)

    knn = NearestNeighbors(n_neighbors=81, metric="haversine", algorithm="ball_tree")
    knn.fit(train_coords_rad)

    # Distances in radians converted to kilometers (* 6371.0)
    tr_dists_rad, tr_indices = knn.kneighbors(train_coords_rad)
    va_dists_rad, va_indices = knn.kneighbors(val_coords_rad)
    te_dists_rad, te_indices = knn.kneighbors(test_coords_rad)

    tr_dists_km = tr_dists_rad * 6371.0
    va_dists_km = va_dists_rad * 6371.0
    te_dists_km = te_dists_rad * 6371.0

    for k_neighbors in [15, 50]:
        # Train aggregations (exclude self point index 0)
        idx_k = tr_indices[:, 1 : k_neighbors + 1]
        dist_km_k = tr_dists_km[:, 1 : k_neighbors + 1]
        weights_k = 1.0 / (dist_km_k + 0.1)

        for col in knn_cols_to_agg:
            vals = X_tr[col].values[idx_k]
            X_tr[f"knn_{col}_mean_{k_neighbors}"] = np.mean(vals, axis=1)
            X_tr[f"knn_{col}_std_{k_neighbors}"] = np.std(vals, axis=1)
            X_tr[f"knn_{col}_wmean_{k_neighbors}"] = np.sum(
                vals * weights_k, axis=1
            ) / np.sum(weights_k, axis=1)

        target_vals = y_tr_log[idx_k]
        X_tr[f"knn_target_mean_{k_neighbors}"] = np.mean(target_vals, axis=1)
        X_tr[f"knn_target_std_{k_neighbors}"] = np.std(target_vals, axis=1)
        X_tr[f"knn_target_wmean_{k_neighbors}"] = np.sum(
            target_vals * weights_k, axis=1
        ) / np.sum(weights_k, axis=1)

        # Validation aggregations
        v_idx_k = va_indices[:, :k_neighbors]
        v_dist_km_k = va_dists_km[:, :k_neighbors]
        v_weights_k = 1.0 / (v_dist_km_k + 0.1)

        for col in knn_cols_to_agg:
            vals = X_tr[col].values[v_idx_k]
            X_va[f"knn_{col}_mean_{k_neighbors}"] = np.mean(vals, axis=1)
            X_va[f"knn_{col}_std_{k_neighbors}"] = np.std(vals, axis=1)
            X_va[f"knn_{col}_wmean_{k_neighbors}"] = np.sum(
                vals * v_weights_k, axis=1
            ) / np.sum(v_weights_k, axis=1)

        v_target_vals = y_tr_log[v_idx_k]
        X_va[f"knn_target_mean_{k_neighbors}"] = np.mean(v_target_vals, axis=1)
        X_va[f"knn_target_std_{k_neighbors}"] = np.std(v_target_vals, axis=1)
        X_va[f"knn_target_wmean_{k_neighbors}"] = np.sum(
            v_target_vals * v_weights_k, axis=1
        ) / np.sum(v_weights_k, axis=1)

        # Test aggregations
        t_idx_k = te_indices[:, :k_neighbors]
        t_dist_km_k = te_dists_km[:, :k_neighbors]
        t_weights_k = 1.0 / (t_dist_km_k + 0.1)

        for col in knn_cols_to_agg:
            vals = X_tr[col].values[t_idx_k]
            X_te[f"knn_{col}_mean_{k_neighbors}"] = np.mean(vals, axis=1)
            X_te[f"knn_{col}_std_{k_neighbors}"] = np.std(vals, axis=1)
            X_te[f"knn_{col}_wmean_{k_neighbors}"] = np.sum(
                vals * t_weights_k, axis=1
            ) / np.sum(t_weights_k, axis=1)

        t_target_vals = y_tr_log[t_idx_k]
        X_te[f"knn_target_mean_{k_neighbors}"] = np.mean(t_target_vals, axis=1)
        X_te[f"knn_target_std_{k_neighbors}"] = np.std(t_target_vals, axis=1)
        X_te[f"knn_target_wmean_{k_neighbors}"] = np.sum(
            t_target_vals * t_weights_k, axis=1
        ) / np.sum(t_weights_k, axis=1)

    # Distance to Nth neighbor in km
    X_tr["dist_knn_5"] = tr_dists_km[:, 5]
    X_tr["dist_knn_20"] = tr_dists_km[:, 20]
    X_va["dist_knn_5"] = va_dists_km[:, 5]
    X_va["dist_knn_20"] = va_dists_km[:, 20]
    X_te["dist_knn_5"] = te_dists_km[:, 5]
    X_te["dist_knn_20"] = te_dists_km[:, 20]

    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr_log,
        eval_set=[(X_va, y_va_log)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_preds["lgb"][val_idx] = lgb_model.predict(X_va)
    test_preds["lgb"] += lgb_model.predict(X_te) / 5.0

    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="rmse",
        early_stopping_rounds=50,
        verbosity=0,
    )
    xgb_model.fit(X_tr, y_tr_log, eval_set=[(X_va, y_va_log)], verbose=False)
    oof_preds["xgb"][val_idx] = xgb_model.predict(X_va)
    test_preds["xgb"] += xgb_model.predict(X_te) / 5.0

    # CatBoost
    cat_model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.04,
        depth=7,
        random_seed=42,
        early_stopping_rounds=50,
        verbose=0,
    )
    cat_model.fit(X_tr, y_tr_log, eval_set=(X_va, y_va_log))
    oof_preds["cat"][val_idx] = cat_model.predict(X_va)
    test_preds["cat"] += cat_model.predict(X_te) / 5.0

    # HistGradientBoosting
    hgb_model = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.05,
        max_leaf_nodes=63,
        random_state=42,
        early_stopping=True,
    )
    hgb_model.fit(X_tr, y_tr_log)
    oof_preds["hgb"][val_idx] = hgb_model.predict(X_va)
    test_preds["hgb"] += hgb_model.predict(X_te) / 5.0

# Optimize Ensemble Blending Weights
oof_matrix = np.column_stack([oof_preds[m] for m in model_names])
test_matrix = np.column_stack([test_preds[m] for m in model_names])


def loss_func(weights):
    weights = weights / np.sum(weights)
    blend_log = oof_matrix @ weights
    blend_orig = np.clip(np.expm1(blend_log), y_min, y_max)
    return root_mean_squared_error(y_true, blend_orig)


init_weights = np.ones(len(model_names)) / len(model_names)
bounds = [(0, 1)] * len(model_names)
res = minimize(loss_func, x0=init_weights, bounds=bounds, method="SLSQP")
opt_weights = res.x / np.sum(res.x)

# Compute Final OOF Blend and Predictions
oof_blend_log = oof_matrix @ opt_weights
final_oof_preds = np.clip(np.expm1(oof_blend_log), y_min, y_max)

final_test_log = test_matrix @ opt_weights
final_test_preds = np.clip(np.expm1(final_test_log), y_min, y_max)

# Compute Final Validation Metric
val_rmse = root_mean_squared_error(y_true, final_oof_preds)
print(f"Validation RMSE: {val_rmse:.4f}")

# Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "medianHouseValue": final_test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
