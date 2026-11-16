import os
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# ---------------------------------------------------------
# 1. Configuration & Reproducibility
# ---------------------------------------------------------
RANDOM_STATE = 42
N_SPLITS = 5
np.random.seed(RANDOM_STATE)

os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ---------------------------------------------------------
# 2. Data Loading & Feature Engineering
# ---------------------------------------------------------
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Strucht"
if target_col not in train_df.columns:
    for col in train_df.columns:
        if col.lower() == "strucht":
            train_df.rename(columns={col: target_col}, inplace=True)
            break

y_train = train_df[target_col].copy()
train_features = train_df.drop(columns=[target_col]).copy()
test_features = test_df.copy()

train_features["_is_train"] = 1
test_features["_is_train"] = 0
combined = pd.concat([train_features, test_features], axis=0, ignore_index=True)

# Identify column types
lat_col = None
lon_col = None
date_cols = []
numeric_cols = []
cat_cols = []

for col in combined.columns:
    if col in ["_is_train", "id", "ID"]:
        continue
    col_lower = col.lower()
    if col_lower in ["lat", "latitude", "y", "lat_deg", "lat_dd"]:
        lat_col = col
    elif col_lower in ["lon", "long", "longitude", "x", "lon_deg", "lon_dd"]:
        lon_col = col
    elif (
        pd.api.types.is_datetime64_any_dtype(combined[col])
        or "date" in col_lower
        or "time" in col_lower
        or "dt" in col_lower
    ):
        date_cols.append(col)
    elif pd.api.types.is_numeric_dtype(combined[col]):
        numeric_cols.append(col)
    else:
        cat_cols.append(col)

new_features_dict = {}

# Geospatial feature engineering
if lat_col and lon_col:
    combined[lat_col] = pd.to_numeric(combined[lat_col], errors="coerce")
    combined[lon_col] = pd.to_numeric(combined[lon_col], errors="coerce")

    train_lat_med = combined.loc[combined["_is_train"] == 1, lat_col].median()
    train_lon_med = combined.loc[combined["_is_train"] == 1, lon_col].median()
    combined[lat_col] = combined[lat_col].fillna(train_lat_med)
    combined[lon_col] = combined[lon_col].fillna(train_lon_med)

    lat_rad = np.radians(combined[lat_col])
    lon_rad = np.radians(combined[lon_col])
    new_features_dict["geo_x"] = np.cos(lat_rad) * np.cos(lon_rad)
    new_features_dict["geo_y"] = np.cos(lat_rad) * np.sin(lon_rad)
    new_features_dict["geo_z"] = np.sin(lat_rad)

    new_features_dict["geo_rot_45"] = combined[lat_col] + combined[lon_col]
    new_features_dict["geo_rot_135"] = combined[lat_col] - combined[lon_col]

    us_center_lat, us_center_lon = np.radians(39.8283), np.radians(-98.5795)
    dlat = lat_rad - us_center_lat
    dlon = lon_rad - us_center_lon
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(us_center_lat) * np.cos(lat_rad) * np.sin(dlon / 2.0) ** 2
    )
    new_features_dict["dist_to_us_center"] = (
        2 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0)) * 6371.0
    )

    train_coords = combined.loc[combined["_is_train"] == 1, [lat_col, lon_col]].values
    all_coords = combined[[lat_col, lon_col]].values

    for n_clusters in [10, 30, 100]:
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=RANDOM_STATE + n_clusters, batch_size=1024, n_init=3
        )
        kmeans.fit(train_coords)
        new_features_dict[f"geo_cluster_{n_clusters}"] = kmeans.predict(all_coords)
        cluster_centers = kmeans.cluster_centers_

        for c_idx in range(min(5, n_clusters)):
            c_lat, c_lon = cluster_centers[c_idx]
            new_features_dict[f"dist_to_cluster_{n_clusters}_{c_idx}"] = np.sqrt(
                (combined[lat_col] - c_lat) ** 2 + (combined[lon_col] - c_lon) ** 2
            )

# Temporal feature engineering
for dcol in date_cols:
    combined[dcol] = pd.to_datetime(combined[dcol], errors="coerce")
    new_features_dict[f"{dcol}_year"] = combined[dcol].dt.year.fillna(combined.loc[combined["_is_train"] == 1, dcol].dt.year.median())
    new_features_dict[f"{dcol}_month"] = combined[dcol].dt.month.fillna(combined.loc[combined["_is_train"] == 1, dcol].dt.month.median())
    new_features_dict[f"{dcol}_day"] = combined[dcol].dt.day.fillna(combined.loc[combined["_is_train"] == 1, dcol].dt.day.median())
    new_features_dict[f"{dcol}_dayofweek"] = combined[dcol].dt.dayofweek.fillna(combined.loc[combined["_is_train"] == 1, dcol].dt.dayofweek.median())
    new_features_dict[f"{dcol}_dayofyear"] = combined[dcol].dt.dayofyear.fillna(combined.loc[combined["_is_train"] == 1, dcol].dt.dayofyear.median())
    new_features_dict[f"{dcol}_quarter"] = combined[dcol].dt.quarter.fillna(combined.loc[combined["_is_train"] == 1, dcol].dt.quarter.median())

    new_features_dict[f"{dcol}_sin_month"] = np.sin(2 * np.pi * new_features_dict[f"{dcol}_month"] / 12.0)
    new_features_dict[f"{dcol}_cos_month"] = np.cos(2 * np.pi * new_features_dict[f"{dcol}_month"] / 12.0)
    new_features_dict[f"{dcol}_sin_dayofyear"] = np.sin(2 * np.pi * new_features_dict[f"{dcol}_dayofyear"] / 365.25)
    new_features_dict[f"{dcol}_cos_dayofyear"] = np.cos(2 * np.pi * new_features_dict[f"{dcol}_dayofyear"] / 365.25)

    combined.drop(columns=[dcol], inplace=True)

# Numeric transformations & physical interactions
height_elev_cols = [
    c
    for c in numeric_cols
    if any(
        k in c.lower()
        for k in ["elev", "ght", "ohag", "ohamsl", "height", "alt", "agl", "amsl"]
    )
]

elev_col = next((c for c in numeric_cols if "elev" in c.lower()), None)
ohag_col = next((c for c in numeric_cols if "ohag" in c.lower()), None)
ohamsl_col = next((c for c in numeric_cols if "ohamsl" in c.lower()), None)

for col in numeric_cols:
    combined[col] = pd.to_numeric(combined[col], errors="coerce")
    new_features_dict[f"{col}_isnan"] = combined[col].isna().astype(np.int32)
    train_med = combined.loc[combined["_is_train"] == 1, col].median()
    if pd.isna(train_med):
        train_med = 0.0
    combined[col] = combined[col].fillna(train_med)

    if (combined[col] >= 0).all():
        new_features_dict[f"{col}_log1p"] = np.log1p(combined[col])

# Physical closure residuals & geometric interactions
if elev_col and ohag_col and ohamsl_col:
    closure_ht = combined[ohamsl_col] - combined[elev_col]
    closure_diff = combined[ohag_col] - closure_ht
    new_features_dict["closure_height_amsl_elev"] = closure_ht
    new_features_dict["closure_residual_ohag_amsl_elev"] = closure_diff
    new_features_dict["closure_abs_residual"] = np.abs(closure_diff)
    new_features_dict["closure_rel_residual"] = np.abs(closure_diff) / (combined[ohag_col] + 1.0)
    new_features_dict["closure_ratio_ohag_amsl_elev"] = (combined[ohag_col] + 1.0) / (np.maximum(closure_ht, 0.0) + 1.0)
    new_features_dict["closure_log_ratio"] = np.log1p(np.maximum(combined[ohag_col], 0.0)) - np.log1p(np.maximum(closure_ht, 0.0))
    new_features_dict["closure_elev_plus_ohag"] = combined[elev_col] + combined[ohag_col]
    new_features_dict["closure_elev_plus_ohag_minus_ohamsl"] = combined[elev_col] + combined[ohag_col] - combined[ohamsl_col]
    new_features_dict["closure_ohamsl_minus_ohag"] = combined[ohamsl_col] - combined[ohag_col]
    new_features_dict["closure_ohamsl_minus_ohag_minus_elev"] = combined[ohamsl_col] - combined[ohag_col] - combined[elev_col]
    new_features_dict["closure_is_exact_match"] = (np.abs(closure_diff) < 1e-3).astype(np.int32)
    new_features_dict["closure_is_close_match"] = (np.abs(closure_diff) < 1.0).astype(np.int32)

# Measurement quantization and modulo features
for h_col in height_elev_cols:
    val = combined[h_col]
    new_features_dict[f"{h_col}_mod_1"] = val % 1.0
    new_features_dict[f"{h_col}_mod_5"] = val % 5.0
    new_features_dict[f"{h_col}_mod_10"] = val % 10.0
    new_features_dict[f"{h_col}_mod_25"] = val % 25.0
    new_features_dict[f"{h_col}_mod_50"] = val % 50.0
    new_features_dict[f"{h_col}_mod_100"] = val % 100.0
    new_features_dict[f"{h_col}_mod_ft_m"] = val % 0.3048
    new_features_dict[f"{h_col}_mod_10ft_m"] = val % 3.048
    new_features_dict[f"{h_col}_mod_25ft_m"] = val % 7.62
    new_features_dict[f"{h_col}_mod_50ft_m"] = val % 15.24
    new_features_dict[f"{h_col}_mod_100ft_m"] = val % 30.48
    new_features_dict[f"{h_col}_mod_m_ft"] = val % 3.28084
    new_features_dict[f"{h_col}_frac"] = val - np.floor(val)
    new_features_dict[f"{h_col}_round_diff"] = val - np.round(val)
    new_features_dict[f"{h_col}_is_int"] = ((val % 1.0) == 0).astype(np.int32)
    new_features_dict[f"{h_col}_is_mult_5"] = ((val % 5.0) == 0).astype(np.int32)
    new_features_dict[f"{h_col}_is_mult_10"] = ((val % 10.0) == 0).astype(np.int32)
    new_features_dict[f"{h_col}_is_mult_25"] = ((val % 25.0) == 0).astype(np.int32)
    new_features_dict[f"{h_col}_is_mult_50"] = ((val % 50.0) == 0).astype(np.int32)

if len(height_elev_cols) >= 2:
    for i in range(len(height_elev_cols)):
        for j in range(i + 1, len(height_elev_cols)):
            c1, c2 = height_elev_cols[i], height_elev_cols[j]
            new_features_dict[f"{c1}_minus_{c2}"] = combined[c1] - combined[c2]
            new_features_dict[f"{c1}_ratio_{c2}"] = (combined[c1] + 1.0) / (combined[c2] + 1.0)
            new_features_dict[f"{c1}_log_ratio_{c2}"] = np.log1p(np.maximum(combined[c1], 0.0)) - np.log1p(np.maximum(combined[c2], 0.0))
            new_features_dict[f"{c1}_abs_diff_{c2}"] = np.abs(combined[c1] - combined[c2])

# Categorical text & frequency features
for ccol in cat_cols:
    combined[ccol] = combined[ccol].astype(str).fillna("MISSING")
    new_features_dict[f"{ccol}_len"] = combined[ccol].apply(len)
    new_features_dict[f"{ccol}_word_count"] = combined[ccol].apply(lambda x: len(x.split()))

    train_counts = (
        combined.loc[combined["_is_train"] == 1, ccol]
        .value_counts(normalize=True)
        .to_dict()
    )
    new_features_dict[f"{ccol}_freq"] = combined[ccol].map(train_counts).fillna(0.0)

# Pairwise category interaction frequencies
if len(cat_cols) >= 2:
    for i in range(min(len(cat_cols), 6)):
        for j in range(i + 1, min(len(cat_cols), 6)):
            c1, c2 = cat_cols[i], cat_cols[j]
            pair_s = combined[c1].astype(str) + "_" + combined[c2].astype(str)
            pair_counts = (
                pair_s.loc[combined["_is_train"] == 1]
                .value_counts(normalize=True)
                .to_dict()
            )
            new_features_dict[f"cat_pair_{c1}_{c2}_freq"] = pair_s.map(pair_counts).fillna(0.0)

# Multi-variable group aggregation statistics
train_mask = combined["_is_train"] == 1
group_keys = [c for c in cat_cols if combined[c].nunique() <= 5000] + ["geo_cluster_10", "geo_cluster_30", "geo_cluster_100"]

for grp_col in group_keys:
    grp_series = combined[grp_col] if grp_col in combined.columns else pd.Series(new_features_dict[grp_col], index=combined.index)
    train_grp = grp_series.loc[train_mask]

    for num_col in height_elev_cols:
        train_num = combined.loc[train_mask, num_col]
        df_grp = pd.DataFrame({"grp": train_grp, "val": train_num})
        stats_df = df_grp.groupby("grp")["val"].agg(["mean", "std", "median", "min", "max"])

        mean_s = grp_series.map(stats_df["mean"]).fillna(train_num.mean())
        std_s = grp_series.map(stats_df["std"]).fillna(0.0)
        median_s = grp_series.map(stats_df["median"]).fillna(train_num.median())
        min_s = grp_series.map(stats_df["min"]).fillna(train_num.min())
        max_s = grp_series.map(stats_df["max"]).fillna(train_num.max())

        new_features_dict[f"{num_col}_by_{grp_col}_mean"] = mean_s
        new_features_dict[f"{num_col}_by_{grp_col}_std"] = std_s
        new_features_dict[f"{num_col}_by_{grp_col}_median"] = median_s
        new_features_dict[f"{num_col}_by_{grp_col}_diff_mean"] = combined[num_col] - mean_s
        new_features_dict[f"{num_col}_by_{grp_col}_diff_median"] = combined[num_col] - median_s
        new_features_dict[f"{num_col}_by_{grp_col}_range"] = max_s - min_s

for ccol in cat_cols:
    le = LabelEncoder()
    unique_vals = list(combined[ccol].unique())
    le.fit(unique_vals)
    new_features_dict[f"{ccol}_encoded"] = le.transform(combined[ccol])
    combined.drop(columns=[ccol], inplace=True)

# Concatenate all generated features at once to prevent fragmentation
new_features_df = pd.DataFrame(new_features_dict, index=combined.index)
combined = pd.concat([combined, new_features_df], axis=1)

# Split back to train and test
train_processed = (
    combined[combined["_is_train"] == 1]
    .drop(columns=["_is_train"])
    .reset_index(drop=True)
)
test_processed = (
    combined[combined["_is_train"] == 0]
    .drop(columns=["_is_train"])
    .reset_index(drop=True)
)
train_processed[target_col] = y_train.values

# Cross-Validation setup & out-of-fold target encoding
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
train_processed["fold"] = -1

for fold, (train_idx, val_idx) in enumerate(kf.split(train_processed)):
    train_processed.loc[val_idx, "fold"] = fold

oof_cat_cols = [
    c
    for c in train_processed.columns
    if c.endswith("_encoded") or c.endswith("_cluster")
]

for ccol in oof_cat_cols:
    oof_mean_col = f"{ccol}_target_mean"
    oof_std_col = f"{ccol}_target_std"
    train_processed[oof_mean_col] = np.nan
    train_processed[oof_std_col] = np.nan

    for m_val in [10, 50]:
        oof_bayes_col = f"{ccol}_target_bayes_m{m_val}"
        train_processed[oof_bayes_col] = np.nan
        test_processed[oof_bayes_col] = np.nan

    for fold in range(N_SPLITS):
        tr_idx = train_processed["fold"] != fold
        va_idx = train_processed["fold"] == fold

        tr_sub = train_processed.loc[tr_idx]
        stats = tr_sub.groupby(ccol)[target_col].agg(["count", "mean", "std"])
        global_mean = tr_sub[target_col].mean()
        global_std = tr_sub[target_col].std()

        train_processed.loc[va_idx, oof_mean_col] = (
            train_processed.loc[va_idx, ccol].map(stats["mean"]).fillna(global_mean)
        )
        train_processed.loc[va_idx, oof_std_col] = (
            train_processed.loc[va_idx, ccol]
            .map(stats["std"])
            .fillna(global_std)
            .fillna(0.0)
        )

        for m_val in [10, 50]:
            oof_bayes_col = f"{ccol}_target_bayes_m{m_val}"
            smooth_te = (stats["count"] * stats["mean"] + m_val * global_mean) / (stats["count"] + m_val)
            train_processed.loc[va_idx, oof_bayes_col] = train_processed.loc[va_idx, ccol].map(smooth_te).fillna(global_mean)

    full_stats = train_processed.groupby(ccol)[target_col].agg(["count", "mean", "std"])
    full_global_mean = train_processed[target_col].mean()
    full_global_std = train_processed[target_col].std()

    test_processed[oof_mean_col] = (
        test_processed[ccol].map(full_stats["mean"]).fillna(full_global_mean)
    )
    test_processed[oof_std_col] = (
        test_processed[ccol].map(full_stats["std"]).fillna(full_global_std).fillna(0.0)
    )

    for m_val in [10, 50]:
        oof_bayes_col = f"{ccol}_target_bayes_m{m_val}"
        full_smooth_te = (full_stats["count"] * full_stats["mean"] + m_val * full_global_mean) / (full_stats["count"] + m_val)
        test_processed[oof_bayes_col] = test_processed[ccol].map(full_smooth_te).fillna(full_global_mean)

# Out-of-fold Spatial KNN Target Aggregations & IDW Interpolation using 3D Cartesian coordinates
if "geo_x" in train_processed.columns and "geo_y" in train_processed.columns and "geo_z" in train_processed.columns:
    spatial_coords = ["geo_x", "geo_y", "geo_z"]
elif lat_col and lon_col:
    spatial_coords = [lat_col, lon_col]
else:
    spatial_coords = None

if spatial_coords:
    knn_neighbors_list = [3, 7, 15, 30, 60]
    stat_list = ["mean", "std", "min", "max", "median", "p25", "p75", "idw", "dist_mean", "dist_min", "dist_max"]
    for k_neighbors in knn_neighbors_list:
        for stat in stat_list:
            train_processed[f"spatial_knn_{k_neighbors}_{stat}"] = np.nan
            test_processed[f"spatial_knn_{k_neighbors}_{stat}"] = np.nan

    for fold in range(N_SPLITS):
        tr_idx = (train_processed["fold"] != fold).values
        va_idx = (train_processed["fold"] == fold).values

        coords_tr = train_processed.loc[tr_idx, spatial_coords].values
        y_tr = train_processed.loc[tr_idx, target_col].values
        coords_va = train_processed.loc[va_idx, spatial_coords].values

        for k_neighbors in knn_neighbors_list:
            knn = NearestNeighbors(n_neighbors=min(k_neighbors, len(coords_tr)), algorithm="auto")
            knn.fit(coords_tr)
            dists, val_indices = knn.kneighbors(coords_va)
            val_neighbor_targets = y_tr[val_indices]

            weights = 1.0 / (dists + 1e-5)
            weights_norm = weights / np.sum(weights, axis=1, keepdims=True)
            idw_val = np.sum(val_neighbor_targets * weights_norm, axis=1)

            train_processed.loc[va_idx, f"spatial_knn_{k_neighbors}_mean"] = np.mean(val_neighbor_targets, axis=1)
            train_processed.loc[va_idx, f"spatial_knn_{k_neighbors}_std"] = np.std(val_neighbor_targets, axis=1)
            train_processed.loc[va_idx, f"spatial_knn_{k_neighbors}_min"] = np.min(val_neighbor_targets, axis=1)
            train_processed.loc[va_idx, f"spatial_knn_{k_neighbors}_max"] = np.max(val_neighbor_targets, axis=1)
            train_processed.loc[va_idx, f"spatial_knn_{k_neighbors}_median"] = np.median(val_neighbor_targets, axis=1)
            train_processed.loc[va_idx, f"spatial_knn_{k_neighbors}_p25"] = np.percentile(val_neighbor_targets, 25, axis=1)
            train_processed.loc[va_idx, f"spatial_knn_{k_neighbors}_p75"] = np.percentile(val_neighbor_targets, 75, axis=1)
            train_processed.loc[va_idx, f"spatial_knn_{k_neighbors}_idw"] = idw_val
            train_processed.loc[va_idx, f"spatial_knn_{k_neighbors}_dist_mean"] = np.mean(dists, axis=1)
            train_processed.loc[va_idx, f"spatial_knn_{k_neighbors}_dist_min"] = dists[:, 0]
            train_processed.loc[va_idx, f"spatial_knn_{k_neighbors}_dist_max"] = dists[:, -1]

    coords_full = train_processed[spatial_coords].values
    y_full_vals = train_processed[target_col].values
    coords_test = test_processed[spatial_coords].values

    for k_neighbors in knn_neighbors_list:
        knn_full = NearestNeighbors(n_neighbors=min(k_neighbors, len(coords_full)), algorithm="auto")
        knn_full.fit(coords_full)
        dists, test_indices = knn_full.kneighbors(coords_test)
        test_neighbor_targets = y_full_vals[test_indices]

        weights = 1.0 / (dists + 1e-5)
        weights_norm = weights / np.sum(weights, axis=1, keepdims=True)
        idw_test = np.sum(test_neighbor_targets * weights_norm, axis=1)

        test_processed[f"spatial_knn_{k_neighbors}_mean"] = np.mean(test_neighbor_targets, axis=1)
        test_processed[f"spatial_knn_{k_neighbors}_std"] = np.std(test_neighbor_targets, axis=1)
        test_processed[f"spatial_knn_{k_neighbors}_min"] = np.min(test_neighbor_targets, axis=1)
        test_processed[f"spatial_knn_{k_neighbors}_max"] = np.max(test_neighbor_targets, axis=1)
        test_processed[f"spatial_knn_{k_neighbors}_median"] = np.median(test_neighbor_targets, axis=1)
        test_processed[f"spatial_knn_{k_neighbors}_p25"] = np.percentile(test_neighbor_targets, 25, axis=1)
        test_processed[f"spatial_knn_{k_neighbors}_p75"] = np.percentile(test_neighbor_targets, 75, axis=1)
        test_processed[f"spatial_knn_{k_neighbors}_idw"] = idw_test
        test_processed[f"spatial_knn_{k_neighbors}_dist_mean"] = np.mean(dists, axis=1)
        test_processed[f"spatial_knn_{k_neighbors}_dist_min"] = dists[:, 0]
        test_processed[f"spatial_knn_{k_neighbors}_dist_max"] = dists[:, -1]

train_processed.fillna(0.0, inplace=True)
test_processed.fillna(0.0, inplace=True)

feature_cols = [
    c for c in train_processed.columns if c not in [target_col, "fold", "id", "ID"]
]

X_full = train_processed[feature_cols].values.astype(np.float32)
y_full = train_processed[target_col].values.astype(np.float32)
X_test_raw = test_processed[feature_cols].values.astype(np.float32)

n_train = len(train_processed)
n_test = len(test_processed)


# ---------------------------------------------------------
# 3. Model Architectures & Factory Functions
# ---------------------------------------------------------
def get_lgb_model(random_state: int = 42) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=3500,
        learning_rate=0.025,
        num_leaves=63,
        max_depth=8,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def get_xgb_model(random_state: int = 42) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=3500,
        learning_rate=0.025,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )


def get_catboost_model(random_state: int = 42) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=3500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=random_state,
        verbose=False,
        early_stopping_rounds=50,
        thread_count=-1,
    )


def get_hgb_model(random_state: int = 42) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.04,
        max_iter=1500,
        max_leaf_nodes=47,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        n_iter_no_change=40,
        random_state=random_state,
    )


# ---------------------------------------------------------
# 4. Cross-Validation Training & Inference Loop
# ---------------------------------------------------------
oof_lgb = np.zeros(n_train, dtype=np.float32)
oof_xgb = np.zeros(n_train, dtype=np.float32)
oof_cb = np.zeros(n_train, dtype=np.float32)
oof_hgb = np.zeros(n_train, dtype=np.float32)

test_preds_lgb = np.zeros(n_test, dtype=np.float32)
test_preds_xgb = np.zeros(n_test, dtype=np.float32)
test_preds_cb = np.zeros(n_test, dtype=np.float32)
test_preds_hgb = np.zeros(n_test, dtype=np.float32)

for fold in range(N_SPLITS):
    train_idx = (train_processed["fold"] != fold).values
    val_idx = (train_processed["fold"] == fold).values

    X_train_fold, y_train_fold = X_full[train_idx], y_full[train_idx]
    X_val_fold, y_val_fold = X_full[val_idx], y_full[val_idx]

    # --- 4.1 LightGBM ---
    lgb_model = get_lgb_model(random_state=RANDOM_STATE + fold)
    lgb_model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    val_pred_lgb = lgb_model.predict(X_val_fold)
    oof_lgb[val_idx] = val_pred_lgb
    test_preds_lgb += lgb_model.predict(X_test_raw) / N_SPLITS

    # --- 4.2 XGBoost ---
    xgb_model = get_xgb_model(random_state=RANDOM_STATE + fold)
    xgb_model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        verbose=False,
    )
    val_pred_xgb = xgb_model.predict(X_val_fold)
    oof_xgb[val_idx] = val_pred_xgb
    test_preds_xgb += xgb_model.predict(X_test_raw) / N_SPLITS

    # --- 4.3 CatBoost ---
    cb_model = get_catboost_model(random_state=RANDOM_STATE + fold)
    cb_model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=(X_val_fold, y_val_fold),
        verbose=False,
    )
    val_pred_cb = cb_model.predict(X_val_fold)
    oof_cb[val_idx] = val_pred_cb
    test_preds_cb += cb_model.predict(X_test_raw) / N_SPLITS

    # --- 4.4 HistGradientBoostingRegressor ---
    hgb_model = get_hgb_model(random_state=RANDOM_STATE + fold)
    hgb_model.fit(X_train_fold, y_train_fold)
    val_pred_hgb = hgb_model.predict(X_val_fold)
    oof_hgb[val_idx] = val_pred_hgb
    test_preds_hgb += hgb_model.predict(X_test_raw) / N_SPLITS

# ---------------------------------------------------------
# 5. Regularized Meta-Ensembling & Submission Generation
# ---------------------------------------------------------
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cb, oof_hgb])
test_matrix = np.column_stack([test_preds_lgb, test_preds_xgb, test_preds_cb, test_preds_hgb])


def rmse_objective(weights):
    w = np.array(weights)
    w_norm = w / (np.sum(w) + 1e-12)
    blend = np.dot(oof_matrix, w_norm)
    blend_clipped = np.clip(blend, 0.0, None)
    return np.sqrt(np.mean((y_full - blend_clipped) ** 2))


init_weights = [0.25, 0.25, 0.25, 0.25]
bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    rmse_objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
best_weights = opt_res.x / np.sum(opt_res.x)

final_oof_preds = np.clip(np.dot(oof_matrix, best_weights), 0.0, None)
final_val_rmse = float(np.sqrt(np.mean((y_full - final_oof_preds) ** 2)))

final_test_preds = np.clip(np.dot(test_matrix, best_weights), 0.0, None)

test_id = (
    test_df["id"]
    if "id" in test_df.columns
    else (
        test_df["ID"]
        if "ID" in test_df.columns
        else np.arange(len(final_test_preds))
    )
)
submission_df = pd.DataFrame(
    {"id": test_id, target_col: final_test_preds}
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_val_rmse}")
