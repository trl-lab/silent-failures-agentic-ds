import copy
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import hstack
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

try:
    import catboost as cb

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# Ensure working and submission directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. Load Data
# ==============================================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "BEDS"
if target_col not in train_df.columns:
    possible_targets = [c for c in train_df.columns if c.upper() == "BEDS"]
    if possible_targets:
        target_col = possible_targets[0]
    else:
        raise KeyError(f"Target column '{target_col}' not found in train.parquet")

y_train = train_df[target_col].values.astype(np.float32)
train_features_df = train_df.drop(columns=[target_col]).copy()
test_features_df = test_df.copy()

n_train = len(train_features_df)
n_test = len(test_features_df)

# ==============================================================================
# 2. Identify Column Types
# ==============================================================================
all_cols = train_features_df.columns.tolist()

lat_cols = [c for c in all_cols if c.upper() in ["LATITUDE", "LAT", "Y"]]
lon_cols = [c for c in all_cols if c.upper() in ["LONGITUDE", "LON", "LONG", "X"]]
lat_col = lat_cols[0] if lat_cols else None
lon_col = lon_cols[0] if lon_cols else None

name_cols = [
    c
    for c in all_cols
    if "NAME" in c.upper() or "FACILITY" in c.upper() or "HOSPITAL" in c.upper()
]
name_col = (
    name_cols[0]
    if name_cols
    else (
        [c for c in all_cols if train_features_df[c].dtype == "object"][0]
        if any(train_features_df[c].dtype == "object" for c in all_cols)
        else None
    )
)

cat_cols = []
num_cols = []
for col in all_cols:
    if col in [lat_col, lon_col, name_col]:
        continue
    if (
        train_features_df[col].dtype == "object"
        or train_features_df[col].dtype.name == "category"
    ):
        cat_cols.append(col)
    else:
        n_unique = train_features_df[col].nunique()
        if n_unique < 20 and train_features_df[col].dtype in [
            "int64",
            "int32",
            "int16",
            "int8",
        ]:
            cat_cols.append(col)
        else:
            num_cols.append(col)

# ==============================================================================
# 3. Stratified Fold Split Setup
# ==============================================================================
n_splits = 5
try:
    y_binned = pd.qcut(y_train, q=10, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = np.zeros(n_train, dtype=np.int32)
    for fold, (_, val_idx) in enumerate(skf.split(train_features_df, y_binned)):
        folds[val_idx] = fold
except Exception:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = np.zeros(n_train, dtype=np.int32)
    for fold, (_, val_idx) in enumerate(kf.split(train_features_df)):
        folds[val_idx] = fold

# ==============================================================================
# 4. Domain Text Feature Extraction & TF-IDF OOF Ridge Meta-Features
# ==============================================================================
keyword_patterns = {
    "kw_medical_center": r"medical\s+center|med\s+ctr",
    "kw_regional": r"regional|region",
    "kw_university": r"university|univ|academic",
    "kw_children": r"children|pediatric|shriners",
    "kw_community": r"community|comm",
    "kw_memorial": r"memorial|mem",
    "kw_health_system": r"health\s+system|healthcare|health\s+services",
    "kw_clinic": r"clinic|outpatient|ambulatory",
    "kw_rehab": r"rehab|rehabilitation|restorative",
    "kw_psych": r"psych|behavioral|mental|psychiatric|state\s+hospital",
    "kw_institute": r"institute|inst",
    "kw_general": r"general|gen",
    "kw_veteran": r"veteran|va\s+|v\.a\.|military",
    "kw_specialty": r"specialty|orthopedic|spine|heart|cancer|women",
    "kw_surgery": r"surgery|surgical|surg",
    "kw_pavilion": r"pavilion|tower|annex",
}


def extract_text_features(df: pd.DataFrame, text_column: Optional[str]) -> pd.DataFrame:
    text_feat = pd.DataFrame(index=df.index)
    if text_column and text_column in df.columns:
        text_series = df[text_column].fillna("").astype(str).str.lower()
        text_feat["name_char_len"] = text_series.apply(len).astype(np.float32)
        text_feat["name_word_count"] = text_series.apply(
            lambda x: len(x.split())
        ).astype(np.float32)
        text_feat["name_has_digit"] = text_series.apply(
            lambda x: bool(re.search(r"\d", x))
        ).astype(np.float32)

        for kw_name, pattern in keyword_patterns.items():
            text_feat[kw_name] = text_series.apply(
                lambda x: 1.0 if re.search(pattern, x) else 0.0
            ).astype(np.float32)
    return text_feat


train_text_feat = extract_text_features(train_features_df, name_col)
test_text_feat = extract_text_features(test_features_df, name_col)

# Text aggregation for TF-IDF
text_cols_all = [
    c
    for c in train_features_df.columns
    if train_features_df[c].dtype == "object"
    or "NAME" in c.upper()
    or "ADDRESS" in c.upper()
]
if not text_cols_all and name_col:
    text_cols_all = [name_col]

train_text_combined = (
    train_features_df[text_cols_all]
    .fillna("")
    .astype(str)
    .agg(" ".join, axis=1)
    .str.lower()
)
test_text_combined = (
    test_features_df[text_cols_all]
    .fillna("")
    .astype(str)
    .agg(" ".join, axis=1)
    .str.lower()
)

# TF-IDF Word and Character N-gram OOF Ridge Meta-Features
tfidf_word = TfidfVectorizer(
    ngram_range=(1, 2), max_features=5000, sublinear_tf=True
)
tfidf_char = TfidfVectorizer(
    ngram_range=(2, 4), analyzer="char", max_features=5000, sublinear_tf=True
)

X_tfidf_word_tr = tfidf_word.fit_transform(train_text_combined)
X_tfidf_word_te = tfidf_word.transform(test_text_combined)
X_tfidf_char_tr = tfidf_char.fit_transform(train_text_combined)
X_tfidf_char_te = tfidf_char.transform(test_text_combined)

X_tfidf_tr = hstack([X_tfidf_word_tr, X_tfidf_char_tr]).tocsr()
X_tfidf_te = hstack([X_tfidf_word_te, X_tfidf_char_te]).tocsr()

oof_ridge_text = np.zeros(n_train, dtype=np.float32)
test_ridge_text = np.zeros(n_test, dtype=np.float32)

for fold in range(n_splits):
    tr_idx = np.where(folds != fold)[0]
    va_idx = np.where(folds == fold)[0]

    ridge = Ridge(alpha=5.0, random_state=42)
    ridge.fit(X_tfidf_tr[tr_idx], y_train[tr_idx])
    oof_ridge_text[va_idx] = ridge.predict(X_tfidf_tr[va_idx]).astype(np.float32)
    test_ridge_text += ridge.predict(X_tfidf_te).astype(np.float32) / n_splits

train_text_feat["tfidf_ridge_pred"] = oof_ridge_text
test_text_feat["tfidf_ridge_pred"] = test_ridge_text

# 32 TruncatedSVD latent semantic components
svd = TruncatedSVD(n_components=32, random_state=42)
svd_tr = svd.fit_transform(X_tfidf_tr).astype(np.float32)
svd_te = svd.transform(X_tfidf_te).astype(np.float32)
for comp_idx in range(32):
    train_text_feat[f"tfidf_svd_{comp_idx}"] = svd_tr[:, comp_idx]
    test_text_feat[f"tfidf_svd_{comp_idx}"] = svd_te[:, comp_idx]

# ==============================================================================
# 5. Spatial Density & Spatial Target Aggregations
# ==============================================================================
spatial_train = pd.DataFrame(index=train_features_df.index)
spatial_test = pd.DataFrame(index=test_features_df.index)

if lat_col and lon_col:
    train_lat_med = train_features_df[lat_col].median()
    train_lon_med = train_features_df[lon_col].median()

    train_lat = (
        train_features_df[lat_col].fillna(train_lat_med).values.astype(np.float64)
    )
    train_lon = (
        train_features_df[lon_col].fillna(train_lon_med).values.astype(np.float64)
    )
    test_lat = test_features_df[lat_col].fillna(train_lat_med).values.astype(np.float64)
    test_lon = test_features_df[lon_col].fillna(train_lon_med).values.astype(np.float64)

    spatial_train["coord_lat"] = train_lat.astype(np.float32)
    spatial_train["coord_lon"] = train_lon.astype(np.float32)
    spatial_test["coord_lat"] = test_lat.astype(np.float32)
    spatial_test["coord_lon"] = test_lon.astype(np.float32)

    spatial_train["coord_dist_center"] = np.sqrt(
        (train_lat - train_lat_med) ** 2 + (train_lon - train_lon_med) ** 2
    ).astype(np.float32)
    spatial_test["coord_dist_center"] = np.sqrt(
        (test_lat - train_lat_med) ** 2 + (test_lon - train_lon_med) ** 2
    ).astype(np.float32)

    for angle_deg in [30, 45, 60]:
        theta = np.radians(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        spatial_train[f"coord_rot_{angle_deg}"] = (
            train_lat * cos_t + train_lon * sin_t
        ).astype(np.float32)
        spatial_test[f"coord_rot_{angle_deg}"] = (
            test_lat * cos_t + test_lon * sin_t
        ).astype(np.float32)

    # Coordinate polynomial and polar interactions
    spatial_train["coord_poly_lat2"] = (train_lat ** 2).astype(np.float32)
    spatial_train["coord_poly_lon2"] = (train_lon ** 2).astype(np.float32)
    spatial_train["coord_poly_lat_lon"] = (train_lat * train_lon).astype(np.float32)
    spatial_train["coord_polar_r"] = np.sqrt(
        (train_lat - train_lat_med) ** 2 + (train_lon - train_lon_med) ** 2
    ).astype(np.float32)
    spatial_train["coord_polar_theta"] = np.arctan2(
        train_lat - train_lat_med, train_lon - train_lon_med
    ).astype(np.float32)

    spatial_test["coord_poly_lat2"] = (test_lat ** 2).astype(np.float32)
    spatial_test["coord_poly_lon2"] = (test_lon ** 2).astype(np.float32)
    spatial_test["coord_poly_lat_lon"] = (test_lat * test_lon).astype(np.float32)
    spatial_test["coord_polar_r"] = np.sqrt(
        (test_lat - train_lat_med) ** 2 + (test_lon - train_lon_med) ** 2
    ).astype(np.float32)
    spatial_test["coord_polar_theta"] = np.arctan2(
        test_lat - train_lat_med, test_lon - train_lon_med
    ).astype(np.float32)

    coords_train = np.column_stack([train_lat, train_lon])
    coords_test = np.column_stack([test_lat, test_lon])

    n_clusters = min(20, max(5, len(train_df) // 150))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    train_cluster_labels = kmeans.fit_predict(coords_train)
    test_cluster_labels = kmeans.predict(coords_test)

    spatial_train["spatial_cluster"] = train_cluster_labels.astype(np.int32)
    spatial_test["spatial_cluster"] = test_cluster_labels.astype(np.int32)

    spatial_train["dist_to_cluster_center"] = np.min(
        kmeans.transform(coords_train), axis=1
    ).astype(np.float32)
    spatial_test["dist_to_cluster_center"] = np.min(
        kmeans.transform(coords_test), axis=1
    ).astype(np.float32)

    # Multi-radius spatial density counts (Haversine distances in km)
    coords_rad_tr = np.radians(coords_train)
    coords_rad_te = np.radians(coords_test)
    earth_radius_km = 6371.0
    density_radii_km = [10.0, 25.0, 50.0, 100.0]

    nn_spatial = NearestNeighbors(metric="haversine", n_jobs=-1)
    nn_spatial.fit(coords_rad_tr)

    for r_km in density_radii_km:
        r_rad = r_km / earth_radius_km
        tr_indices = nn_spatial.radius_neighbors(
            coords_rad_tr, radius=r_rad, return_distance=False
        )
        te_indices = nn_spatial.radius_neighbors(
            coords_rad_te, radius=r_rad, return_distance=False
        )

        spatial_train[f"density_count_{int(r_km)}km"] = np.array(
            [max(0, len(idx) - 1) for idx in tr_indices], dtype=np.float32
        )
        spatial_test[f"density_count_{int(r_km)}km"] = np.array(
            [len(idx) for idx in te_indices], dtype=np.float32
        )

    # Multi-scale distance-weighted spatial KNN target estimations
    spatial_knn_k_list = [3, 5, 10, 25]
    for k_val in spatial_knn_k_list:
        spatial_train[f"spatial_knn_target_mean_k{k_val}"] = 0.0
        spatial_train[f"spatial_knn_target_wmean_k{k_val}"] = 0.0
        spatial_test[f"spatial_knn_target_mean_k{k_val}"] = 0.0
        spatial_test[f"spatial_knn_target_wmean_k{k_val}"] = 0.0

    for fold in range(n_splits):
        tr_idx = np.where(folds != fold)[0]
        va_idx = np.where(folds == fold)[0]

        nn_fold = NearestNeighbors(metric="haversine", n_jobs=-1)
        nn_fold.fit(coords_rad_tr[tr_idx])

        for k_val in spatial_knn_k_list:
            k_eff = min(k_val, len(tr_idx))
            va_dists, va_knn_idx = nn_fold.kneighbors(coords_rad_tr[va_idx], n_neighbors=k_eff)
            va_dist_km = va_dists * earth_radius_km
            va_weights = 1.0 / (va_dist_km + 1.0)
            va_weights_norm = va_weights / np.sum(va_weights, axis=1, keepdims=True)

            spatial_train.iloc[
                va_idx,
                spatial_train.columns.get_loc(f"spatial_knn_target_mean_k{k_val}"),
            ] = np.mean(y_train[tr_idx][va_knn_idx], axis=1).astype(np.float32)
            spatial_train.iloc[
                va_idx,
                spatial_train.columns.get_loc(f"spatial_knn_target_wmean_k{k_val}"),
            ] = np.sum(y_train[tr_idx][va_knn_idx] * va_weights_norm, axis=1).astype(np.float32)

            te_dists, te_knn_idx = nn_fold.kneighbors(coords_rad_te, n_neighbors=k_eff)
            te_dist_km = te_dists * earth_radius_km
            te_weights = 1.0 / (te_dist_km + 1.0)
            te_weights_norm = te_weights / np.sum(te_weights, axis=1, keepdims=True)

            spatial_test[f"spatial_knn_target_mean_k{k_val}"] += (
                np.mean(y_train[tr_idx][te_knn_idx], axis=1).astype(np.float32)
                / n_splits
            )
            spatial_test[f"spatial_knn_target_wmean_k{k_val}"] += (
                np.sum(y_train[tr_idx][te_knn_idx] * te_weights_norm, axis=1).astype(np.float32)
                / n_splits
            )

    k_neighbors = [2, 5, 10]
    nn_model = NearestNeighbors(
        n_neighbors=max(k_neighbors) + 1, metric="euclidean", n_jobs=-1
    )
    nn_model.fit(coords_train)

    train_nn_dist, _ = nn_model.kneighbors(coords_train)
    test_nn_dist, _ = nn_model.kneighbors(coords_test)

    for k in k_neighbors:
        spatial_train[f"knn_dist_k{k}"] = train_nn_dist[:, k].astype(np.float32)
        spatial_test[f"knn_dist_k{k}"] = test_nn_dist[:, k - 1].astype(np.float32)
        spatial_train[f"knn_mean_dist_k{k}"] = np.mean(
            train_nn_dist[:, 1 : k + 1], axis=1
        ).astype(np.float32)
        spatial_test[f"knn_mean_dist_k{k}"] = np.mean(
            test_nn_dist[:, :k], axis=1
        ).astype(np.float32)

# ==============================================================================
# 6. Categorical, Geographical & OOF Target Statistics
# ==============================================================================
encoded_cat_train = pd.DataFrame(index=train_features_df.index)
encoded_cat_test = pd.DataFrame(index=test_features_df.index)

# Check for ZIP column to engineer ZIP-3 prefix
zip_cols = [c for c in all_cols if "ZIP" in c.upper() or "POSTAL" in c.upper()]
if zip_cols:
    zip_c = zip_cols[0]
    train_zip_pfx = (
        train_features_df[zip_c].astype(str).str.extract(r"^(\d{3})")[0].fillna("000")
    )
    test_zip_pfx = (
        test_features_df[zip_c].astype(str).str.extract(r"^(\d{3})")[0].fillna("000")
    )
    train_features_df["ZIP_PREFIX"] = train_zip_pfx
    test_features_df["ZIP_PREFIX"] = test_zip_pfx
    if "ZIP_PREFIX" not in cat_cols:
        cat_cols.append("ZIP_PREFIX")

for col in cat_cols:
    train_s = train_features_df[col].fillna("__MISSING__").astype(str)
    test_s = test_features_df[col].fillna("__MISSING__").astype(str)

    if (
        train_features_df[col].isna().sum() > 0
        or test_features_df[col].isna().sum() > 0
    ):
        encoded_cat_train[f"{col}_isna"] = (
            train_features_df[col].isna().astype(np.float32)
        )
        encoded_cat_test[f"{col}_isna"] = (
            test_features_df[col].isna().astype(np.float32)
        )

    freq_map = train_s.value_counts(normalize=True).to_dict()
    encoded_cat_train[f"{col}_freq"] = (
        train_s.map(freq_map).fillna(0.0).astype(np.float32)
    )
    encoded_cat_test[f"{col}_freq"] = (
        test_s.map(freq_map).fillna(0.0).astype(np.float32)
    )

    categories = sorted(train_s.unique())
    cat_map = {c: i for i, c in enumerate(categories)}
    encoded_cat_train[f"{col}_code"] = train_s.map(lambda x: cat_map.get(x, -1)).astype(
        np.int32
    )
    encoded_cat_test[f"{col}_code"] = test_s.map(lambda x: cat_map.get(x, -1)).astype(
        np.int32
    )

    # Out-of-fold smoothed target statistics (mean, std)
    smoothing_prior = 10.0
    global_mean = float(np.mean(y_train))
    encoded_cat_train[f"{col}_te_mean"] = 0.0
    encoded_cat_train[f"{col}_te_std"] = 0.0
    encoded_cat_test[f"{col}_te_mean"] = 0.0
    encoded_cat_test[f"{col}_te_std"] = 0.0

    for fold in range(n_splits):
        tr_idx = np.where(folds != fold)[0]
        va_idx = np.where(folds == fold)[0]

        fold_df = pd.DataFrame({"cat": train_s.iloc[tr_idx], "y": y_train[tr_idx]})
        stats = fold_df.groupby("cat")["y"].agg(["count", "mean", "std"])
        smoothed_mean = (stats["count"] * stats["mean"] + smoothing_prior * global_mean) / (
            stats["count"] + smoothing_prior
        )
        smoothed_std = stats["std"].fillna(0.0)

        val_cat = train_s.iloc[va_idx]
        encoded_cat_train.iloc[
            va_idx, encoded_cat_train.columns.get_loc(f"{col}_te_mean")
        ] = val_cat.map(smoothed_mean).fillna(global_mean).astype(np.float32)
        encoded_cat_train.iloc[
            va_idx, encoded_cat_train.columns.get_loc(f"{col}_te_std")
        ] = val_cat.map(smoothed_std).fillna(0.0).astype(np.float32)

        encoded_cat_test[f"{col}_te_mean"] += (
            test_s.map(smoothed_mean).fillna(global_mean).astype(np.float32) / n_splits
        )
        encoded_cat_test[f"{col}_te_std"] += (
            test_s.map(smoothed_std).fillna(0.0).astype(np.float32) / n_splits
        )

if len(cat_cols) >= 2:
    for i in range(min(3, len(cat_cols))):
        for j in range(i + 1, min(4, len(cat_cols))):
            col_a, col_b = cat_cols[i], cat_cols[j]
            comb_train = (
                train_features_df[col_a].astype(str)
                + "_"
                + train_features_df[col_b].astype(str)
            )
            comb_test = (
                test_features_df[col_a].astype(str)
                + "_"
                + test_features_df[col_b].astype(str)
            )

            comb_freq = comb_train.value_counts(normalize=True).to_dict()
            pair_name = f"inter_{col_a}_{col_b}_freq"
            encoded_cat_train[pair_name] = (
                comb_train.map(comb_freq).fillna(0.0).astype(np.float32)
            )
            encoded_cat_test[pair_name] = (
                comb_test.map(comb_freq).fillna(0.0).astype(np.float32)
            )

            pair_te_mean = f"inter_{col_a}_{col_b}_te_mean"
            pair_te_std = f"inter_{col_a}_{col_b}_te_std"
            encoded_cat_train[pair_te_mean] = 0.0
            encoded_cat_train[pair_te_std] = 0.0
            encoded_cat_test[pair_te_mean] = 0.0
            encoded_cat_test[pair_te_std] = 0.0

            smoothing_prior = 10.0
            for fold in range(n_splits):
                tr_idx = np.where(folds != fold)[0]
                va_idx = np.where(folds == fold)[0]

                fold_df = pd.DataFrame({"cat": comb_train.iloc[tr_idx], "y": y_train[tr_idx]})
                stats = fold_df.groupby("cat")["y"].agg(["count", "mean", "std"])
                smoothed_mean = (stats["count"] * stats["mean"] + smoothing_prior * global_mean) / (
                    stats["count"] + smoothing_prior
                )
                smoothed_std = stats["std"].fillna(0.0)

                val_cat = comb_train.iloc[va_idx]
                encoded_cat_train.iloc[
                    va_idx, encoded_cat_train.columns.get_loc(pair_te_mean)
                ] = val_cat.map(smoothed_mean).fillna(global_mean).astype(np.float32)
                encoded_cat_train.iloc[
                    va_idx, encoded_cat_train.columns.get_loc(pair_te_std)
                ] = val_cat.map(smoothed_std).fillna(0.0).astype(np.float32)

                encoded_cat_test[pair_te_mean] += (
                    comb_test.map(smoothed_mean).fillna(global_mean).astype(np.float32) / n_splits
                )
                encoded_cat_test[pair_te_std] += (
                    comb_test.map(smoothed_std).fillna(0.0).astype(np.float32) / n_splits
                )

# ==============================================================================
# 7. Numeric Features
# ==============================================================================
clean_num_train = pd.DataFrame(index=train_features_df.index)
clean_num_test = pd.DataFrame(index=test_features_df.index)

for col in num_cols:
    med_val = train_features_df[col].median()
    if pd.isna(med_val):
        med_val = 0.0

    if (
        train_features_df[col].isna().sum() > 0
        or test_features_df[col].isna().sum() > 0
    ):
        clean_num_train[f"{col}_isna"] = (
            train_features_df[col].isna().astype(np.float32)
        )
        clean_num_test[f"{col}_isna"] = test_features_df[col].isna().astype(np.float32)

    train_vals = train_features_df[col].fillna(med_val).astype(np.float32)
    test_vals = test_features_df[col].fillna(med_val).astype(np.float32)

    clean_num_train[col] = train_vals
    clean_num_test[col] = test_vals

    if (train_vals >= 0).all():
        clean_num_train[f"{col}_log1p"] = np.log1p(train_vals).astype(np.float32)
        clean_num_test[f"{col}_log1p"] = np.log1p(test_vals).astype(np.float32)

# ==============================================================================
# 8. Assemble Feature Matrices
# ==============================================================================
X_train_df = pd.concat(
    [spatial_train, train_text_feat, encoded_cat_train, clean_num_train], axis=1
)
X_test_df = pd.concat(
    [spatial_test, test_text_feat, encoded_cat_test, clean_num_test], axis=1
)

feature_columns = X_train_df.columns.tolist()
X_train_df = X_train_df[feature_columns].fillna(0.0).replace([np.inf, -np.inf], 0.0)
X_test_df = X_test_df[feature_columns].fillna(0.0).replace([np.inf, -np.inf], 0.0)


# ==============================================================================
# 8. Model Definitions
# ==============================================================================
class GatedResidualBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.15):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, out_features * 2)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.shortcut = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        h = self.norm(x)
        h = self.fc1(h)
        gate, value = torch.chunk(h, 2, dim=-1)
        h = self.act(gate) * value
        h = self.dropout(h)
        h = self.fc2(h)
        return residual + h


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        dropout_rate: float = 0.15,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        blocks = []
        for i in range(len(hidden_dims) - 1):
            blocks.append(
                GatedResidualBlock(
                    in_features=hidden_dims[i],
                    out_features=hidden_dims[i + 1],
                    dropout_rate=dropout_rate,
                )
            )
        self.res_blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dims[-1], output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = self.res_blocks(h)
        out = self.head(h)
        return out.squeeze(-1)


def get_pytorch_model_and_optimizer(
    input_dim: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dims: Tuple[int, ...] = (256, 128, 64),
    dropout_rate: float = 0.15,
):
    model = TabularResNet(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        output_dim=1,
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1e-6
    )
    return model, criterion, optimizer, scheduler


def get_lgb_model(custom_params: Optional[Dict[str, Any]] = None) -> lgb.LGBMRegressor:
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if custom_params:
        params.update(custom_params)
    return lgb.LGBMRegressor(**params)


def get_xgb_model(custom_params: Optional[Dict[str, Any]] = None) -> xgb.XGBRegressor:
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    if custom_params:
        params.update(custom_params)
    return xgb.XGBRegressor(**params)


def get_catboost_model(custom_params: Optional[Dict[str, Any]] = None):
    if not HAS_CATBOOST:
        return None
    params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 1500,
        "learning_rate": 0.04,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
    }
    if custom_params:
        params.update(custom_params)
    return cb.CatBoostRegressor(**params)


# ==============================================================================
# 9. Out-of-Fold Cross-Validation Training Loop
# ==============================================================================
oof_lgb = np.zeros(n_train, dtype=np.float32)
oof_lgb_log = np.zeros(n_train, dtype=np.float32)
oof_xgb = np.zeros(n_train, dtype=np.float32)
oof_xgb_log = np.zeros(n_train, dtype=np.float32)
oof_nn = np.zeros(n_train, dtype=np.float32)
if HAS_CATBOOST:
    oof_cb = np.zeros(n_train, dtype=np.float32)
    oof_cb_log = np.zeros(n_train, dtype=np.float32)

test_lgb = np.zeros(n_test, dtype=np.float32)
test_lgb_log = np.zeros(n_test, dtype=np.float32)
test_xgb = np.zeros(n_test, dtype=np.float32)
test_xgb_log = np.zeros(n_test, dtype=np.float32)
test_nn = np.zeros(n_test, dtype=np.float32)
if HAS_CATBOOST:
    test_cb = np.zeros(n_test, dtype=np.float32)
    test_cb_log = np.zeros(n_test, dtype=np.float32)

n_features = X_train_df.shape[1]

for fold in range(n_splits):
    train_idx = np.where(folds != fold)[0]
    val_idx = np.where(folds == fold)[0]

    X_tr_df, y_tr = X_train_df.iloc[train_idx], y_train[train_idx]
    X_val_df, y_val = X_train_df.iloc[val_idx], y_train[val_idx]

    y_tr_log = np.log1p(np.maximum(0, y_tr))
    y_val_log = np.log1p(np.maximum(0, y_val))

    # --- LightGBM (Raw RMSE) ---
    lgb_model = get_lgb_model({"random_state": 42 + fold})
    lgb_model.fit(
        X_tr_df,
        y_tr,
        eval_set=[(X_val_df, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_val_df)
    test_lgb += lgb_model.predict(X_test_df) / n_splits

    # --- LightGBM (Log1p Target) ---
    lgb_log_model = get_lgb_model({"random_state": 142 + fold})
    lgb_log_model.fit(
        X_tr_df,
        y_tr_log,
        eval_set=[(X_val_df, y_val_log)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    oof_lgb_log[val_idx] = np.expm1(np.maximum(0, lgb_log_model.predict(X_val_df)))
    test_lgb_log += np.expm1(np.maximum(0, lgb_log_model.predict(X_test_df))) / n_splits

    # --- XGBoost (Raw RMSE) ---
    xgb_model = get_xgb_model({"random_state": 42 + fold, "early_stopping_rounds": 50})
    xgb_model.fit(X_tr_df, y_tr, eval_set=[(X_val_df, y_val)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict(X_val_df)
    test_xgb += xgb_model.predict(X_test_df) / n_splits

    # --- XGBoost (Log1p Target) ---
    xgb_log_model = get_xgb_model({"random_state": 142 + fold, "early_stopping_rounds": 50})
    xgb_log_model.fit(X_tr_df, y_tr_log, eval_set=[(X_val_df, y_val_log)], verbose=False)
    oof_xgb_log[val_idx] = np.expm1(np.maximum(0, xgb_log_model.predict(X_val_df)))
    test_xgb_log += np.expm1(np.maximum(0, xgb_log_model.predict(X_test_df))) / n_splits

    # --- CatBoost ---
    if HAS_CATBOOST:
        cb_model = get_catboost_model({"random_seed": 42 + fold})
        cb_model.fit(
            X_tr_df,
            y_tr,
            eval_set=(X_val_df, y_val),
            early_stopping_rounds=50,
            verbose=False,
        )
        oof_cb[val_idx] = cb_model.predict(X_val_df)
        test_cb += cb_model.predict(X_test_df) / n_splits

        cb_log_model = get_catboost_model({"random_seed": 142 + fold})
        cb_log_model.fit(
            X_tr_df,
            y_tr_log,
            eval_set=(X_val_df, y_val_log),
            early_stopping_rounds=50,
            verbose=False,
        )
        oof_cb_log[val_idx] = np.expm1(np.maximum(0, cb_log_model.predict(X_val_df)))
        test_cb_log += np.expm1(np.maximum(0, cb_log_model.predict(X_test_df))) / n_splits

    # --- TabularResNet ---
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_df.values).astype(np.float32)
    X_val_scaled = scaler.transform(X_val_df.values).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_df.values).astype(np.float32)

    train_dataset = TensorDataset(torch.from_numpy(X_tr_scaled), torch.from_numpy(y_tr))
    val_dataset = TensorDataset(torch.from_numpy(X_val_scaled), torch.from_numpy(y_val))

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    nn_model, criterion, optimizer, scheduler = get_pytorch_model_and_optimizer(
        input_dim=n_features,
        lr=1e-3,
        weight_decay=1e-4,
        hidden_dims=(256, 128, 64),
        dropout_rate=0.15,
    )
    nn_model.to(device)

    best_val_rmse = float("inf")
    best_weights = None
    patience = 15
    patience_cnt = 0

    for epoch in range(60):
        nn_model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = nn_model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        val_preds_list = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                val_preds_list.append(nn_model(batch_x).cpu().numpy())
        val_preds_arr = np.concatenate(val_preds_list)
        epoch_rmse = np.sqrt(mean_squared_error(y_val, val_preds_arr))

        if epoch_rmse < best_val_rmse:
            best_val_rmse = epoch_rmse
            best_weights = copy.deepcopy(nn_model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    nn_model.load_state_dict(best_weights)
    nn_model.eval()

    with torch.no_grad():
        val_tensor = torch.from_numpy(X_val_scaled).to(device)
        oof_nn[val_idx] = nn_model(val_tensor).cpu().numpy()

        test_tensor = torch.from_numpy(X_test_scaled).to(device)
        test_nn += nn_model(test_tensor).cpu().numpy() / n_splits

# ==============================================================================
# 10. SLSQP Ensemble Weight Optimization & Final Predictions
# ==============================================================================
model_oof_list = [oof_lgb, oof_lgb_log, oof_xgb, oof_xgb_log, oof_nn]
model_test_list = [test_lgb, test_lgb_log, test_xgb, test_xgb_log, test_nn]
if HAS_CATBOOST:
    model_oof_list.extend([oof_cb, oof_cb_log])
    model_test_list.extend([test_cb, test_cb_log])

OOF_matrix = np.column_stack(model_oof_list)
TEST_matrix = np.column_stack(model_test_list)
n_models = OOF_matrix.shape[1]


def blend_rmse_loss(weights: np.ndarray) -> float:
    w = weights / np.sum(weights)
    pred = np.clip(np.dot(OOF_matrix, w), 0, None)
    return float(np.sqrt(mean_squared_error(y_train, pred)))


init_weights = np.ones(n_models) / n_models
bounds = [(0.0, 1.0) for _ in range(n_models)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    blend_rmse_loss,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
opt_weights = opt_res.x / np.sum(opt_res.x)

final_oof_predictions = np.clip(np.dot(OOF_matrix, opt_weights), 0, None)
final_test_predictions = np.clip(np.dot(TEST_matrix, opt_weights), 0, None)

final_validation_score = np.sqrt(mean_squared_error(y_train, final_oof_predictions))

# ==============================================================================
# 11. Output Submission & Report Final Score
# ==============================================================================
submission_df = pd.DataFrame(
    {"id": np.arange(n_test, dtype=int), "BEDS": final_test_predictions}
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_validation_score}")
