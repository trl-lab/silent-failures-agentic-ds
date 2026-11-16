import gc
import json
import os
import pickle
import re
import warnings

from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------
# Directories Setup
# -------------------------------------------------------------------------
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# 1. Load Datasets
# -------------------------------------------------------------------------
print("Loading raw parquet files...")
train_df = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
test_df = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

print(f"Raw Train Shape: {train_df.shape}")
print(f"Raw Test Shape:  {test_df.shape}")

# Ensure 0-based integer id for test
if "id" not in test_df.columns:
    test_df["id"] = np.arange(len(test_df), dtype=np.int64)

TARGET_COL = "VOLT_CLASS"
if TARGET_COL not in train_df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in training data.")

# Drop training rows with missing target
train_df = train_df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

# -------------------------------------------------------------------------
# 2. Target Encoding
# -------------------------------------------------------------------------
train_df[TARGET_COL] = train_df[TARGET_COL].astype(str).str.strip()
label_encoder = LabelEncoder()
y_train_full = label_encoder.fit_transform(train_df[TARGET_COL]).astype(np.int64)
num_classes = len(label_encoder.classes_)
print(f"Target '{TARGET_COL}' encoded: {num_classes} unique classes.")

with open(os.path.join(WORKING_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)


# -------------------------------------------------------------------------
# 3. Geometry & Spatial Feature Engineering
# -------------------------------------------------------------------------
def extract_coordinates(val):
    """Extract (x, y) coordinate sequences from WKT strings, lists, or shapely geometries."""
    if val is None:
        return []
    if hasattr(val, "coords"):
        return list(val.coords)
    if hasattr(val, "geoms"):
        coords = []
        for g in val.geoms:
            if hasattr(g, "coords"):
                coords.extend(list(g.coords))
        return coords
    if isinstance(val, (list, tuple, np.ndarray)):
        if len(val) > 0 and isinstance(val[0], (list, tuple, np.ndarray)):
            return [(float(p[0]), float(p[1])) for p in val if len(p) >= 2]
        elif len(val) >= 2:
            return [(float(val[0]), float(val[1]))]
    if isinstance(val, str):
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", val)
        if len(numbers) >= 2:
            return [
                (float(numbers[i]), float(numbers[i + 1]))
                for i in range(0, len(numbers) - 1, 2)
            ]
    return []


def compute_geometry_features(df):
    """Extract structural, spatial, and geometric features from dataset."""
    geom_cols = [
        col
        for col in df.columns
        if col.lower()
        in [
            "geometry",
            "geom",
            "shape",
            "wkt",
            "the_geom",
            "coordinates",
            "coords",
        ]
    ]

    features = pd.DataFrame(index=df.index)

    if geom_cols:
        primary_geom_col = geom_cols[0]
        parsed_coords = df[primary_geom_col].apply(extract_coordinates)

        n_pts = []
        start_x, start_y = [], []
        end_x, end_y = [], []
        min_x, max_x = [], []
        min_y, max_y = [], []
        mean_x, mean_y = [], []
        span_dist = []
        path_length = []
        bearing = []

        for coords in parsed_coords:
            if len(coords) == 0:
                n_pts.append(0)
                start_x.append(np.nan)
                start_y.append(np.nan)
                end_x.append(np.nan)
                end_y.append(np.nan)
                min_x.append(np.nan)
                max_x.append(np.nan)
                min_y.append(np.nan)
                max_y.append(np.nan)
                mean_x.append(np.nan)
                mean_y.append(np.nan)
                span_dist.append(np.nan)
                path_length.append(np.nan)
                bearing.append(np.nan)
            else:
                arr = np.array(coords, dtype=np.float64)
                xs = arr[:, 0]
                ys = arr[:, 1]

                n = len(coords)
                sx, sy = xs[0], ys[0]
                ex, ey = xs[-1], ys[-1]
                mix, max_ = xs.min(), xs.max()
                miy, may = ys.min(), ys.max()
                mx, my = xs.mean(), ys.mean()

                dx = ex - sx
                dy = ey - sy
                span = np.sqrt(dx**2 + dy**2)
                angle = np.arctan2(dy, dx) * 180.0 / np.pi

                if n > 1:
                    seg_dx = np.diff(xs)
                    seg_dy = np.diff(ys)
                    total_len = np.sum(np.sqrt(seg_dx**2 + seg_dy**2))
                else:
                    total_len = 0.0

                n_pts.append(n)
                start_x.append(sx)
                start_y.append(sy)
                end_x.append(ex)
                end_y.append(ey)
                min_x.append(mix)
                max_x.append(max_)
                min_y.append(miy)
                max_y.append(may)
                mean_x.append(mx)
                mean_y.append(my)
                span_dist.append(span)
                path_length.append(total_len)
                bearing.append(angle)

        features["geom_num_points"] = np.array(n_pts, dtype=np.float32)
        features["geom_start_x"] = np.array(start_x, dtype=np.float32)
        features["geom_start_y"] = np.array(start_y, dtype=np.float32)
        features["geom_end_x"] = np.array(end_x, dtype=np.float32)
        features["geom_end_y"] = np.array(end_y, dtype=np.float32)
        features["geom_min_x"] = np.array(min_x, dtype=np.float32)
        features["geom_max_x"] = np.array(max_x, dtype=np.float32)
        features["geom_min_y"] = np.array(min_y, dtype=np.float32)
        features["geom_max_y"] = np.array(max_y, dtype=np.float32)
        features["geom_centroid_x"] = np.array(mean_x, dtype=np.float32)
        features["geom_centroid_y"] = np.array(mean_y, dtype=np.float32)

        bbox_w = features["geom_max_x"] - features["geom_min_x"]
        bbox_h = features["geom_max_y"] - features["geom_min_y"]
        features["geom_bbox_width"] = bbox_w
        features["geom_bbox_height"] = bbox_h
        features["geom_bbox_area"] = bbox_w * bbox_h
        features["geom_bbox_aspect_ratio"] = bbox_w / (bbox_h + 1e-6)
        features["geom_bbox_diag"] = np.sqrt(bbox_w**2 + bbox_h**2)

        features["geom_span_distance"] = np.array(span_dist, dtype=np.float32)
        features["geom_path_length"] = np.array(path_length, dtype=np.float32)
        features["geom_bearing"] = np.array(bearing, dtype=np.float32)
        features["geom_sin_bearing"] = np.sin(np.radians(features["geom_bearing"].fillna(0.0))).astype(np.float32)
        features["geom_cos_bearing"] = np.cos(np.radians(features["geom_bearing"].fillna(0.0))).astype(np.float32)
        features["geom_tortuosity"] = features["geom_path_length"] / (
            features["geom_span_distance"] + 1e-6
        )

        # Multi-scale rotated coordinates
        cx = features["geom_centroid_x"]
        cy = features["geom_centroid_y"]
        for angle_deg in [30, 45, 60]:
            rad = np.radians(angle_deg)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            features[f"geom_rot_{angle_deg}_x"] = (cx * cos_a - cy * sin_a).astype(np.float32)
            features[f"geom_rot_{angle_deg}_y"] = (cx * sin_a + cy * cos_a).astype(np.float32)

        # Polar coordinates relative to dataset spatial median
        med_x = cx.median() if not cx.dropna().empty else 0.0
        med_y = cy.median() if not cy.dropna().empty else 0.0
        dx = cx - med_x
        dy = cy - med_y
        features["geom_polar_r"] = np.sqrt(dx**2 + dy**2).astype(np.float32)
        features["geom_polar_theta"] = np.arctan2(dy, dx).astype(np.float32)

    # Check for explicit coordinate columns
    lat_cols = [
        c
        for c in df.columns
        if c.lower() in ["lat", "latitude", "y", "start_lat", "min_y"]
    ]
    lon_cols = [
        c
        for c in df.columns
        if c.lower() in ["lon", "long", "longitude", "x", "start_lon", "min_x"]
    ]
    if lat_cols and lon_cols and ("geom_centroid_y" not in features or features.empty):
        features["geom_centroid_y"] = pd.to_numeric(
            df[lat_cols[0]], errors="coerce"
        ).astype(np.float32)
        features["geom_centroid_x"] = pd.to_numeric(
            df[lon_cols[0]], errors="coerce"
        ).astype(np.float32)

    return features


# -------------------------------------------------------------------------
# 4. Tabular & Infrastructure Feature Engineering
# -------------------------------------------------------------------------
def process_infrastructure_and_text(train_raw, test_raw):
    """Process domain categorical fields, substations, and temporal attributes."""
    train_feat = pd.DataFrame(index=train_raw.index)
    test_feat = pd.DataFrame(index=test_raw.index)

    sub1_col = next(
        (c for c in train_raw.columns if c.upper() in ["SUB_1", "SUBSTATION_1"]),
        None,
    )
    sub2_col = next(
        (c for c in train_raw.columns if c.upper() in ["SUB_2", "SUBSTATION_2"]),
        None,
    )

    if sub1_col and sub2_col:
        s1_train = train_raw[sub1_col].fillna("MISSING").astype(str).str.strip()
        s2_train = train_raw[sub2_col].fillna("MISSING").astype(str).str.strip()
        s1_test = test_raw[sub1_col].fillna("MISSING").astype(str).str.strip()
        s2_test = test_raw[sub2_col].fillna("MISSING").astype(str).str.strip()

        train_feat["sub_is_same"] = (s1_train == s2_train).astype(np.float32)
        test_feat["sub_is_same"] = (s1_test == s2_test).astype(np.float32)

        train_feat["sub1_len"] = s1_train.str.len().astype(np.float32)
        test_feat["sub1_len"] = s1_test.str.len().astype(np.float32)
        train_feat["sub2_len"] = s2_train.str.len().astype(np.float32)
        test_feat["sub2_len"] = s2_test.str.len().astype(np.float32)

        # Canonical symmetric pair (min, max) to capture undirected link identity
        canon_pair_train = np.where(s1_train <= s2_train, s1_train + "___" + s2_train, s2_train + "___" + s1_train)
        canon_pair_test = np.where(s1_test <= s2_test, s1_test + "___" + s2_test, s2_test + "___" + s1_test)
        train_raw["canon_sub_pair"] = canon_pair_train
        test_raw["canon_sub_pair"] = canon_pair_test

        directed_pair_train = s1_train + "___" + s2_train
        directed_pair_test = s1_test + "___" + s2_test

        # Build substation bipartite graph / degree centrality across all known lines
        all_s1 = pd.concat([s1_train, s1_test], ignore_index=True)
        all_s2 = pd.concat([s2_train, s2_test], ignore_index=True)
        all_subs = pd.concat([all_s1, all_s2], ignore_index=True)
        sub_degree_map = all_subs.value_counts().to_dict()

        train_deg1 = s1_train.map(sub_degree_map).fillna(0).astype(np.float32)
        train_deg2 = s2_train.map(sub_degree_map).fillna(0).astype(np.float32)
        test_deg1 = s1_test.map(sub_degree_map).fillna(0).astype(np.float32)
        test_deg2 = s2_test.map(sub_degree_map).fillna(0).astype(np.float32)

        train_feat["sub1_degree"] = train_deg1
        train_feat["sub2_degree"] = train_deg2
        test_feat["sub1_degree"] = test_deg1
        test_feat["sub2_degree"] = test_deg2

        train_feat["sub_deg_min"] = np.minimum(train_deg1, train_deg2)
        train_feat["sub_deg_max"] = np.maximum(train_deg1, train_deg2)
        train_feat["sub_deg_sum"] = train_deg1 + train_deg2
        train_feat["sub_deg_diff"] = np.abs(train_deg1 - train_deg2)
        train_feat["sub_deg_ratio"] = train_deg1 / (train_deg2 + 1.0)

        test_feat["sub_deg_min"] = np.minimum(test_deg1, test_deg2)
        test_feat["sub_deg_max"] = np.maximum(test_deg1, test_deg2)
        test_feat["sub_deg_sum"] = test_deg1 + test_deg2
        test_feat["sub_deg_diff"] = np.abs(test_deg1 - test_deg2)
        test_feat["sub_deg_ratio"] = test_deg1 / (test_deg2 + 1.0)

        for col_name, tr_series, te_series in [
            ("sub1_freq", s1_train, s1_test),
            ("sub2_freq", s2_train, s2_test),
            ("sub_pair_freq", pd.Series(directed_pair_train, index=train_raw.index), pd.Series(directed_pair_test, index=test_raw.index)),
            ("canon_sub_pair_freq", pd.Series(canon_pair_train, index=train_raw.index), pd.Series(canon_pair_test, index=test_raw.index)),
        ]:
            counts = tr_series.value_counts(normalize=True)
            train_feat[col_name] = tr_series.map(counts).fillna(0.0).astype(np.float32)
            test_feat[col_name] = te_series.map(counts).fillna(0.0).astype(np.float32)

    date_candidates = [
        c
        for c in train_raw.columns
        if any(
            k in c.upper()
            for k in ["DATE", "SOURCEDAT", "VAL_DATE", "YEAR", "CREATED", "MOD"]
        )
    ]
    for d_col in date_candidates:
        if d_col in test_raw.columns:
            tr_dt = pd.to_datetime(train_raw[d_col], errors="coerce")
            te_dt = pd.to_datetime(test_raw[d_col], errors="coerce")
            if tr_dt.notna().mean() > 0.1:
                train_feat[f"{d_col}_year"] = tr_dt.dt.year.astype(np.float32)
                test_feat[f"{d_col}_year"] = te_dt.dt.year.astype(np.float32)
                train_feat[f"{d_col}_month"] = tr_dt.dt.month.astype(np.float32)
                test_feat[f"{d_col}_month"] = te_dt.dt.month.astype(np.float32)
                train_feat[f"{d_col}_dayofweek"] = tr_dt.dt.dayofweek.astype(np.float32)
                test_feat[f"{d_col}_dayofweek"] = te_dt.dt.dayofweek.astype(np.float32)
                min_year = tr_dt.dt.year.dropna().min()
                if pd.notna(min_year):
                    train_feat[f"{d_col}_elapsed_years"] = (
                        tr_dt.dt.year - min_year
                    ).astype(np.float32)
                    test_feat[f"{d_col}_elapsed_years"] = (
                        te_dt.dt.year - min_year
                    ).astype(np.float32)

    # Text SVD extraction across text / categorical columns
    text_cols = [
        c
        for c in train_raw.columns
        if c in test_raw.columns
        and (
            train_raw[c].dtype == object
            or pd.api.types.is_string_dtype(train_raw[c])
        )
        and c.lower()
        not in [
            "geometry",
            "geom",
            "wkt",
            "the_geom",
            "shape",
            "volt_class",
            "target",
        ]
    ]
    if text_cols:
        combined_text_tr = (
            train_raw[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
        )
        combined_text_te = (
            test_raw[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
        )
        tfidf = TfidfVectorizer(
            max_features=2500, ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b"
        )
        tfidf_tr = tfidf.fit_transform(combined_text_tr)
        tfidf_te = tfidf.transform(combined_text_te)

        svd = TruncatedSVD(n_components=12, random_state=42)
        svd_tr = svd.fit_transform(tfidf_tr)
        svd_te = svd.transform(tfidf_te)

        for k in range(svd_tr.shape[1]):
            train_feat[f"text_svd_{k}"] = svd_tr[:, k].astype(np.float32)
            test_feat[f"text_svd_{k}"] = svd_te[:, k].astype(np.float32)

    return train_feat, test_feat


# -------------------------------------------------------------------------
# 5. Build Base Engineered Datasets
# -------------------------------------------------------------------------
print("Extracting geometric and infrastructure features...")
train_geom = compute_geometry_features(train_df)
test_geom = compute_geometry_features(test_df)
train_infra, test_infra = process_infrastructure_and_text(train_df, test_df)

X_train_base = pd.concat([train_geom, train_infra], axis=1)
X_test_base = pd.concat([test_geom, test_infra], axis=1)

ignored_cols = set(
    [
        TARGET_COL,
        "target_encoded",
        "id",
        "ID",
        "OBJECTID",
        "objectid",
        "index",
    ]
    + [
        c
        for c in train_df.columns
        if c.lower()
        in [
            "geometry",
            "geom",
            "shape",
            "wkt",
            "the_geom",
            "coordinates",
            "coords",
        ]
    ]
)

candidate_numeric_cols = [
    col
    for col in train_df.columns
    if col not in ignored_cols
    and pd.api.types.is_numeric_dtype(train_df[col])
    and col in test_df.columns
]

for col in candidate_numeric_cols:
    tr_col = pd.to_numeric(train_df[col], errors="coerce").astype(np.float32)
    te_col = pd.to_numeric(test_df[col], errors="coerce").astype(np.float32)

    X_train_base[f"num_{col}"] = tr_col
    X_test_base[f"num_{col}"] = te_col

    if (tr_col.dropna() >= 0).all() and (
        "length" in col.lower() or "miles" in col.lower() or "shape" in col.lower()
    ):
        X_train_base[f"num_{col}_log1p"] = np.log1p(np.maximum(0, tr_col))
        X_test_base[f"num_{col}_log1p"] = np.log1p(np.maximum(0, te_col))

categorical_cols = [
    col
    for col in train_df.columns
    if col not in ignored_cols
    and not pd.api.types.is_numeric_dtype(train_df[col])
    and col in test_df.columns
    and col not in ["geometry", "geom", "shape", "wkt"]
]

for col in categorical_cols:
    tr_cat = train_df[col].fillna("MISSING").astype(str).str.strip()
    te_cat = test_df[col].fillna("MISSING").astype(str).str.strip()

    freq_map = tr_cat.value_counts(normalize=True)
    X_train_base[f"cat_freq_{col}"] = (
        tr_cat.map(freq_map).fillna(0.0).astype(np.float32)
    )
    X_test_base[f"cat_freq_{col}"] = te_cat.map(freq_map).fillna(0.0).astype(np.float32)

    le = LabelEncoder()
    all_cats = list(set(tr_cat.unique()).union(set(te_cat.unique())))
    all_cats.append("<UNK>")
    le.fit(all_cats)

    X_train_base[f"cat_ord_{col}"] = le.transform(tr_cat).astype(np.float32)
    X_test_base[f"cat_ord_{col}"] = le.transform(te_cat).astype(np.float32)

# Multi-resolution Spatial KMeans Clustering (K=8 and K=24)
if (
    "geom_centroid_x" in X_train_base.columns
    and "geom_centroid_y" in X_train_base.columns
):
    median_x = X_train_base["geom_centroid_x"].median()
    median_y = X_train_base["geom_centroid_y"].median()

    coords_train = np.column_stack(
        [
            X_train_base["geom_centroid_x"].fillna(median_x).values,
            X_train_base["geom_centroid_y"].fillna(median_y).values,
        ]
    )
    coords_test = np.column_stack(
        [
            X_test_base["geom_centroid_x"].fillna(median_x).values,
            X_test_base["geom_centroid_y"].fillna(median_y).values,
        ]
    )

    for n_clusters in [8, 24]:
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42 + n_clusters, batch_size=512)
        kmeans.fit(coords_train)

        train_cluster_dists = kmeans.transform(coords_train)
        test_cluster_dists = kmeans.transform(coords_test)

        for k in range(n_clusters):
            X_train_base[f"spatial_k{n_clusters}_dist_{k}"] = train_cluster_dists[:, k].astype(np.float32)
            X_test_base[f"spatial_k{n_clusters}_dist_{k}"] = test_cluster_dists[:, k].astype(np.float32)

        X_train_base[f"spatial_k{n_clusters}_id"] = kmeans.predict(coords_train).astype(np.float32)
        X_test_base[f"spatial_k{n_clusters}_id"] = kmeans.predict(coords_test).astype(np.float32)

# Impute missing values
feature_cols = [c for c in X_train_base.columns]
for col in feature_cols:
    med = X_train_base[col].median()
    if pd.isna(med):
        med = 0.0

    if X_train_base[col].isna().sum() > 0 or X_test_base[col].isna().sum() > 0:
        X_train_base[f"is_na_{col}"] = X_train_base[col].isna().astype(np.float32)
        X_test_base[f"is_na_{col}"] = X_test_base[col].isna().astype(np.float32)

    X_train_base[col] = X_train_base[col].fillna(med).astype(np.float32)
    X_test_base[col] = X_test_base[col].fillna(med).astype(np.float32)

final_feature_cols = list(X_train_base.columns)
for df_obj in [X_train_base, X_test_base]:
    for col in final_feature_cols:
        df_obj[col] = np.nan_to_num(
            df_obj[col].values, nan=0.0, posinf=1e6, neginf=-1e6
        ).astype(np.float32)

print(f"Total features prepared: {len(final_feature_cols)}")

# -------------------------------------------------------------------------
# 6. Leak-Free Out-Of-Fold Multi-Class Encodings & Spatial KNN
# -------------------------------------------------------------------------
N_SPLITS = 5
RANDOM_STATE = 42

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
folds = np.zeros(len(train_df), dtype=np.int32)
for fold_idx, (_, val_idx) in enumerate(skf.split(X_train_base, y_train_full)):
    folds[val_idx] = fold_idx

X_train_mat = X_train_base[final_feature_cols].copy()
X_test_mat = X_test_base[final_feature_cols].copy()
test_ids = test_df["id"].values.astype(np.int64)

oof_probs_lgb = np.zeros((len(train_df), num_classes), dtype=np.float32)
oof_probs_xgb = np.zeros((len(train_df), num_classes), dtype=np.float32)
oof_probs_cat = np.zeros((len(train_df), num_classes), dtype=np.float32)

test_probs_lgb = np.zeros((len(test_df), num_classes), dtype=np.float32)
test_probs_xgb = np.zeros((len(test_df), num_classes), dtype=np.float32)
test_probs_cat = np.zeros((len(test_df), num_classes), dtype=np.float32)

te_candidate_cols = [
    c for c in ["SUB_1", "SUB_2", "canon_sub_pair", "OWNER", "TYPE", "STATUS"]
    if c in train_df.columns and c in test_df.columns
]

print(f"Training 3-Way Heterogeneous Ensemble across {N_SPLITS} folds...")

for fold_idx in range(N_SPLITS):
    idx_tr = np.where(folds != fold_idx)[0]
    idx_va = np.where(folds == fold_idx)[0]

    X_tr = X_train_mat.iloc[idx_tr].copy()
    y_tr = y_train_full[idx_tr]
    X_va = X_train_mat.iloc[idx_va].copy()
    y_va = y_train_full[idx_va]
    X_te = X_test_mat.copy()

    # --- Out-of-fold Bayesian Smoothed Multi-Class Target Encodings ---
    smoothing_m = 10.0
    priors = np.bincount(y_tr, minlength=num_classes) / len(y_tr)

    for te_col in te_candidate_cols:
        tr_s = train_df[te_col].iloc[idx_tr].fillna("MISSING").astype(str).str.strip()
        va_s = train_df[te_col].iloc[idx_va].fillna("MISSING").astype(str).str.strip()
        te_s = test_df[te_col].fillna("MISSING").astype(str).str.strip()

        for c_idx in range(num_classes):
            target_binary = (y_tr == c_idx).astype(np.float32)
            grp = pd.DataFrame({"cat": tr_s, "target": target_binary}).groupby("cat")["target"].agg(["sum", "count"])
            smoothed = (grp["sum"] + smoothing_m * priors[c_idx]) / (grp["count"] + smoothing_m)
            smooth_dict = smoothed.to_dict()

            feat_name = f"oof_te_{te_col}_c{c_idx}"
            X_tr[feat_name] = tr_s.map(smooth_dict).fillna(priors[c_idx]).astype(np.float32).values
            X_va[feat_name] = va_s.map(smooth_dict).fillna(priors[c_idx]).astype(np.float32).values
            X_te[feat_name] = te_s.map(smooth_dict).fillna(priors[c_idx]).astype(np.float32).values

    # --- Spatial K-Nearest Neighbor Target Features ---
    if "geom_centroid_x" in X_tr.columns and "geom_centroid_y" in X_tr.columns:
        coords_tr = X_tr[["geom_centroid_x", "geom_centroid_y"]].values
        coords_va = X_va[["geom_centroid_x", "geom_centroid_y"]].values
        coords_te = X_te[["geom_centroid_x", "geom_centroid_y"]].values

        nn = NearestNeighbors(n_neighbors=26, algorithm="ball_tree", metric="euclidean")
        nn.fit(coords_tr)

        dists_tr, idxs_tr = nn.kneighbors(coords_tr, n_neighbors=26)
        dists_va, idxs_va = nn.kneighbors(coords_va, n_neighbors=25)
        dists_te, idxs_te = nn.kneighbors(coords_te, n_neighbors=25)

        targets_tr_neighbors = y_tr[idxs_tr[:, 1:26]]
        dists_tr_neighbors = dists_tr[:, 1:26]

        targets_va_neighbors = y_tr[idxs_va[:, :25]]
        dists_va_neighbors = dists_va[:, :25]

        targets_te_neighbors = y_tr[idxs_te[:, :25]]
        dists_te_neighbors = dists_te[:, :25]

        for k_val in [5, 15, 25]:
            X_tr[f"knn_dist_mean_k{k_val}"] = np.mean(dists_tr_neighbors[:, :k_val], axis=1).astype(np.float32)
            X_va[f"knn_dist_mean_k{k_val}"] = np.mean(dists_va_neighbors[:, :k_val], axis=1).astype(np.float32)
            X_te[f"knn_dist_mean_k{k_val}"] = np.mean(dists_te_neighbors[:, :k_val], axis=1).astype(np.float32)

            for c_idx in range(num_classes):
                w_tr = 1.0 / (dists_tr_neighbors[:, :k_val] + 1e-4)
                match_tr = (targets_tr_neighbors[:, :k_val] == c_idx).astype(np.float32)
                weighted_prob_tr = np.sum(w_tr * match_tr, axis=1) / np.sum(w_tr, axis=1)

                w_va = 1.0 / (dists_va_neighbors[:, :k_val] + 1e-4)
                match_va = (targets_va_neighbors[:, :k_val] == c_idx).astype(np.float32)
                weighted_prob_va = np.sum(w_va * match_va, axis=1) / np.sum(w_va, axis=1)

                w_te = 1.0 / (dists_te_neighbors[:, :k_val] + 1e-4)
                match_te = (targets_te_neighbors[:, :k_val] == c_idx).astype(np.float32)
                weighted_prob_te = np.sum(w_te * match_te, axis=1) / np.sum(w_te, axis=1)

                X_tr[f"knn_weighted_prob_k{k_val}_c{c_idx}"] = weighted_prob_tr.astype(np.float32)
                X_va[f"knn_weighted_prob_k{k_val}_c{c_idx}"] = weighted_prob_va.astype(np.float32)
                X_te[f"knn_weighted_prob_k{k_val}_c{c_idx}"] = weighted_prob_te.astype(np.float32)

    # --- 1. LightGBM Classifier ---
    lgb_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=20,
        random_state=RANDOM_STATE + fold_idx,
        n_jobs=-1,
        verbosity=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="multi_logloss",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    va_preds_lgb = lgb_model.predict_proba(X_va)
    oof_probs_lgb[idx_va] = va_preds_lgb
    test_probs_lgb += lgb_model.predict_proba(X_te) / N_SPLITS

    # --- 2. XGBoost Classifier ---
    xgb_model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=3,
        random_state=RANDOM_STATE + fold_idx,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )
    xgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )
    va_preds_xgb = xgb_model.predict_proba(X_va)
    oof_probs_xgb[idx_va] = va_preds_xgb
    test_probs_xgb += xgb_model.predict_proba(X_te) / N_SPLITS

    # --- 3. CatBoost Classifier ---
    cat_model = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="MultiClass",
        random_seed=RANDOM_STATE + fold_idx,
        verbose=False,
        early_stopping_rounds=50,
    )
    cat_model.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        verbose=False,
    )
    va_preds_cat = cat_model.predict_proba(X_va)
    oof_probs_cat[idx_va] = va_preds_cat
    test_probs_cat += cat_model.predict_proba(X_te) / N_SPLITS

    fold_blend = (va_preds_lgb + va_preds_xgb + va_preds_cat) / 3.0
    fold_acc = accuracy_score(y_va, np.argmax(fold_blend, axis=1))
    print(f"Fold {fold_idx + 1}/{N_SPLITS} Validation Accuracy: {fold_acc:.5f}")

# -------------------------------------------------------------------------
# 7. Optimal Multi-Model Blending & Out-of-Fold Evaluation
# -------------------------------------------------------------------------
def loss_func(weights):
    w = np.maximum(weights, 0.0)
    w_sum = np.sum(w)
    if w_sum <= 1e-6:
        return 1e6
    w = w / w_sum
    blend = w[0] * oof_probs_lgb + w[1] * oof_probs_xgb + w[2] * oof_probs_cat
    preds = np.argmax(blend, axis=1)
    return 1.0 - accuracy_score(y_train_full, preds)

init_weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
opt_res = minimize(
    loss_func,
    init_weights,
    method="Nelder-Mead",
    options={"maxiter": 500, "disp": False},
)
best_weights = np.maximum(opt_res.x, 0.0)
best_weights /= np.sum(best_weights)
print(f"Optimized Ensemble Weights (LGB, XGB, CAT): {best_weights.round(4)}")

final_oof_probs = (
    best_weights[0] * oof_probs_lgb
    + best_weights[1] * oof_probs_xgb
    + best_weights[2] * oof_probs_cat
)
oof_predictions = np.argmax(final_oof_probs, axis=1)
overall_acc = accuracy_score(y_train_full, oof_predictions)

# -------------------------------------------------------------------------
# 8. Test Prediction & Submission Generation
# -------------------------------------------------------------------------
final_test_probs = (
    best_weights[0] * test_probs_lgb
    + best_weights[1] * test_probs_xgb
    + best_weights[2] * test_probs_cat
)
test_pred_idx = np.argmax(final_test_probs, axis=1)
test_pred_classes = label_encoder.inverse_transform(test_pred_idx)

submission_df = pd.DataFrame(
    {"id": test_ids.astype(np.int64), TARGET_COL: test_pred_classes}
)

submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df.to_csv(submission_path, index=False)

print(
    f"Submission saved successfully to {submission_path} ({len(submission_df)} rows)."
)
print(f"Final Validation Score: {overall_acc}")
