import gc
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import haversine_distances
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int = 42):
    """Set random seeds across all libraries for deterministic execution."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


# ==========================================
# 1. Feature Engineering & Preprocessing
# ==========================================
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to uppercase stripped strings."""
    df = df.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df


def find_spatial_columns(df: pd.DataFrame):
    """Detect latitude and longitude column names flexibly."""
    lat_col, lon_col = None, None
    for c in df.columns:
        c_lower = c.lower()
        if c_lower in ["latitude", "lat", "y", "lat_dd", "geom_y"]:
            lat_col = c
        elif c_lower in ["longitude", "lon", "long", "x", "lon_dd", "geom_x"]:
            lon_col = c
    return lat_col, lon_col


def sanitize_sentinels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Detect and map standard HIFLD sentinel values (-999, -9999, 999999) to NaN and return sentinel indicator flags."""
    df_clean = df.copy()
    sentinel_flags = pd.DataFrame(index=df.index)
    sentinel_num_values = {-999, -9999, 999999, -999.0, -9999.0, 999999.0}
    sentinel_str_values = {
        "-999",
        "-9999",
        "999999",
        "NOT AVAILABLE",
        "UNAVAILABLE",
        "NONE",
        "NULL",
        "UNKNOWN",
        "NOT REPORTED",
    }

    for c in df_clean.columns:
        if c.upper() == "CAPACITY":
            continue
        if np.issubdtype(df_clean[c].dtype, np.number):
            is_sentinel = df_clean[c].isin(sentinel_num_values)
            if is_sentinel.any():
                sentinel_flags[f"SENTINEL_FLAG_{c}"] = is_sentinel.astype(np.float32)
                df_clean.loc[is_sentinel, c] = np.nan
        else:
            str_s = df_clean[c].astype(str).str.strip().str.upper()
            is_sentinel = str_s.isin(sentinel_str_values)
            if is_sentinel.any():
                sentinel_flags[f"SENTINEL_FLAG_{c}"] = is_sentinel.astype(np.float32)
                df_clean.loc[is_sentinel, c] = np.nan
    return df_clean, sentinel_flags


def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from date/time columns if available."""
    feats = pd.DataFrame(index=df.index)
    date_cols = [
        c
        for c in df.columns
        if any(k in c.upper() for k in ["DATE", "TIME", "YEAR", "UPDATED", "RECORD"])
        and not c.startswith("SENTINEL_FLAG_")
    ]
    for col in date_cols:
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().sum() > 0:
            feats[f"DATE_{col}_year"] = parsed.dt.year.astype(np.float32)
            feats[f"DATE_{col}_month"] = parsed.dt.month.astype(np.float32)
            feats[f"DATE_{col}_day"] = parsed.dt.day.astype(np.float32)
            feats[f"DATE_{col}_dayofweek"] = parsed.dt.dayofweek.astype(np.float32)
            feats[f"DATE_{col}_elapsed_days"] = (
                parsed - pd.Timestamp("2000-01-01")
            ).dt.days.astype(np.float32)
    return feats


def extract_text_keyword_features(
    series: pd.Series, prefix: str = "TXT"
) -> pd.DataFrame:
    """Extract domain-specific structural indicators from facility names and descriptions."""
    text = series.fillna("").astype(str).str.lower()
    keywords = [
        "federal",
        "state",
        "county",
        "detention",
        "correctional",
        "institution",
        "jail",
        "penitentiary",
        "juvenile",
        "camp",
        "center",
        "annex",
        "complex",
        "maximum",
        "medium",
        "minimum",
        "close",
        "medical",
        "psychiatric",
        "women",
        "community",
        "work",
        "youth",
        "treatment",
        "regional",
        "holding",
        "intake",
    ]
    feats = pd.DataFrame(index=series.index)
    feats[f"{prefix}_char_len"] = text.str.len()
    feats[f"{prefix}_word_count"] = text.str.split().apply(len)
    feats[f"{prefix}_num_digits"] = text.apply(lambda s: sum(c.isdigit() for c in s))

    for kw in keywords:
        feats[f"{prefix}_has_{kw}"] = text.str.contains(
            r"\b" + re.escape(kw) + r"\b", regex=True
        ).astype(np.float32)

    return feats


def compute_geospatial_features(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    state_centroids: Optional[Dict[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """Generate spherical, polar, distance-to-centers, state centroids, and spatial binning features."""
    feats = pd.DataFrame(index=df.index)

    lat = pd.to_numeric(df[lat_col], errors="coerce").fillna(
        df[lat_col].median() if lat_col in df else 0.0
    )
    lon = pd.to_numeric(df[lon_col], errors="coerce").fillna(
        df[lon_col].median() if lon_col in df else 0.0
    )

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # 3D Cartesian coordinates on unit sphere
    feats["GEO_X"] = np.cos(lat_rad) * np.cos(lon_rad)
    feats["GEO_Y"] = np.cos(lat_rad) * np.sin(lon_rad)
    feats["GEO_Z"] = np.sin(lat_rad)

    # Distance to US Geographic Center (Lebanon, Kansas: 39.8283 N, -98.5795 W)
    us_center_lat, us_center_lon = np.radians(39.8283), np.radians(-98.5795)
    dlat = lat_rad - us_center_lat
    dlon = lon_rad - us_center_lon
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(us_center_lat) * np.cos(lat_rad) * np.sin(dlon / 2.0) ** 2
    )
    feats["GEO_dist_us_center_km"] = 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    # Distances to major reference points
    reference_points = {
        "NYC": (40.7128, -74.0060),
        "LA": (34.0522, -118.2437),
        "CHI": (41.8781, -87.6298),
        "HOU": (29.7604, -95.3698),
        "ATL": (33.7490, -84.3880),
    }
    for name, (ref_lat, ref_lon) in reference_points.items():
        ref_lat_rad, ref_lon_rad = np.radians(ref_lat), np.radians(ref_lon)
        d_lat = lat_rad - ref_lat_rad
        d_lon = lon_rad - ref_lon_rad
        hav = (
            np.sin(d_lat / 2.0) ** 2
            + np.cos(ref_lat_rad) * np.cos(lat_rad) * np.sin(d_lon / 2.0) ** 2
        )
        feats[f"GEO_dist_{name}_km"] = (
            6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))
        )

    # State centroid distance
    if state_centroids is not None and "STATE" in df.columns:
        state_series = df["STATE"].astype(str)
        dist_state_list = []
        for s_val, l_rad, ln_rad in zip(state_series, lat_rad, lon_rad):
            if s_val in state_centroids:
                sc_lat, sc_lon = state_centroids[s_val]
                sc_lat_rad, sc_lon_rad = np.radians(sc_lat), np.radians(sc_lon)
                d_lat_sc = l_rad - sc_lat_rad
                d_lon_sc = ln_rad - sc_lon_rad
                hav_sc = (
                    np.sin(d_lat_sc / 2.0) ** 2
                    + np.cos(sc_lat_rad) * np.cos(l_rad) * np.sin(d_lon_sc / 2.0) ** 2
                )
                dist_state_list.append(6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(hav_sc, 0, 1))))
            else:
                dist_state_list.append(0.0)
        feats["GEO_dist_state_centroid_km"] = np.array(dist_state_list, dtype=np.float32)

    # Spatial Grids
    feats["GEO_lat_bin_05"] = (lat / 0.5).round()
    feats["GEO_lon_bin_05"] = (lon / 0.5).round()
    feats["GEO_lat_bin_10"] = (lat / 1.0).round()
    feats["GEO_lon_bin_10"] = (lon / 1.0).round()
    feats["GEO_lat_bin_20"] = (lat / 2.0).round()
    feats["GEO_lon_bin_20"] = (lon / 2.0).round()

    return feats


def smooth_target_encode(
    train_series: pd.Series,
    target: pd.Series,
    test_series: pd.Series,
    weight: float = 10.0,
):
    """Compute smoothed target encoding strictly avoiding leakage."""
    global_mean = target.mean()
    stats = target.groupby(train_series).agg(["count", "mean"])
    counts = stats["count"]
    means = stats["mean"]
    smoothed = (counts * means + weight * global_mean) / (counts + weight)
    train_encoded = train_series.map(smoothed).fillna(global_mean).astype(np.float32)
    test_encoded = test_series.map(smoothed).fillna(global_mean).astype(np.float32)
    return train_encoded, test_encoded


# ==========================================
# 2. PyTorch Tabular Residual Architecture
# ==========================================
class GatedResidualBlock(nn.Module):
    """Gated Residual Block with LayerNorm, SwiGLU / GLU gating, and Dropout."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.15):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.fc = nn.Linear(in_features, out_features * 2)
        self.drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(out_features, out_features)

        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        norm_x = self.norm(x)
        gate_out = self.fc(norm_x)
        h, gate = gate_out.chunk(2, dim=-1)
        h = h * F.silu(gate)
        h = self.drop(h)
        h = self.out_proj(h)
        return h + residual


class GatedTabularNet(nn.Module):
    """Gated Residual Tabular Network for deep continuous and relational feature learning."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])

        blocks = []
        for i in range(len(hidden_dims) - 1):
            blocks.append(
                GatedResidualBlock(
                    in_features=hidden_dims[i],
                    out_features=hidden_dims[i + 1],
                    dropout=dropout,
                )
            )
        self.blocks = nn.Sequential(*blocks)

        self.head_norm = nn.LayerNorm(hidden_dims[-1])
        self.head_fc = nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2)
        self.head_drop = nn.Dropout(dropout / 2.0)
        self.out_proj = nn.Linear(hidden_dims[-1] // 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(self.input_norm(x))
        h = self.blocks(h)
        h = self.head_drop(F.gelu(self.head_fc(self.head_norm(h))))
        return self.out_proj(h)


def build_optimizer_and_scheduler(
    model: nn.Module,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 100,
    min_lr: float = 1e-6,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Instantiate AdamW optimizer with Cosine Annealing learning rate schedule."""
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    return optimizer, scheduler


# ==========================================
# 3. GBDT Model Configurations
# ==========================================
def get_lgb_default_params(random_seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_catboost_default_params(random_seed: int = 42) -> Dict[str, Any]:
    return {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 2000,
        "learning_rate": 0.04,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": random_seed,
        "verbose": False,
        "thread_count": -1,
    }


def get_xgb_default_params(random_seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_seed,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    }


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error faithfully."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


# ==========================================
# 4. End-to-End Pipeline Execution
# ==========================================
def main():
    seed_everything(42)
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # 1. Load Data
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    train_df = clean_column_names(train_df)
    test_df = clean_column_names(test_df)

    target_col = "CAPACITY"
    assert (
        target_col in train_df.columns
    ), f"Target '{target_col}' not found in train.parquet"

    y = train_df[target_col].copy()

    # Active domain sentinel sanitization
    train_df, train_sentinel_flags = sanitize_sentinels(train_df)
    test_df, test_sentinel_flags = sanitize_sentinels(test_df)

    train_features = pd.DataFrame(index=train_df.index)
    test_features = pd.DataFrame(index=test_df.index)

    if not train_sentinel_flags.empty:
        train_features = pd.concat([train_features, train_sentinel_flags], axis=1)
    if not test_sentinel_flags.empty:
        test_features = pd.concat([test_features, test_sentinel_flags], axis=1)

    for col in train_sentinel_flags.columns:
        if col not in test_features.columns:
            test_features[col] = 0.0
    for col in test_sentinel_flags.columns:
        if col not in train_features.columns:
            train_features[col] = 0.0

    # 2. Geospatial Feature Engineering & Multi-Resolution Spatial Clustering
    lat_col, lon_col = find_spatial_columns(train_df)
    train_coords = None
    test_coords = None
    if lat_col and lon_col:
        geo_train = compute_geospatial_features(train_df, lat_col, lon_col)
        geo_test = compute_geospatial_features(test_df, lat_col, lon_col)

        train_features = pd.concat([train_features, geo_train], axis=1)
        test_features = pd.concat([test_features, geo_test], axis=1)

        lat_med = train_df[lat_col].median() if lat_col in train_df else 0.0
        lon_med = train_df[lon_col].median() if lon_col in train_df else 0.0
        tr_raw_coords = train_df[[lat_col, lon_col]].fillna({lat_col: lat_med, lon_col: lon_med}).values
        te_raw_coords = test_df[[lat_col, lon_col]].fillna({lat_col: lat_med, lon_col: lon_med}).values
        train_coords = np.radians(tr_raw_coords)
        test_coords = np.radians(te_raw_coords)

        # Multi-resolution KMeans Spatial Clusters
        combined_coords = np.vstack([tr_raw_coords, te_raw_coords])
        for n_clusters in [8, 20, 40]:
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
            km.fit(combined_coords)
            tr_dist = km.transform(tr_raw_coords)
            te_dist = km.transform(te_raw_coords)

            train_features[f"GEO_cluster_min_dist_{n_clusters}"] = tr_dist.min(axis=1).astype(np.float32)
            test_features[f"GEO_cluster_min_dist_{n_clusters}"] = te_dist.min(axis=1).astype(np.float32)
            train_features[f"GEO_cluster_label_{n_clusters}"] = tr_dist.argmin(axis=1).astype(np.int32)
            test_features[f"GEO_cluster_label_{n_clusters}"] = te_dist.argmin(axis=1).astype(np.int32)

        for k in [3, 7, 15]:
            knn = NearestNeighbors(n_neighbors=k + 1, metric="haversine")
            knn.fit(train_coords)

            dists_tr, _ = knn.kneighbors(train_coords)
            train_features[f"GEO_knn_mean_dist_k{k}"] = (
                dists_tr[:, 1:].mean(axis=1) * 6371.0
            )

            dists_te, _ = knn.kneighbors(test_coords)
            test_features[f"GEO_knn_mean_dist_k{k}"] = (
                dists_te[:, :k].mean(axis=1) * 6371.0
            )

    # 3. Multi-Field Text and Structural Embeddings
    text_fields = [c for c in ["NAME", "ADDRESS", "CITY", "COUNTY", "NAICS_DESC"] if c in train_df.columns]
    if "NAME" in train_df.columns:
        train_kw = extract_text_keyword_features(train_df["NAME"], prefix="NAME")
        test_kw = extract_text_keyword_features(test_df["NAME"], prefix="NAME")
        train_features = pd.concat([train_features, train_kw], axis=1)
        test_features = pd.concat([test_features, test_kw], axis=1)

    if len(text_fields) > 0:
        combined_text_train = train_df[text_fields].fillna("").astype(str).agg(" ".join, axis=1)
        combined_text_test = (
            test_df[[c for c in text_fields if c in test_df.columns]]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            if any(c in test_df.columns for c in text_fields)
            else pd.Series([""] * len(test_df))
        )

        tfidf = TfidfVectorizer(
            max_features=2000, stop_words="english", ngram_range=(1, 2)
        )
        svd = TruncatedSVD(n_components=16, random_state=42)
        train_tfidf = tfidf.fit_transform(combined_text_train)
        test_tfidf = tfidf.transform(combined_text_test)

        train_svd = svd.fit_transform(train_tfidf)
        test_svd = svd.transform(test_tfidf)

        for i in range(16):
            train_features[f"TEXT_svd_{i}"] = train_svd[:, i].astype(np.float32)
            test_features[f"TEXT_svd_{i}"] = test_svd[:, i].astype(np.float32)

    # 4. Categorical Encodings (Frequency & Interactions)
    cat_candidates = [
        c
        for c in train_df.columns
        if c not in [target_col, lat_col, lon_col]
        and (
            train_df[c].dtype == "object"
            or train_df[c].dtype.name == "category"
            or train_df[c].nunique() < 200
        )
    ]

    for cat in cat_candidates:
        if cat in train_df.columns and cat in test_df.columns:
            s_train = train_df[cat].astype(str).fillna("__MISSING__")
            s_test = test_df[cat].astype(str).fillna("__MISSING__")

            freq = s_train.value_counts(normalize=True).to_dict()
            train_features[f"FREQ_{cat}"] = (
                s_train.map(freq).fillna(0.0).astype(np.float32)
            )
            test_features[f"FREQ_{cat}"] = (
                s_test.map(freq).fillna(0.0).astype(np.float32)
            )

            categories = {val: idx for idx, val in enumerate(s_train.unique())}
            train_features[f"CAT_{cat}"] = (
                s_train.map(categories).fillna(-1).astype(np.int32)
            )
            test_features[f"CAT_{cat}"] = (
                s_test.map(categories).fillna(-1).astype(np.int32)
            )

    if "STATE" in train_df.columns and "TYPE" in train_df.columns:
        inter_train = train_df["STATE"].astype(str) + "_" + train_df["TYPE"].astype(str)
        inter_test = test_df["STATE"].astype(str) + "_" + test_df["TYPE"].astype(str)
        freq_inter = inter_train.value_counts(normalize=True).to_dict()
        train_features["FREQ_STATE_TYPE"] = (
            inter_train.map(freq_inter).fillna(0.0).astype(np.float32)
        )
        test_features["FREQ_STATE_TYPE"] = (
            inter_test.map(freq_inter).fillna(0.0).astype(np.float32)
        )

    if "STATE" in train_df.columns and "SECLEVEL" in train_df.columns:
        inter_train2 = (
            train_df["STATE"].astype(str) + "_" + train_df["SECLEVEL"].astype(str)
        )
        inter_test2 = (
            test_df["STATE"].astype(str) + "_" + test_df["SECLEVEL"].astype(str)
        )
        freq_inter2 = inter_train2.value_counts(normalize=True).to_dict()
        train_features["FREQ_STATE_SECLEVEL"] = (
            inter_train2.map(freq_inter2).fillna(0.0).astype(np.float32)
        )
        test_features["FREQ_STATE_SECLEVEL"] = (
            inter_test2.map(freq_inter2).fillna(0.0).astype(np.float32)
        )

    # 5. Numerical Features, Missingness Indicators, and Group Statistical Aggregations
    num_cols = [
        c
        for c in train_df.columns
        if c not in [target_col] and np.issubdtype(train_df[c].dtype, np.number)
    ]
    for num in num_cols:
        train_features[f"NUM_{num}"] = pd.to_numeric(
            train_df[num], errors="coerce"
        ).astype(np.float32)
        test_features[f"NUM_{num}"] = (
            pd.to_numeric(test_df[num], errors="coerce").astype(np.float32)
            if num in test_df.columns
            else np.nan
        )

        train_features[f"MISSING_{num}"] = train_df[num].isna().astype(np.float32)
        test_features[f"MISSING_{num}"] = (
            test_df[num].isna().astype(np.float32) if num in test_df.columns else 1.0
        )

    # Categorical Group Statistical Aggregations & Disparities
    grp_keys = [c for c in ["STATE", "TYPE", "SECLEVEL"] if c in train_df.columns]
    num_agg_candidates = [c for c in ["LATITUDE", "LONGITUDE", "POPULATION", "ZIP"] if c in train_df.columns and np.issubdtype(train_df[c].dtype, np.number)]
    for gkey in grp_keys:
        for ncol in num_agg_candidates:
            grp_stats = train_df.groupby(gkey)[ncol].agg(["mean", "std"]).reset_index()
            grp_stats.columns = [gkey, f"GRP_{gkey}_{ncol}_mean", f"GRP_{gkey}_{ncol}_std"]
            train_m = train_df[[gkey]].merge(grp_stats, on=gkey, how="left")
            train_features[f"GRP_{gkey}_{ncol}_mean"] = train_m[f"GRP_{gkey}_{ncol}_mean"].values.astype(np.float32)
            train_features[f"GRP_{gkey}_{ncol}_std"] = train_m[f"GRP_{gkey}_{ncol}_std"].fillna(0).values.astype(np.float32)

            if gkey in test_df.columns:
                test_m = test_df[[gkey]].merge(grp_stats, on=gkey, how="left")
                test_features[f"GRP_{gkey}_{ncol}_mean"] = test_m[f"GRP_{gkey}_{ncol}_mean"].values.astype(np.float32)
                test_features[f"GRP_{gkey}_{ncol}_std"] = test_m[f"GRP_{gkey}_{ncol}_std"].fillna(0).values.astype(np.float32)
            else:
                test_features[f"GRP_{gkey}_{ncol}_mean"] = np.nan
                test_features[f"GRP_{gkey}_{ncol}_std"] = np.nan

            if f"NUM_{ncol}" in train_features.columns:
                diff_tr = train_features[f"NUM_{ncol}"] - train_features[f"GRP_{gkey}_{ncol}_mean"]
                ratio_tr = train_features[f"NUM_{ncol}"] / (train_features[f"GRP_{gkey}_{ncol}_mean"].abs() + 1.0)
                train_features[f"DISP_diff_{ncol}_{gkey}"] = diff_tr.astype(np.float32)
                train_features[f"DISP_ratio_{ncol}_{gkey}"] = ratio_tr.astype(np.float32)

                diff_te = test_features[f"NUM_{ncol}"] - test_features[f"GRP_{gkey}_{ncol}_mean"]
                ratio_te = test_features[f"NUM_{ncol}"] / (test_features[f"GRP_{gkey}_{ncol}_mean"].abs() + 1.0)
                test_features[f"DISP_diff_{ncol}_{gkey}"] = diff_te.astype(np.float32)
                test_features[f"DISP_ratio_{ncol}_{gkey}"] = ratio_te.astype(np.float32)

    # 6. Stratified 5-Fold Splitting and Out-Of-Fold Target Encoding
    n_splits = 5
    binned_y = pd.qcut(
        y, q=min(10, max(2, y.nunique())), labels=False, duplicates="drop"
    )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_assignments = np.zeros(len(train_df), dtype=int)
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_features, binned_y)):
        fold_assignments[val_idx] = fold

    train_features["FOLD"] = fold_assignments

    te_cols = [
        c
        for c in ["STATE", "COUNTY", "TYPE", "STATUS", "SECLEVEL", "OPERATOR"]
        if c in train_df.columns
    ]
    for te_col in te_cols:
        oof_te = np.zeros(len(train_df), dtype=np.float32)
        test_te_folds = np.zeros(len(test_df), dtype=np.float32)

        for fold in range(n_splits):
            tr_idx = np.where(fold_assignments != fold)[0]
            val_idx = np.where(fold_assignments == fold)[0]

            tr_s = train_df.iloc[tr_idx][te_col].astype(str)
            val_s = train_df.iloc[val_idx][te_col].astype(str)
            te_s = (
                test_df[te_col].astype(str)
                if te_col in test_df.columns
                else pd.Series([""] * len(test_df))
            )

            tr_enc, val_enc = smooth_target_encode(
                tr_s, y.iloc[tr_idx], val_s, weight=15.0
            )
            _, test_enc_fold = smooth_target_encode(
                tr_s, y.iloc[tr_idx], te_s, weight=15.0
            )

            oof_te[val_idx] = val_enc
            test_te_folds += test_enc_fold / n_splits

        train_features[f"TE_{te_col}"] = oof_te
        test_features[f"TE_{te_col}"] = test_te_folds

    # 6b. Out-of-Fold Distance-Weighted Spatial Nearest Neighbor Target Aggregations
    if train_coords is not None and test_coords is not None:
        y_raw = y.values.astype(np.float32)
        n_tr_samples = len(train_df)
        n_te_samples = len(test_df)
        for k_val in [3, 7, 15]:
            oof_knn_mean = np.zeros(n_tr_samples, dtype=np.float32)
            oof_knn_std = np.zeros(n_tr_samples, dtype=np.float32)
            oof_knn_min = np.zeros(n_tr_samples, dtype=np.float32)
            oof_knn_max = np.zeros(n_tr_samples, dtype=np.float32)

            test_knn_mean_folds = np.zeros(n_te_samples, dtype=np.float32)
            test_knn_std_folds = np.zeros(n_te_samples, dtype=np.float32)
            test_knn_min_folds = np.zeros(n_te_samples, dtype=np.float32)
            test_knn_max_folds = np.zeros(n_te_samples, dtype=np.float32)

            for fold in range(n_splits):
                tr_idx = np.where(fold_assignments != fold)[0]
                val_idx = np.where(fold_assignments == fold)[0]

                tr_c_fold = train_coords[tr_idx]
                val_c_fold = train_coords[val_idx]
                y_tr_fold = y_raw[tr_idx]

                knn_spatial = NearestNeighbors(n_neighbors=min(k_val, len(tr_idx)), metric="haversine")
                knn_spatial.fit(tr_c_fold)

                dists_val, idx_val = knn_spatial.kneighbors(val_c_fold)
                val_targets = y_tr_fold[idx_val]
                weights_val = 1.0 / (dists_val * 6371.0 + 1.0)
                weights_val /= np.maximum(weights_val.sum(axis=1, keepdims=True), 1e-6)

                oof_knn_mean[val_idx] = np.sum(val_targets * weights_val, axis=1)
                oof_knn_std[val_idx] = np.std(val_targets, axis=1)
                oof_knn_min[val_idx] = np.min(val_targets, axis=1)
                oof_knn_max[val_idx] = np.max(val_targets, axis=1)

                dists_te, idx_te = knn_spatial.kneighbors(test_coords)
                te_targets = y_tr_fold[idx_te]
                weights_te = 1.0 / (dists_te * 6371.0 + 1.0)
                weights_te /= np.maximum(weights_te.sum(axis=1, keepdims=True), 1e-6)

                test_knn_mean_folds += np.sum(te_targets * weights_te, axis=1) / n_splits
                test_knn_std_folds += np.std(te_targets, axis=1) / n_splits
                test_knn_min_folds += np.min(te_targets, axis=1) / n_splits
                test_knn_max_folds += np.max(te_targets, axis=1) / n_splits

            train_features[f"SPATIAL_knn_target_mean_k{k_val}"] = oof_knn_mean
            train_features[f"SPATIAL_knn_target_std_k{k_val}"] = oof_knn_std
            train_features[f"SPATIAL_knn_target_min_k{k_val}"] = oof_knn_min
            train_features[f"SPATIAL_knn_target_max_k{k_val}"] = oof_knn_max

            test_features[f"SPATIAL_knn_target_mean_k{k_val}"] = test_knn_mean_folds
            test_features[f"SPATIAL_knn_target_std_k{k_val}"] = test_knn_std_folds
            test_features[f"SPATIAL_knn_target_min_k{k_val}"] = test_knn_min_folds
            test_features[f"SPATIAL_knn_target_max_k{k_val}"] = test_knn_max_folds

        # 6c. Multi-Scale Exponential Kernel Spatial Neighbor Target Aggregations
        bandwidths = [25.0, 75.0, 250.0]
        for bw in bandwidths:
            oof_exp_mean = np.zeros(n_tr_samples, dtype=np.float32)
            oof_exp_std = np.zeros(n_tr_samples, dtype=np.float32)
            test_exp_mean_folds = np.zeros(n_te_samples, dtype=np.float32)
            test_exp_std_folds = np.zeros(n_te_samples, dtype=np.float32)

            for fold in range(n_splits):
                tr_idx = np.where(fold_assignments != fold)[0]
                val_idx = np.where(fold_assignments == fold)[0]

                tr_c_fold = train_coords[tr_idx]
                val_c_fold = train_coords[val_idx]
                y_tr_fold = y_raw[tr_idx]

                dist_val_km = haversine_distances(val_c_fold, tr_c_fold) * 6371.0
                weights_val = np.exp(-dist_val_km / bw)
                sum_w_val = np.maximum(weights_val.sum(axis=1, keepdims=True), 1e-6)
                norm_w_val = weights_val / sum_w_val

                val_exp_m = np.sum(norm_w_val * y_tr_fold, axis=1)
                val_exp_s = np.sqrt(np.sum(norm_w_val * ((y_tr_fold - val_exp_m[:, None]) ** 2), axis=1))

                oof_exp_mean[val_idx] = val_exp_m
                oof_exp_std[val_idx] = val_exp_s

                dist_te_km = haversine_distances(test_coords, tr_c_fold) * 6371.0
                weights_te = np.exp(-dist_te_km / bw)
                sum_w_te = np.maximum(weights_te.sum(axis=1, keepdims=True), 1e-6)
                norm_w_te = weights_te / sum_w_te

                te_exp_m = np.sum(norm_w_te * y_tr_fold, axis=1)
                te_exp_s = np.sqrt(np.sum(norm_w_te * ((y_tr_fold - te_exp_m[:, None]) ** 2), axis=1))

                test_exp_mean_folds += te_exp_m / n_splits
                test_exp_std_folds += te_exp_s / n_splits

            train_features[f"SPATIAL_exp_kernel_mean_bw{int(bw)}"] = oof_exp_mean
            train_features[f"SPATIAL_exp_kernel_std_bw{int(bw)}"] = oof_exp_std
            test_features[f"SPATIAL_exp_kernel_mean_bw{int(bw)}"] = test_exp_mean_folds
            test_features[f"SPATIAL_exp_kernel_std_bw{int(bw)}"] = test_exp_std_folds

    # 6d. Out-of-Fold TF-IDF Ridge Regression on Combined Text
    if len(text_fields) > 0:
        oof_text_ridge = np.zeros(len(train_df), dtype=np.float32)
        test_text_ridge = np.zeros(len(test_df), dtype=np.float32)

        for fold in range(n_splits):
            tr_idx = np.where(fold_assignments != fold)[0]
            val_idx = np.where(fold_assignments == fold)[0]

            ridge_tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
            X_text_tr = ridge_tfidf.fit_transform(combined_text_train.iloc[tr_idx])
            X_text_val = ridge_tfidf.transform(combined_text_train.iloc[val_idx])
            X_text_te = ridge_tfidf.transform(combined_text_test)

            ridge_model = Ridge(alpha=2.0, random_state=42)
            ridge_model.fit(X_text_tr, y.iloc[tr_idx].values)

            oof_text_ridge[val_idx] = ridge_model.predict(X_text_val).astype(np.float32)
            test_text_ridge += ridge_model.predict(X_text_te).astype(np.float32) / n_splits

        train_features["TEXT_ridge_oof_pred"] = oof_text_ridge
        test_features["TEXT_ridge_oof_pred"] = test_text_ridge

    # 7. Robust Imputation
    feature_cols = [c for c in train_features.columns if c != "FOLD"]
    col_medians = train_features[feature_cols].median()
    train_features[feature_cols] = train_features[feature_cols].fillna(col_medians)
    test_features[feature_cols] = test_features[feature_cols].fillna(col_medians)

    # 8. Model Training & Validation Setup
    X = train_features[feature_cols].copy()
    y_vals = y.values.astype(np.float32)
    folds = train_features["FOLD"].values
    X_test = test_features[feature_cols].copy()

    n_samples = len(train_df)
    n_test = len(test_df)

    oof_lgb = np.zeros(n_samples, dtype=np.float32)
    oof_xgb = np.zeros(n_samples, dtype=np.float32)
    oof_cat = np.zeros(n_samples, dtype=np.float32)
    oof_et = np.zeros(n_samples, dtype=np.float32)
    oof_nn = np.zeros(n_samples, dtype=np.float32)

    test_preds_lgb = np.zeros(n_test, dtype=np.float32)
    test_preds_xgb = np.zeros(n_test, dtype=np.float32)
    test_preds_cat = np.zeros(n_test, dtype=np.float32)
    test_preds_et = np.zeros(n_test, dtype=np.float32)
    test_preds_nn = np.zeros(n_test, dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 9. Cross-Validation Training Loop
    for fold in range(n_splits):
        tr_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        X_tr, y_tr = X.iloc[tr_idx].values, y_vals[tr_idx]
        X_val, y_val = X.iloc[val_idx].values, y_vals[val_idx]
        X_te = X_test.values

        # --- A. LightGBM ---
        lgb_params = get_lgb_default_params(random_seed=42 + fold)
        lgb_train_data = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
        lgb_val_data = lgb.Dataset(
            X_val, label=y_val, reference=lgb_train_data, free_raw_data=False
        )

        lgb_model = lgb.train(
            lgb_params,
            lgb_train_data,
            num_boost_round=2000,
            valid_sets=[lgb_val_data],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        val_pred_lgb = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
        test_pred_lgb = lgb_model.predict(X_te, num_iteration=lgb_model.best_iteration)

        oof_lgb[val_idx] = val_pred_lgb
        test_preds_lgb += test_pred_lgb / n_splits

        # --- B. XGBoost ---
        xgb_params = get_xgb_default_params(random_seed=42 + fold)
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        val_pred_xgb = xgb_model.predict(X_val)
        test_pred_xgb = xgb_model.predict(X_te)

        oof_xgb[val_idx] = val_pred_xgb
        test_preds_xgb += test_pred_xgb / n_splits

        # --- C. CatBoost ---
        cat_params = get_catboost_default_params(random_seed=42 + fold)
        cat_model = CatBoostRegressor(**cat_params, early_stopping_rounds=50)
        cat_model.fit(
            X_tr, y_tr, eval_set=(X_val, y_val), verbose=False, use_best_model=True
        )

        val_pred_cat = cat_model.predict(X_val)
        test_pred_cat = cat_model.predict(X_te)

        oof_cat[val_idx] = val_pred_cat
        test_preds_cat += test_pred_cat / n_splits

        # --- D. ExtraTrees Regressor ---
        et_model = ExtraTreesRegressor(
            n_estimators=150,
            max_depth=14,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42 + fold,
            n_jobs=-1,
        )
        et_model.fit(np.nan_to_num(X_tr, nan=0.0), y_tr)
        val_pred_et = et_model.predict(np.nan_to_num(X_val, nan=0.0))
        test_pred_et = et_model.predict(np.nan_to_num(X_te, nan=0.0))

        oof_et[val_idx] = val_pred_et
        test_preds_et += test_pred_et / n_splits

        # --- E. PyTorch GatedTabularNet ---
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(np.nan_to_num(X_tr, nan=0.0))
        X_val_sc = scaler.transform(np.nan_to_num(X_val, nan=0.0))
        X_te_sc = scaler.transform(np.nan_to_num(X_te, nan=0.0))

        y_mean, y_std = y_tr.mean(), y_tr.std() + 1e-6
        y_tr_norm = (y_tr - y_mean) / y_std
        y_val_norm = (y_val - y_mean) / y_std

        train_ds = TensorDataset(
            torch.tensor(X_tr_sc, dtype=torch.float32),
            torch.tensor(y_tr_norm, dtype=torch.float32).unsqueeze(1),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val_sc, dtype=torch.float32),
            torch.tensor(y_val_norm, dtype=torch.float32).unsqueeze(1),
        )

        train_loader = DataLoader(
            train_ds, batch_size=128, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

        nn_model = GatedTabularNet(
            input_dim=X_tr.shape[1], hidden_dims=[256, 128, 64], dropout=0.2
        ).to(device)
        nn_optimizer, nn_scheduler = build_optimizer_and_scheduler(
            nn_model, learning_rate=2e-3, weight_decay=1e-4, epochs=60
        )
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_nn_weights = None
        patience = 12
        patience_counter = 0

        for epoch in range(60):
            nn_model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                nn_optimizer.zero_grad()
                pred = nn_model(bx)
                loss = criterion(pred, by)
                loss.backward()
                nn_optimizer.step()
            nn_scheduler.step()

            nn_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    val_loss += criterion(nn_model(bx), by).item() * len(bx)
            val_loss /= len(val_ds)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_nn_weights = {
                    k: v.cpu().clone() for k, v in nn_model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_nn_weights is not None:
            nn_model.load_state_dict(best_nn_weights)
        nn_model.eval()

        with torch.no_grad():
            val_t = torch.tensor(X_val_sc, dtype=torch.float32).to(device)
            val_pred_nn = (
                (nn_model(val_t).cpu().numpy().ravel() * y_std) + y_mean
            ).astype(np.float32)

            test_t = torch.tensor(X_te_sc, dtype=torch.float32).to(device)
            test_pred_nn = (
                (nn_model(test_t).cpu().numpy().ravel() * y_std) + y_mean
            ).astype(np.float32)

        oof_nn[val_idx] = val_pred_nn
        test_preds_nn += test_pred_nn / n_splits

        gc.collect()

    # 10. Optimize Ensemble Blend Weights via SLSQP
    oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_et, oof_nn])
    test_matrix = np.column_stack(
        [test_preds_lgb, test_preds_xgb, test_preds_cat, test_preds_et, test_preds_nn]
    )

    def blend_objective(weights):
        w = np.array(weights)
        pred = np.dot(oof_matrix, w)
        pred = np.clip(pred, 0, None)
        rmse = compute_rmse(y_vals, pred)
        l2_penalty = 1e-3 * np.sum(w ** 2)
        return rmse + l2_penalty

    init_weights = np.ones(5) / 5.0
    bounds = [(0.0, 1.0)] * 5
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    opt_res = minimize(
        blend_objective,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-7},
    )
    best_weights = np.clip(opt_res.x, 0, None)
    best_weights = best_weights / np.sum(best_weights)

    # 11. Final Validation Score and Submission Generation
    final_oof_preds = np.clip(np.dot(oof_matrix, best_weights), 0, None)
    final_test_preds = np.clip(np.dot(test_matrix, best_weights), 0, None)

    final_score = compute_rmse(y_vals, final_oof_preds)

    submission_path = "./submission/submission.csv"
    sub_df = pd.DataFrame(
        {"id": np.arange(len(final_test_preds)), "CAPACITY": final_test_preds}
    )
    sub_df.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()
