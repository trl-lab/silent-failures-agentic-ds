import json
import os
import warnings
from typing import Any, Dict, List, Optional

from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# Suppress non-critical warnings
warnings.filterwarnings("ignore")


# =========================================================================
# 1. Data Loading and Feature Engineering
# =========================================================================


def load_raw_data(data_dir: str = "./input"):
    """Load train and test parquet datasets."""
    train_path = os.path.join(data_dir, "train.parquet")
    test_path = os.path.join(data_dir, "test.parquet")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    return train_df, test_df


def identify_columns(df: pd.DataFrame, target_col: str = "HPSA Score"):
    """Identify column modalities: numeric, categorical, datetime, text, and target."""
    cols = [c for c in df.columns if c != target_col]

    datetime_cols = []
    categorical_cols = []
    numerical_cols = []
    text_cols = []

    for c in cols:
        if (
            pd.api.types.is_datetime64_any_dtype(df[c])
            or "date" in c.lower()
            or "time" in c.lower()
        ):
            try:
                sample_non_null = df[c].dropna().iloc[:10]
                if len(sample_non_null) > 0:
                    pd.to_datetime(sample_non_null)
                    datetime_cols.append(c)
                    continue
            except Exception:
                pass

        if pd.api.types.is_numeric_dtype(df[c]):
            numerical_cols.append(c)
        else:
            unique_count = df[c].nunique(dropna=True)
            if unique_count > 10:
                text_cols.append(c)
            categorical_cols.append(c)

    return {
        "datetime": datetime_cols,
        "categorical": categorical_cols,
        "numerical": numerical_cols,
        "text": text_cols,
        "target": target_col,
    }


def extract_datetime_features(df: pd.DataFrame, dt_cols: List[str]):
    """Extract temporal signals from datetime columns."""
    features = pd.DataFrame(index=df.index)
    parsed_dts = {}

    for col in dt_cols:
        try:
            dt_series = pd.to_datetime(df[col], errors="coerce")
            parsed_dts[col] = dt_series
            prefix = f"dt_{col}"
            features[f"{prefix}_year"] = dt_series.dt.year.fillna(-1).astype(np.int32)
            features[f"{prefix}_month"] = dt_series.dt.month.fillna(-1).astype(np.int8)
            features[f"{prefix}_day"] = dt_series.dt.day.fillna(-1).astype(np.int8)
            features[f"{prefix}_dayofweek"] = dt_series.dt.dayofweek.fillna(-1).astype(
                np.int8
            )
            features[f"{prefix}_quarter"] = dt_series.dt.quarter.fillna(-1).astype(
                np.int8
            )

            epoch = pd.Timestamp("1970-01-01")
            features[f"{prefix}_days_since_epoch"] = (
                dt_series - epoch
            ).dt.total_seconds() / 86400.0
            features[f"{prefix}_is_null"] = dt_series.isnull().astype(np.int8)
        except Exception:
            continue

    dt_names = list(parsed_dts.keys())
    for i in range(len(dt_names)):
        for j in range(i + 1, len(dt_names)):
            d1, d2 = dt_names[i], dt_names[j]
            diff_days = (parsed_dts[d1] - parsed_dts[d2]).dt.total_seconds() / 86400.0
            features[f"diff_days_{d1}_minus_{d2}"] = diff_days.fillna(0.0).astype(
                np.float32
            )

    return features


def extract_spatial_features(df: pd.DataFrame, num_cols: List[str], train_mask=None):
    """Extract coordinate transformations and spatial clustering if lat/lon exist."""
    features = pd.DataFrame(index=df.index)

    lat_col = next(
        (c for c in num_cols if c.lower() in ["latitude", "lat", "hpsa_latitude", "y"]),
        None,
    )
    lon_col = next(
        (
            c
            for c in num_cols
            if c.lower() in ["longitude", "lon", "long", "hpsa_longitude", "x"]
        ),
        None,
    )

    if lat_col and lon_col:
        lat = df[lat_col].fillna(
            df[lat_col].median()
            if train_mask is None
            else df.loc[train_mask, lat_col].median()
        )
        lon = df[lon_col].fillna(
            df[lon_col].median()
            if train_mask is None
            else df.loc[train_mask, lon_col].median()
        )

        center_lat, center_lon = 39.8283, -98.5795
        d_lat = np.radians(lat - center_lat)
        d_lon = np.radians(lon - center_lon)

        a = (
            np.sin(d_lat / 2.0) ** 2
            + np.cos(np.radians(center_lat))
            * np.cos(np.radians(lat))
            * np.sin(d_lon / 2.0) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        features["spatial_dist_to_center_km"] = (6371.0 * c).astype(np.float32)
        features["spatial_bearing_rad"] = np.arctan2(d_lon, d_lat).astype(np.float32)

        coords = np.column_stack([lat, lon])
        train_coords = coords if train_mask is None else coords[train_mask]

        for k in [10, 25]:
            kmeans = MiniBatchKMeans(
                n_clusters=k, random_state=42, batch_size=256, n_init=3
            )
            kmeans.fit(train_coords)
            features[f"spatial_cluster_{k}"] = kmeans.predict(coords).astype(np.int32)
            features[f"spatial_dist_cluster_{k}"] = np.min(
                kmeans.transform(coords), axis=1
            ).astype(np.float32)

    return features


def extract_domain_ratios(df: pd.DataFrame, num_cols: List[str]):
    """Compute statutory and domain-specific ratios and threshold indicators for HPSA estimation."""
    features = pd.DataFrame(index=df.index)
    col_map = {c.lower(): c for c in num_cols}

    pop_col = next(
        (col_map[k] for k in col_map if "population" in k or "pop" in k), None
    )
    fte_col = next((col_map[k] for k in col_map if "fte" in k or "provider" in k), None)
    shortage_col = next((col_map[k] for k in col_map if "shortage" in k), None)
    poverty_col = next(
        (col_map[k] for k in col_map if "poverty" in k or "pov" in k), None
    )
    uninsured_col = next((col_map[k] for k in col_map if "uninsured" in k), None)

    eps = 1e-4

    if pop_col and fte_col:
        pop_fte_val = df[pop_col] / (df[fte_col] + eps)
        features["ratio_pop_to_fte"] = pop_fte_val.astype(np.float32)
        features["ratio_fte_to_pop_10k"] = (
            (df[fte_col] * 10000.0) / (df[pop_col] + eps)
        ).astype(np.float32)

        # Statutory piecewise PPR brackets (<1500 to >=5000)
        features["statutory_ppr_lt_1500"] = (pop_fte_val < 1500.0).astype(np.float32)
        features["statutory_ppr_1500_2000"] = (
            (pop_fte_val >= 1500.0) & (pop_fte_val < 2000.0)
        ).astype(np.float32)
        features["statutory_ppr_2000_2500"] = (
            (pop_fte_val >= 2000.0) & (pop_fte_val < 2500.0)
        ).astype(np.float32)
        features["statutory_ppr_2500_3000"] = (
            (pop_fte_val >= 2500.0) & (pop_fte_val < 3000.0)
        ).astype(np.float32)
        features["statutory_ppr_3000_3500"] = (
            (pop_fte_val >= 3000.0) & (pop_fte_val < 3500.0)
        ).astype(np.float32)
        features["statutory_ppr_3500_4000"] = (
            (pop_fte_val >= 3500.0) & (pop_fte_val < 4000.0)
        ).astype(np.float32)
        features["statutory_ppr_4000_5000"] = (
            (pop_fte_val >= 4000.0) & (pop_fte_val < 5000.0)
        ).astype(np.float32)
        features["statutory_ppr_gte_5000"] = (pop_fte_val >= 5000.0).astype(np.float32)

        # Statutory PPR discrete scoring mapping (0 to 7)
        ppr_pts = np.zeros(len(df), dtype=np.float32)
        ppr_pts[pop_fte_val >= 1500.0] = 1.0
        ppr_pts[pop_fte_val >= 2000.0] = 2.0
        ppr_pts[pop_fte_val >= 2500.0] = 3.0
        ppr_pts[pop_fte_val >= 3000.0] = 4.0
        ppr_pts[pop_fte_val >= 3500.0] = 5.0
        ppr_pts[pop_fte_val >= 4000.0] = 6.0
        ppr_pts[pop_fte_val >= 5000.0] = 7.0
        features["statutory_ppr_points"] = ppr_pts

    if shortage_col and pop_col:
        features["ratio_shortage_to_pop_10k"] = (
            (df[shortage_col] * 10000.0) / (df[pop_col] + eps)
        ).astype(np.float32)

    if shortage_col and fte_col:
        features["ratio_shortage_to_fte"] = (
            df[shortage_col] / (df[fte_col] + eps)
        ).astype(np.float32)
        features["total_required_fte"] = (df[shortage_col] + df[fte_col]).astype(
            np.float32
        )

    if poverty_col and pop_col:
        features["estimated_poverty_population"] = (
            df[pop_col] * (df[poverty_col] / 100.0)
        ).astype(np.float32)

    if poverty_col:
        pov_val = df[poverty_col].fillna(0.0)
        # Statutory piecewise poverty brackets (<15% to >=40%)
        features["statutory_pov_lt_15"] = (pov_val < 15.0).astype(np.float32)
        features["statutory_pov_15_20"] = (
            (pov_val >= 15.0) & (pov_val < 20.0)
        ).astype(np.float32)
        features["statutory_pov_20_25"] = (
            (pov_val >= 20.0) & (pov_val < 25.0)
        ).astype(np.float32)
        features["statutory_pov_25_30"] = (
            (pov_val >= 25.0) & (pov_val < 30.0)
        ).astype(np.float32)
        features["statutory_pov_30_40"] = (
            (pov_val >= 30.0) & (pov_val < 40.0)
        ).astype(np.float32)
        features["statutory_pov_gte_40"] = (pov_val >= 40.0).astype(np.float32)

        # Statutory poverty discrete scoring mapping (0 to 5)
        pov_pts = np.zeros(len(df), dtype=np.float32)
        pov_pts[pov_val >= 15.0] = 1.0
        pov_pts[pov_val >= 20.0] = 2.0
        pov_pts[pov_val >= 25.0] = 3.0
        pov_pts[pov_val >= 30.0] = 4.0
        pov_pts[pov_val >= 40.0] = 5.0
        features["statutory_poverty_points"] = pov_pts

    if "statutory_ppr_points" in features.columns and "statutory_poverty_points" in features.columns:
        features["statutory_combined_estimated_score"] = (
            features["statutory_ppr_points"] + features["statutory_poverty_points"]
        ).astype(np.float32)

    if poverty_col and uninsured_col:
        features["combined_vulnerability_index"] = (
            df[poverty_col] * df[uninsured_col]
        ).astype(np.float32)
        features["ratio_poverty_to_uninsured"] = (
            df[poverty_col] / (df[uninsured_col] + eps)
        ).astype(np.float32)

    # Skewed demographic and provider log1p counts
    for c in [pop_col, fte_col, shortage_col]:
        if c is not None and c in df.columns:
            features[f"log1p_{c}"] = np.log1p(
                np.maximum(0.0, df[c].fillna(0.0))
            ).astype(np.float32)

    numeric_present = [
        c
        for c in [pop_col, fte_col, shortage_col, poverty_col, uninsured_col]
        if c is not None
    ]
    if len(numeric_present) < 2 and len(num_cols) >= 2:
        top_nums = num_cols[:4]
        for i in range(len(top_nums)):
            for j in range(i + 1, len(top_nums)):
                c1, c2 = top_nums[i], top_nums[j]
                features[f"prod_{c1}_{c2}"] = (
                    df[c1].fillna(0) * df[c2].fillna(0)
                ).astype(np.float32)
                features[f"ratio_{c1}_{c2}"] = (
                    df[c1].fillna(0) / (df[c2].fillna(0).abs() + eps)
                ).astype(np.float32)

    return features


def extract_text_svd_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_cols: List[str],
    n_components: int = 6,
):
    """Extract TF-IDF + SVD semantics from descriptive text/name columns fitted strictly on train."""
    train_svd_feats = pd.DataFrame(index=train_df.index)
    test_svd_feats = pd.DataFrame(index=test_df.index)

    for col in text_cols:
        if col not in train_df.columns:
            continue
        try:
            train_text = train_df[col].fillna("").astype(str)
            test_text = test_df[col].fillna("").astype(str)

            tfidf = TfidfVectorizer(
                max_features=2500,
                ngram_range=(1, 2),
                min_df=2,
                token_pattern=r"(?u)\b\w+\b",
            )
            tr_tfidf = tfidf.fit_transform(train_text)
            te_tfidf = tfidf.transform(test_text)

            actual_dim = min(n_components, tr_tfidf.shape[1] - 1)
            if actual_dim >= 2:
                svd = TruncatedSVD(n_components=actual_dim, random_state=42)
                tr_svd = svd.fit_transform(tr_tfidf)
                te_svd = svd.transform(te_tfidf)

                for k in range(actual_dim):
                    train_svd_feats[f"svd_{col}_{k}"] = tr_svd[:, k].astype(np.float32)
                    test_svd_feats[f"svd_{col}_{k}"] = te_svd[:, k].astype(np.float32)
        except Exception:
            continue

    return train_svd_feats, test_svd_feats


def compute_group_aggregations(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_cols: List[str],
    target_num_cols: List[str],
):
    """Compute mean and std statistics for numerical features grouped by categories strictly fitted on train."""
    train_agg_feats = pd.DataFrame(index=train_df.index)
    test_agg_feats = pd.DataFrame(index=test_df.index)

    for g_col in group_cols:
        for num_col in target_num_cols:
            if (
                g_col == num_col
                or num_col not in train_df.columns
                or g_col not in train_df.columns
            ):
                continue

            agg_stats = (
                train_df.groupby(g_col)[num_col]
                .agg(["mean", "std", "median"])
                .reset_index()
            )
            agg_stats.columns = [
                g_col,
                f"grp_{g_col}_{num_col}_mean",
                f"grp_{g_col}_{num_col}_std",
                f"grp_{g_col}_{num_col}_median",
            ]

            eps = 1e-4
            m_train = pd.merge(train_df[[g_col]], agg_stats, on=g_col, how="left")
            tr_mean = m_train[f"grp_{g_col}_{num_col}_mean"].values
            tr_std = m_train[f"grp_{g_col}_{num_col}_std"].fillna(0.0).values
            train_agg_feats[f"grp_{g_col}_{num_col}_mean"] = tr_mean
            train_agg_feats[f"grp_{g_col}_{num_col}_std"] = tr_std
            train_agg_feats[f"grp_{g_col}_{num_col}_diff_mean"] = (
                train_df[num_col].values - tr_mean
            )
            train_agg_feats[f"grp_{g_col}_{num_col}_zscore"] = (
                (train_df[num_col].values - tr_mean) / (tr_std + eps)
            ).astype(np.float32)

            m_test = pd.merge(test_df[[g_col]], agg_stats, on=g_col, how="left")
            te_mean = m_test[f"grp_{g_col}_{num_col}_mean"].values
            te_std = m_test[f"grp_{g_col}_{num_col}_std"].fillna(0.0).values
            test_agg_feats[f"grp_{g_col}_{num_col}_mean"] = te_mean
            test_agg_feats[f"grp_{g_col}_{num_col}_std"] = te_std
            test_agg_feats[f"grp_{g_col}_{num_col}_diff_mean"] = (
                test_df[num_col].values - te_mean
            )
            test_agg_feats[f"grp_{g_col}_{num_col}_zscore"] = (
                (test_df[num_col].values - te_mean) / (te_std + eps)
            ).astype(np.float32)

    return train_agg_feats, test_agg_feats


def compute_out_of_fold_target_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_cols: List[str],
    target_col: str,
    fold_indices: List,
    smoothing: float = 10.0,
):
    """Compute leak-free Out-Of-Fold smoothed target encodings for high-cardinality categorical features."""
    train_te = pd.DataFrame(index=train_df.index)
    test_te = pd.DataFrame(index=test_df.index)

    global_mean = float(train_df[target_col].mean())

    for col in cat_cols:
        te_train_col = np.full(len(train_df), np.nan, dtype=np.float32)
        te_test_col = np.zeros(len(test_df), dtype=np.float32)

        for tr_idx, val_idx in fold_indices:
            fold_tr = train_df.iloc[tr_idx]
            fold_val = train_df.iloc[val_idx]

            stats = fold_tr.groupby(col)[target_col].agg(["count", "mean"])
            smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
                stats["count"] + smoothing
            )

            val_mapped = fold_val[col].map(smoothed).fillna(global_mean).values
            te_train_col[val_idx] = val_mapped

            test_mapped = test_df[col].map(smoothed).fillna(global_mean).values
            te_test_col += test_mapped / len(fold_indices)

        train_te[f"te_{col}"] = te_train_col
        test_te[f"te_{col}"] = te_test_col.astype(np.float32)

    return train_te, test_te


def compute_frequency_encoding(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cat_cols: List[str]
):
    """Compute frequency encoding based strictly on training frequency counts."""
    train_fe = pd.DataFrame(index=train_df.index)
    test_fe = pd.DataFrame(index=test_df.index)

    n_train = len(train_df)
    for col in cat_cols:
        counts = train_df[col].value_counts(dropna=True)
        freq_map = counts / n_train

        train_fe[f"freq_{col}"] = (
            train_df[col].map(freq_map).fillna(0.0).astype(np.float32)
        )
        test_fe[f"freq_{col}"] = (
            test_df[col].map(freq_map).fillna(0.0).astype(np.float32)
        )

    return train_fe, test_fe


def run_pipeline(
    input_dir: str = "./input",
    output_dir: str = "./working",
    n_splits: int = 5,
    seed: int = 42,
):
    """Execute end-to-end data loading, feature engineering, and leak-free stratified split generation."""
    os.makedirs(output_dir, exist_ok=True)

    train_df, test_df = load_raw_data(input_dir)
    target_col = "HPSA Score"
    if target_col not in train_df.columns:
        match_col = next(
            (
                c
                for c in train_df.columns
                if "score" in c.lower() or "hpsa" in c.lower()
            ),
            None,
        )
        if match_col:
            target_col = match_col

    col_meta = identify_columns(train_df, target_col=target_col)
    y_train = train_df[target_col].values

    n_bins = min(10, len(np.unique(y_train)))
    target_bins = pd.qcut(y_train, q=n_bins, labels=False, duplicates="drop")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_indices = list(skf.split(train_df, target_bins))

    fold_assignments = np.zeros(len(train_df), dtype=np.int32)
    for fold_idx, (_, val_idx) in enumerate(fold_indices):
        fold_assignments[val_idx] = fold_idx

    train_base_num = pd.DataFrame(index=train_df.index)
    test_base_num = pd.DataFrame(index=test_df.index)

    for col in col_meta["numerical"]:
        train_base_num[col] = train_df[col].astype(np.float32)
        test_base_num[col] = test_df[col].astype(np.float32)

        if train_df[col].isnull().sum() > 0 or test_df[col].isnull().sum() > 0:
            train_base_num[f"isna_{col}"] = train_df[col].isnull().astype(np.int8)
            test_base_num[f"isna_{col}"] = test_df[col].isnull().astype(np.int8)

    train_dt = extract_datetime_features(train_df, col_meta["datetime"])
    test_dt = extract_datetime_features(test_df, col_meta["datetime"])

    train_spatial = extract_spatial_features(train_df, col_meta["numerical"])
    test_spatial = extract_spatial_features(test_df, col_meta["numerical"])

    train_ratios = extract_domain_ratios(train_df, col_meta["numerical"])
    test_ratios = extract_domain_ratios(test_df, col_meta["numerical"])

    train_text_svd, test_text_svd = extract_text_svd_features(
        train_df, test_df, col_meta["text"], n_components=6
    )

    # High-order categorical interactions
    cat_cols = col_meta["categorical"]
    inter_cols = []
    if len(cat_cols) >= 2:
        top_cats = cat_cols[: min(4, len(cat_cols))]
        for i in range(len(top_cats)):
            for j in range(i + 1, len(top_cats)):
                c1, c2 = top_cats[i], top_cats[j]
                inter_name = f"inter_{c1}_{c2}"
                train_df[inter_name] = (
                    train_df[c1].astype(str) + "_" + train_df[c2].astype(str)
                )
                test_df[inter_name] = (
                    test_df[c1].astype(str) + "_" + test_df[c2].astype(str)
                )
                inter_cols.append(inter_name)

    all_cats = cat_cols + inter_cols

    train_fe, test_fe = compute_frequency_encoding(train_df, test_df, all_cats)

    train_te, test_te = compute_out_of_fold_target_encoding(
        train_df, test_df, all_cats, target_col, fold_indices
    )

    group_cats = all_cats[: min(6, len(all_cats))]
    target_nums = col_meta["numerical"]
    train_agg, test_agg = compute_group_aggregations(
        train_df, test_df, group_cats, target_nums
    )

    train_engineered = pd.concat(
        [
            train_base_num,
            train_dt,
            train_spatial,
            train_ratios,
            train_text_svd,
            train_fe,
            train_te,
            train_agg,
        ],
        axis=1,
    )

    test_engineered = pd.concat(
        [
            test_base_num,
            test_dt,
            test_spatial,
            test_ratios,
            test_text_svd,
            test_fe,
            test_te,
            test_agg,
        ],
        axis=1,
    )

    train_engineered = train_engineered.loc[:, ~train_engineered.columns.duplicated()]
    test_engineered = test_engineered.loc[:, ~test_engineered.columns.duplicated()]

    common_cols = [c for c in train_engineered.columns if c in test_engineered.columns]
    X_train = train_engineered[common_cols].copy()
    X_test = test_engineered[common_cols].copy()

    for col in common_cols:
        med_val = X_train[col].median()
        if pd.isna(med_val):
            med_val = 0.0
        X_train[col] = X_train[col].fillna(med_val)
        X_test[col] = X_test[col].fillna(med_val)

    X_train[target_col] = y_train
    X_train["fold"] = fold_assignments

    train_feat_path = os.path.join(output_dir, "train_features.parquet")
    test_feat_path = os.path.join(output_dir, "test_features.parquet")
    meta_path = os.path.join(output_dir, "feature_metadata.json")

    X_train.to_parquet(train_feat_path, index=False)
    X_test.to_parquet(test_feat_path, index=False)

    metadata = {
        "target_col": target_col,
        "n_features": len(common_cols),
        "feature_names": common_cols,
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "n_folds": n_splits,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


# =========================================================================
# 2. Model Architecture Definitions
# =========================================================================


def get_lightgbm_regressor(
    random_state: int = 42, custom_params: Optional[Dict[str, Any]] = None
) -> lgb.LGBMRegressor:
    """Create a regularized LightGBM Regressor configured for RMSE minimization."""
    default_params = {
        "objective": "regression",
        "metric": "rmse",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "num_leaves": 45,
        "max_depth": 7,
        "min_child_samples": 20,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.5,
        "reg_lambda": 1.5,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }
    if custom_params:
        default_params.update(custom_params)
    return lgb.LGBMRegressor(**default_params)


def get_dart_lightgbm_regressor(
    random_state: int = 42, custom_params: Optional[Dict[str, Any]] = None
) -> lgb.LGBMRegressor:
    """Create a DART (Dropouts meet Multiple Additive Regression Trees) LightGBM Regressor."""
    default_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "dart",
        "n_estimators": 1200,
        "learning_rate": 0.05,
        "num_leaves": 35,
        "max_depth": 6,
        "min_child_samples": 25,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "drop_rate": 0.1,
        "max_drop": 50,
        "skip_drop": 0.5,
        "reg_alpha": 0.5,
        "reg_lambda": 1.5,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }
    if custom_params:
        default_params.update(custom_params)
    return lgb.LGBMRegressor(**default_params)


def get_extratrees_regressor(
    random_state: int = 42, custom_params: Optional[Dict[str, Any]] = None
) -> ExtraTreesRegressor:
    """Create an ExtraTrees Regressor for tree-bagging variance reduction."""
    default_params = {
        "n_estimators": 300,
        "max_depth": 18,
        "min_samples_split": 5,
        "min_samples_leaf": 3,
        "max_features": 0.75,
        "bootstrap": True,
        "random_state": random_state,
        "n_jobs": -1,
    }
    if custom_params:
        default_params.update(custom_params)
    return ExtraTreesRegressor(**default_params)


def get_xgboost_regressor(
    random_state: int = 42, custom_params: Optional[Dict[str, Any]] = None
) -> xgb.XGBRegressor:
    """Create a regularized XGBoost Regressor configured for RMSE minimization."""
    default_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 4,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.5,
        "reg_lambda": 1.5,
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
    }
    if custom_params:
        default_params.update(custom_params)
    return xgb.XGBRegressor(**default_params)


def get_catboost_regressor(
    random_state: int = 42, custom_params: Optional[Dict[str, Any]] = None
) -> CatBoostRegressor:
    """Create a regularized CatBoost Regressor configured for RMSE minimization."""
    default_params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 2500,
        "learning_rate": 0.04,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": random_state,
        "verbose": False,
        "thread_count": -1,
    }
    if custom_params:
        default_params.update(custom_params)
    return CatBoostRegressor(**default_params)


class ResidualBlock(nn.Module):
    """Residual Dense Block with Layer Normalization, GELU, and Dropout."""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class TabularResNet(nn.Module):
    """Deep Tabular Network with residual connections for non-linear regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dims[0], dropout_rate=dropout_rate) for _ in range(2)]
        )

        self.transition_layers = nn.ModuleList()
        for in_d, out_d in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.transition_layers.append(
                nn.Sequential(
                    nn.LayerNorm(in_d),
                    nn.Linear(in_d, out_d),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                )
            )

        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        for res_block in self.res_blocks:
            out = res_block(out)
        for transition in self.transition_layers:
            out = transition(out)
        return self.head(out).squeeze(-1)


class RMSELoss(nn.Module):
    """Exact Root Mean Squared Error Loss for PyTorch optimization."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(y_pred, y_true) + self.eps)


def get_tabular_nn_components(
    input_dim: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dims: Optional[List[int]] = None,
    dropout_rate: float = 0.15,
) -> Dict[str, Any]:
    """Instantiate the TabularResNet model, RMSE criterion, AdamW optimizer, and Cosine scheduler."""
    if hidden_dims is None:
        hidden_dims = [256, 128, 64]
    model = TabularResNet(
        input_dim=input_dim, hidden_dims=hidden_dims, dropout_rate=dropout_rate
    )
    criterion = RMSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6
    )

    return {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }


# =========================================================================
# 3. Model Training, Ensembling, and Submission
# =========================================================================


def train_tabular_resnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    epochs: int = 30,
    batch_size: int = 256,
    device: Optional[str] = None,
):
    """Train TabularResNet model on standardized tabular features with early stopping."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    scaler = RobustScaler()
    X_tr_norm = np.nan_to_num(scaler.fit_transform(X_train).astype(np.float32), nan=0.0)
    X_val_norm = np.nan_to_num(scaler.transform(X_val).astype(np.float32), nan=0.0)
    X_test_norm = np.nan_to_num(scaler.transform(X_test).astype(np.float32), nan=0.0)

    train_ds = TensorDataset(
        torch.tensor(X_tr_norm), torch.tensor(y_train, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_norm), torch.tensor(y_val, dtype=torch.float32)
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False, drop_last=False
    )

    components = get_tabular_nn_components(input_dim=X_tr_norm.shape[1], lr=1.5e-3)
    model = components["model"].to(device)
    criterion = components["criterion"]
    optimizer = components["optimizer"]
    scheduler = components["scheduler"]

    best_val_loss = float("inf")
    best_weights = None
    patience = 8
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        val_preds = []
        with torch.no_grad():
            for bx, _ in val_loader:
                bx = bx.to(device)
                pred = model(bx)
                val_preds.append(pred.cpu().numpy())

        val_preds_arr = np.concatenate(val_preds)
        val_rmse = float(np.sqrt(np.mean((val_preds_arr - y_val) ** 2)))

        if val_rmse < best_val_loss:
            best_val_loss = val_rmse
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    model.eval()

    with torch.no_grad():
        oof_pred_list = []
        for bx, _ in val_loader:
            bx = bx.to(device)
            oof_pred_list.append(model(bx).cpu().numpy())
        oof_preds = np.concatenate(oof_pred_list)

        test_tensor = torch.tensor(X_test_norm).to(device)
        test_preds = model(test_tensor).cpu().numpy()

    return oof_preds, test_preds


def train_and_evaluate():
    """Execute end-to-end multi-model cross-validation, ensemble weighting, and test prediction."""
    train_feat_path = "./working/train_features.parquet"
    test_feat_path = "./working/test_features.parquet"

    if not os.path.exists(train_feat_path) or not os.path.exists(test_feat_path):
        run_pipeline()

    train_df = pd.read_parquet(train_feat_path)
    test_df = pd.read_parquet(test_feat_path)

    target_col = "HPSA Score"
    fold_col = "fold"
    feature_cols = [c for c in train_df.columns if c not in [target_col, fold_col]]

    X_train_full = train_df[feature_cols].values
    y_train_full = train_df[target_col].values
    folds = train_df[fold_col].values
    X_test_full = test_df[feature_cols].values

    n_folds = len(np.unique(folds))
    n_train = len(train_df)
    n_test = len(test_df)

    oof_lgb = np.zeros(n_train, dtype=np.float32)
    oof_dart = np.zeros(n_train, dtype=np.float32)
    oof_xgb = np.zeros(n_train, dtype=np.float32)
    oof_cat = np.zeros(n_train, dtype=np.float32)
    oof_et = np.zeros(n_train, dtype=np.float32)
    oof_nn = np.zeros(n_train, dtype=np.float32)

    test_lgb = np.zeros(n_test, dtype=np.float32)
    test_dart = np.zeros(n_test, dtype=np.float32)
    test_xgb = np.zeros(n_test, dtype=np.float32)
    test_cat = np.zeros(n_test, dtype=np.float32)
    test_et = np.zeros(n_test, dtype=np.float32)
    test_nn = np.zeros(n_test, dtype=np.float32)

    for fold_idx in range(n_folds):
        train_idx = np.where(folds != fold_idx)[0]
        val_idx = np.where(folds == fold_idx)[0]

        X_tr, y_tr = X_train_full[train_idx], y_train_full[train_idx]
        X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

        # 1. LightGBM Regressor
        lgb_model = get_lightgbm_regressor(random_state=42 + fold_idx)
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        oof_lgb[val_idx] = lgb_model.predict(X_val)
        test_lgb += lgb_model.predict(X_test_full) / n_folds

        # 2. DART LightGBM Regressor
        dart_model = get_dart_lightgbm_regressor(random_state=42 + fold_idx)
        dart_model.fit(
            X_tr,
            y_tr,
            callbacks=[lgb.log_evaluation(period=0)],
        )
        oof_dart[val_idx] = dart_model.predict(X_val)
        test_dart += dart_model.predict(X_test_full) / n_folds

        # 3. XGBoost Regressor
        xgb_model = get_xgboost_regressor(random_state=42 + fold_idx)
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[val_idx] = xgb_model.predict(X_val)
        test_xgb += xgb_model.predict(X_test_full) / n_folds

        # 4. CatBoost Regressor
        cat_model = get_catboost_regressor(random_state=42 + fold_idx)
        cat_model.fit(
            X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False
        )
        oof_cat[val_idx] = cat_model.predict(X_val)
        test_cat += cat_model.predict(X_test_full) / n_folds

        # 5. ExtraTrees Regressor
        et_model = get_extratrees_regressor(random_state=42 + fold_idx)
        et_model.fit(X_tr, y_tr)
        oof_et[val_idx] = et_model.predict(X_val)
        test_et += et_model.predict(X_test_full) / n_folds

        # 6. Tabular ResNet
        nn_val_pred, nn_test_pred = train_tabular_resnet(
            X_tr, y_tr, X_val, y_val, X_test_full, epochs=25
        )
        oof_nn[val_idx] = nn_val_pred
        test_nn += nn_test_pred / n_folds

    # Multi-Model Stacking and Regularized Ensembling
    OOF_matrix = np.column_stack([oof_lgb, oof_dart, oof_xgb, oof_cat, oof_et, oof_nn])
    TEST_matrix = np.column_stack([test_lgb, test_dart, test_xgb, test_cat, test_et, test_nn])

    # 1. Non-negative RidgeCV Stacking Meta-Learner
    ridge_meta = RidgeCV(alphas=np.logspace(-3, 3, 20), fit_intercept=True)
    ridge_meta.fit(OOF_matrix, y_train_full)
    ridge_oof = ridge_meta.predict(OOF_matrix)
    ridge_test = ridge_meta.predict(TEST_matrix)

    # 2. Constrained SLSQP Blending
    def ensemble_objective(weights):
        w_norm = weights / np.sum(weights)
        pred = OOF_matrix @ w_norm
        reg = 1e-4 * np.sum(weights ** 2)
        return np.sqrt(np.mean((pred - y_train_full) ** 2)) + reg

    init_weights = np.ones(6) / 6.0
    bounds = [(0.0, 1.0) for _ in range(6)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    opt_res = minimize(
        ensemble_objective,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    slsqp_weights = opt_res.x / np.sum(opt_res.x)
    slsqp_oof = OOF_matrix @ slsqp_weights
    slsqp_test = TEST_matrix @ slsqp_weights

    # Blend meta-learner and constrained SLSQP predictions
    oof_final = 0.5 * ridge_oof + 0.5 * slsqp_oof
    oof_final = np.clip(oof_final, 0.0, 26.0)
    final_rmse = float(np.sqrt(np.mean((oof_final - y_train_full) ** 2)))

    # Generate Statutory-Clipped Final Test Predictions
    test_final_preds = 0.5 * ridge_test + 0.5 * slsqp_test
    test_final_preds = np.clip(test_final_preds, 0.0, 26.0)

    # Save Submission File
    os.makedirs("./submission", exist_ok=True)
    submission_df = pd.DataFrame(
        {"id": np.arange(len(test_final_preds)), "HPSA Score": test_final_preds}
    )
    submission_path = "./submission/submission.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {final_rmse}")


if __name__ == "__main__":
    train_and_evaluate()