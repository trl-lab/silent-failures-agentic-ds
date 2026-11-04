import os
import re
import warnings
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")


# ==============================================================================
# 1. Feature Engineering Utilities
# ==============================================================================


def extract_generation_features(df, gen_cols, prefix="gen"):
    """Extract statistical and trend aggregates from time-series generation columns."""
    if not gen_cols:
        return pd.DataFrame(index=df.index)

    gen_df = df[gen_cols].apply(pd.to_numeric, errors="coerce")
    features = pd.DataFrame(index=df.index)

    features[f"{prefix}_mean"] = gen_df.mean(axis=1)
    features[f"{prefix}_std"] = gen_df.std(axis=1).fillna(0)
    features[f"{prefix}_min"] = gen_df.min(axis=1)
    features[f"{prefix}_max"] = gen_df.max(axis=1)
    features[f"{prefix}_sum"] = gen_df.sum(axis=1)
    features[f"{prefix}_median"] = gen_df.median(axis=1)
    features[f"{prefix}_cv"] = features[f"{prefix}_std"] / (
        features[f"{prefix}_mean"] + 1e-4
    )
    features[f"{prefix}_peak_to_mean"] = features[f"{prefix}_max"] / (
        features[f"{prefix}_mean"] + 1e-4
    )
    features[f"{prefix}_non_zeros"] = (gen_df > 0).sum(axis=1)
    features[f"{prefix}_missing_count"] = gen_df.isna().sum(axis=1)

    years = []
    for c in gen_cols:
        matches = re.findall(r"\d{4}", str(c))
        years.append(int(matches[0]) if matches else 0)

    if len(years) >= 2 and all(y > 0 for y in years):
        sorted_indices = np.argsort(years)
        first_col = gen_cols[sorted_indices[0]]
        last_col = gen_cols[sorted_indices[-1]]
        year_span = max(1, years[sorted_indices[-1]] - years[sorted_indices[0]])
        features[f"{prefix}_overall_trend"] = (
            gen_df[last_col] - gen_df[first_col]
        ) / year_span

        weights = np.linspace(1.0, 2.0, len(gen_cols))
        weighted_sum = np.zeros(len(df))
        weight_normalizer = np.zeros(len(df))
        for col_idx, col in enumerate(gen_cols):
            val = gen_df[col].values
            valid_mask = ~np.isnan(val)
            weighted_sum[valid_mask] += val[valid_mask] * weights[col_idx]
            weight_normalizer[valid_mask] += weights[col_idx]
        features[f"{prefix}_weighted_recent_mean"] = np.where(
            weight_normalizer > 0,
            weighted_sum / np.maximum(weight_normalizer, 1e-4),
            np.nan,
        )
    return features


def extract_spatial_features(df, train_coords=None, n_clusters=15):
    """Compute geometric and spatial clustering features from coordinates."""
    features = pd.DataFrame(index=df.index)
    has_lat = "latitude" in df.columns
    has_lon = "longitude" in df.columns

    if has_lat and has_lon:
        lat = pd.to_numeric(df["latitude"], errors="coerce")
        lon = pd.to_numeric(df["longitude"], errors="coerce")

        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        features["coord_x"] = np.cos(lat_rad) * np.cos(lon_rad)
        features["coord_y"] = np.cos(lat_rad) * np.sin(lon_rad)
        features["coord_z"] = np.sin(lat_rad)
        features["abs_latitude"] = np.abs(lat)
        features["coord_polar_r"] = np.sqrt(lat**2 + lon**2)
        features["coord_polar_theta"] = np.arctan2(lat, lon)

        if train_coords is not None:
            valid_train = train_coords.dropna()
            if len(valid_train) >= n_clusters:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(valid_train[["latitude", "longitude"]])

                coords_filled = df[["latitude", "longitude"]].fillna(
                    train_coords.mean()
                )
                features["spatial_cluster"] = kmeans.predict(coords_filled)
                cluster_centers = kmeans.cluster_centers_

                dists = np.zeros((len(df), n_clusters))
                for k in range(n_clusters):
                    dists[:, k] = np.sqrt(
                        (coords_filled["latitude"].values - cluster_centers[k, 0]) ** 2
                        + (coords_filled["longitude"].values - cluster_centers[k, 1])
                        ** 2
                    )
                features["dist_nearest_cluster"] = np.min(dists, axis=1)
                features["dist_mean_cluster"] = np.mean(dists, axis=1)
    return features


def extract_text_name_features(df):
    """Extract metadata heuristics from power plant names and identifiers."""
    features = pd.DataFrame(index=df.index)
    if "name" in df.columns:
        name_str = df["name"].fillna("").astype(str)
        features["name_length"] = name_str.str.len()
        features["name_word_count"] = name_str.apply(lambda s: len(s.split()))
        features["name_digit_count"] = name_str.apply(
            lambda s: sum(c.isdigit() for c in s)
        )

        keywords = [
            "solar",
            "wind",
            "hydro",
            "thermal",
            "nuclear",
            "gas",
            "coal",
            "station",
            "phase",
            "project",
        ]
        for kw in keywords:
            features[f"name_contains_{kw}"] = (
                name_str.str.lower().str.contains(kw).astype(int)
            )

        def parse_capacity_from_name(val):
            val = str(val).lower()
            match = re.search(
                r"(\d+(?:\.\d+)?)\s*(gw|gigawatt|mw|megawatt|kw|kilowatt)", val
            )
            if match:
                num = float(match.group(1))
                unit = match.group(2)
                if "gw" in unit or "gigawatt" in unit:
                    return num * 1000.0
                elif "kw" in unit or "kilowatt" in unit:
                    return num / 1000.0
                else:
                    return num
            match_plain = re.search(r"(\d+(?:\.\d+)?)", val)
            if match_plain:
                try:
                    num = float(match_plain.group(1))
                    if 0 < num < 50000:
                        return num
                except ValueError:
                    pass
            return np.nan

        features["name_extracted_num"] = name_str.apply(parse_capacity_from_name)
    return features


def extract_temporal_features(df):
    """Extract operational age and era features."""
    features = pd.DataFrame(index=df.index)
    ref_year = 2021

    if "commissioning_year" in df.columns:
        comm_yr = pd.to_numeric(df["commissioning_year"], errors="coerce")
        features["comm_year"] = comm_yr
        features["plant_age"] = ref_year - comm_yr
        features["is_historic"] = (comm_yr < 1980).astype(int)
        features["is_modern"] = (comm_yr >= 2010).astype(int)
        features["comm_decade"] = (comm_yr // 10) * 10

    if "year_of_capacity_data" in df.columns:
        cap_yr = pd.to_numeric(df["year_of_capacity_data"], errors="coerce")
        features["capacity_data_age"] = ref_year - cap_yr
        if "commissioning_year" in df.columns:
            features["years_until_capacity_recorded"] = cap_yr - pd.to_numeric(
                df["commissioning_year"], errors="coerce"
            )
    return features


# ==============================================================================
# 2. Pipeline Data Processing & CV Setup
# ==============================================================================


def process_and_engineer_features(
    train_path="./input/train.parquet", test_path="./input/test.parquet"
):
    """Main pipeline for feature engineering, encoding, and fold splitting."""
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "capacity_mw"
    y = train_df[target_col].copy()

    actual_gen_cols = [
        c for c in train_df.columns if re.match(r"^generation_gwh_\d{4}$", c)
    ]
    est_gen_cols = [
        c for c in train_df.columns if re.match(r"^estimated_generation_gwh_\d{4}$", c)
    ]

    train_feats = pd.DataFrame(index=train_df.index)
    test_feats = pd.DataFrame(index=test_df.index)

    # 1. Generation Features
    train_gen = extract_generation_features(
        train_df, actual_gen_cols, prefix="actual_gen"
    )
    test_gen = extract_generation_features(
        test_df, actual_gen_cols, prefix="actual_gen"
    )
    train_est_gen = extract_generation_features(
        train_df, est_gen_cols, prefix="est_gen"
    )
    test_est_gen = extract_generation_features(test_df, est_gen_cols, prefix="est_gen")

    train_feats = pd.concat([train_feats, train_gen, train_est_gen], axis=1)
    test_feats = pd.concat([test_feats, test_gen, test_est_gen], axis=1)

    # 2. Spatial Features
    spatial_coords = (
        train_df[["latitude", "longitude"]]
        if ("latitude" in train_df.columns and "longitude" in train_df.columns)
        else None
    )
    train_spatial = extract_spatial_features(train_df, train_coords=spatial_coords)
    test_spatial = extract_spatial_features(test_df, train_coords=spatial_coords)

    train_feats = pd.concat([train_feats, train_spatial], axis=1)
    test_feats = pd.concat([test_feats, test_spatial], axis=1)

    # 3. Temporal Features
    train_temp = extract_temporal_features(train_df)
    test_temp = extract_temporal_features(test_df)

    train_feats = pd.concat([train_feats, train_temp], axis=1)
    test_feats = pd.concat([test_feats, test_temp], axis=1)

    # 4. Text / Name Features
    train_text = extract_text_name_features(train_df)
    test_text = extract_text_name_features(test_df)

    train_feats = pd.concat([train_feats, train_text], axis=1)
    test_feats = pd.concat([test_feats, test_text], axis=1)

    # 5. Multi-fuel combinations & Hierarchical Interactions
    fuel_cols = [
        c
        for c in ["primary_fuel", "other_fuel1", "other_fuel2", "other_fuel3"]
        if c in train_df.columns
    ]
    if fuel_cols:
        for df, feats in [(train_df, train_feats), (test_df, test_feats)]:
            feats["num_fuels_reported"] = df[fuel_cols].notna().sum(axis=1)
            if "primary_fuel" in df.columns and "other_fuel1" in df.columns:
                feats["fuel_combination"] = (
                    df["primary_fuel"].fillna("None").astype(str)
                    + "_"
                    + df["other_fuel1"].fillna("None").astype(str)
                )
            elif "primary_fuel" in df.columns:
                feats["fuel_combination"] = (
                    df["primary_fuel"].fillna("None").astype(str)
                )

    if "country" in train_df.columns and "primary_fuel" in train_df.columns:
        for df, feats in [(train_df, train_feats), (test_df, test_feats)]:
            feats["country_fuel"] = (
                df["country"].fillna("None").astype(str)
                + "_"
                + df["primary_fuel"].fillna("None").astype(str)
            )

    # Physical capacity factor proxies and generation consistency
    fuel_cf_map = {
        "nuclear": 0.90,
        "geothermal": 0.75,
        "biomass": 0.60,
        "waste": 0.60,
        "coal": 0.55,
        "gas": 0.45,
        "hydro": 0.40,
        "wind": 0.35,
        "oil": 0.30,
        "solar": 0.20,
        "petroleum": 0.30,
        "cogeneration": 0.60,
        "storage": 0.20,
        "wave and tidal": 0.25,
        "other": 0.40,
    }

    for df, feats in [(train_df, train_feats), (test_df, test_feats)]:
        if "primary_fuel" in df.columns:
            fuel_s = df["primary_fuel"].fillna("other").astype(str).str.lower()
            cf_s = fuel_s.map(fuel_cf_map).fillna(0.40)
            feats["fuel_capacity_factor"] = cf_s.astype(np.float32)

            if "actual_gen_mean" in feats.columns:
                feats["implied_mw_actual_mean"] = (
                    feats["actual_gen_mean"] / (8.76 * cf_s + 1e-6)
                ).astype(np.float32)
                feats["implied_mw_actual_max"] = (
                    feats["actual_gen_max"] / (8.76 * cf_s + 1e-6)
                ).astype(np.float32)
            if "est_gen_mean" in feats.columns:
                feats["implied_mw_est_mean"] = (
                    feats["est_gen_mean"] / (8.76 * cf_s + 1e-6)
                ).astype(np.float32)
            if "actual_gen_weighted_recent_mean" in feats.columns:
                feats["implied_mw_recent"] = (
                    feats["actual_gen_weighted_recent_mean"] / (8.76 * cf_s + 1e-6)
                ).astype(np.float32)

        if "actual_gen_mean" in feats.columns and "est_gen_mean" in feats.columns:
            feats["gen_ratio_actual_to_est"] = (
                feats["actual_gen_mean"] / (feats["est_gen_mean"] + 1e-4)
            ).astype(np.float32)
            feats["gen_diff_actual_est"] = (
                feats["actual_gen_mean"] - feats["est_gen_mean"]
            ).astype(np.float32)
            feats["gen_max_actual_est"] = np.maximum(
                feats["actual_gen_max"].fillna(0), feats["est_gen_max"].fillna(0)
            ).astype(np.float32)

    # 6. Raw Numeric and Categorical Features
    cat_cols_candidates = [
        "country",
        "country_long",
        "primary_fuel",
        "other_fuel1",
        "other_fuel2",
        "owner",
        "source",
        "geolocation_source",
        "generation_data_source",
        "fuel_combination",
        "country_fuel",
    ]
    cat_cols = [
        c
        for c in cat_cols_candidates
        if c in train_df.columns or c in train_feats.columns
    ]

    raw_num_cols = [
        c
        for c in train_df.columns
        if c not in cat_cols
        and c not in [target_col, "name", "gppd_idnr", "wepp_id", "url"]
        and train_df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]
    for col in raw_num_cols:
        if col not in train_feats.columns:
            train_feats[col] = pd.to_numeric(train_df[col], errors="coerce")
            test_feats[col] = pd.to_numeric(test_df[col], errors="coerce")

    # 7. Frequency Encoding & Unified Category Types
    for col in cat_cols:
        series_train = train_df[col] if col in train_df.columns else train_feats[col]
        series_test = test_df[col] if col in test_df.columns else test_feats[col]

        freq_map = series_train.astype(str).value_counts(normalize=True).to_dict()
        train_feats[f"{col}_freq"] = (
            series_train.astype(str).map(freq_map).fillna(0).astype(np.float32)
        )
        test_feats[f"{col}_freq"] = (
            series_test.astype(str).map(freq_map).fillna(0).astype(np.float32)
        )

        all_cats = (
            pd.concat([series_train, series_test])
            .fillna("Unknown")
            .astype(str)
            .unique()
        )
        cat_type = pd.CategoricalDtype(categories=all_cats)
        train_feats[col] = series_train.fillna("Unknown").astype(str).astype(cat_type)
        test_feats[col] = series_test.fillna("Unknown").astype(str).astype(cat_type)

    # 8. Leak-Free Stratified 5-Fold Splits
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_bins = pd.qcut(y, q=10, labels=False, duplicates="drop").astype(str)
    strat_key = y_bins
    if "primary_fuel" in train_df.columns:
        fuel_key = train_df["primary_fuel"].fillna("Unknown").astype(str)
        strat_key = strat_key + "_" + fuel_key
        counts = strat_key.value_counts()
        strat_key = strat_key.apply(lambda k: k if counts[k] >= n_splits else "other")

    folds = np.zeros(len(train_df), dtype=int)
    for fold, (_, val_idx) in enumerate(skf.split(train_feats, strat_key)):
        folds[val_idx] = fold
    train_feats["fold"] = folds

    # 9. Out-of-fold Group Target Statistics (Leak-Free)
    group_keys = [
        c
        for c in ["country_fuel", "primary_fuel", "country", "fuel_combination"]
        if c in train_feats.columns
    ]
    for grp in group_keys:
        train_feats[f"{grp}_target_mean_oof"] = np.nan
        train_feats[f"{grp}_target_std_oof"] = np.nan
        test_feats[f"{grp}_target_mean_oof"] = np.nan
        test_feats[f"{grp}_target_std_oof"] = np.nan

        for fold in range(n_splits):
            tr_idx = folds != fold
            val_idx = folds == fold

            grp_series_tr = train_feats.loc[tr_idx, grp].astype(str)
            grp_stats = (
                train_df.loc[tr_idx]
                .groupby(grp_series_tr)[target_col]
                .agg(["mean", "std"])
            )

            val_grps = train_feats.loc[val_idx, grp].astype(str)
            train_feats.loc[val_idx, f"{grp}_target_mean_oof"] = (
                val_grps.map(grp_stats["mean"]).astype(np.float32)
            )
            train_feats.loc[val_idx, f"{grp}_target_std_oof"] = (
                val_grps.map(grp_stats["std"]).fillna(0).astype(np.float32)
            )

        full_grp_series = train_feats[grp].astype(str)
        full_grp_stats = train_df.groupby(full_grp_series)[target_col].agg(
            ["mean", "std"]
        )
        test_grps = test_feats[grp].astype(str)
        test_feats[f"{grp}_target_mean_oof"] = (
            test_grps.map(full_grp_stats["mean"]).astype(np.float32)
        )
        test_feats[f"{grp}_target_std_oof"] = (
            test_grps.map(full_grp_stats["std"]).fillna(0).astype(np.float32)
        )

    for col in train_feats.columns:
        if train_feats[col].dtype == "object" or (
            col in test_feats.columns and test_feats[col].dtype == "object"
        ):
            all_u = pd.concat(
                [
                    train_feats[col].astype(str),
                    test_feats[col].astype(str)
                    if col in test_feats.columns
                    else pd.Series(dtype=str),
                ]
            ).unique()
            c_type = pd.CategoricalDtype(categories=all_u)
            train_feats[col] = train_feats[col].astype(str).astype(c_type)
            if col in test_feats.columns:
                test_feats[col] = test_feats[col].astype(str).astype(c_type)

    train_feats[target_col] = y.values
    return train_feats, test_feats


# ==============================================================================
# 3. Model Architectures & Configurations
# ==============================================================================


def get_lgb_params(seed: int = 42) -> dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_estimators": 3000,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_params(seed: int = 42) -> dict:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "learning_rate": 0.03,
        "max_depth": 7,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_estimators": 3000,
        "n_jobs": -1,
    }


def get_cb_params(seed: int = 42) -> dict:
    return {
        "iterations": 3000,
        "learning_rate": 0.03,
        "depth": 6,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": seed,
        "verbose": False,
        "early_stopping_rounds": 60,
    }


class ResidualBlock(nn.Module):
    """Dense Residual Block with Layer Normalization, Dropout, and Swish/SiLU."""

    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.15):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_features)
        self.linear1 = nn.Linear(in_features, out_features)
        self.activation = nn.SiLU()
        self.norm2 = nn.LayerNorm(out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(dropout_rate)

        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.norm1(x)
        out = self.linear1(out)
        out = self.activation(out)
        out = self.norm2(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return self.activation(out + residual)


class TabularResNet(nn.Module):
    """Deep Tabular ResNet with categorical entity embeddings and continuous projections."""

    def __init__(
        self,
        num_continuous_features: int,
        categorical_cardinalities: list = None,
        hidden_dims: list = None,
        dropout_rate: float = 0.15,
        output_dim: int = 1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 128, 64]

        self.embeddings = nn.ModuleList()
        total_emb_dim = 0
        if categorical_cardinalities:
            for num_cats in categorical_cardinalities:
                emb_dim = min(50, max(4, (num_cats + 1) // 2))
                self.embeddings.append(
                    nn.Embedding(num_cats + 1, emb_dim, padding_idx=0)
                )
                total_emb_dim += emb_dim

        in_features = num_continuous_features + total_emb_dim
        self.input_norm = (
            nn.LayerNorm(num_continuous_features)
            if num_continuous_features > 0
            else nn.Identity()
        )
        self.input_projection = nn.Linear(in_features, hidden_dims[0])

        blocks = []
        for i in range(len(hidden_dims) - 1):
            blocks.append(
                ResidualBlock(
                    hidden_dims[i], hidden_dims[i + 1], dropout_rate=dropout_rate
                )
            )
        self.res_blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dims[-1], output_dim),
        )

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor = None) -> torch.Tensor:
        features = []
        if x_cont is not None and x_cont.shape[1] > 0:
            features.append(self.input_norm(x_cont))

        if x_cat is not None and len(self.embeddings) > 0:
            emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            features.extend(emb_outs)

        x = torch.cat(features, dim=1) if len(features) > 1 else features[0]
        x = nn.SiLU()(self.input_projection(x))
        x = self.res_blocks(x)
        return self.head(x).squeeze(-1)


# ==============================================================================
# 4. Training, Ensembling & Submission
# ==============================================================================


def run_pipeline():
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Feature Engineering
    train_df, test_df = process_and_engineer_features()

    target_col = "capacity_mw"
    fold_col = "fold"

    feature_cols = [c for c in train_df.columns if c not in [target_col, fold_col]]
    cat_cols = [c for c in feature_cols if train_df[c].dtype.name == "category"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    y_train_full = train_df[target_col].values.astype(np.float32)
    n_samples = len(train_df)
    n_test = len(test_df)
    n_folds = train_df[fold_col].nunique()

    oof_lgb = np.zeros(n_samples, dtype=np.float32)
    test_lgb = np.zeros(n_test, dtype=np.float32)

    oof_xgb = np.zeros(n_samples, dtype=np.float32)
    test_xgb = np.zeros(n_test, dtype=np.float32)

    oof_cb = np.zeros(n_samples, dtype=np.float32)
    test_cb = np.zeros(n_test, dtype=np.float32)

    oof_nn = np.zeros(n_samples, dtype=np.float32)
    test_nn = np.zeros(n_test, dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================================================
    # Model 1: LightGBM
    # ==========================================================================
    lgb_params = get_lgb_params(seed=42)

    for fold in range(n_folds):
        tr_mask = train_df[fold_col] != fold
        val_mask = train_df[fold_col] == fold

        X_tr, y_tr = train_df.loc[tr_mask, feature_cols], y_train_full[tr_mask]
        X_val, y_val = (
            train_df.loc[val_mask, feature_cols],
            y_train_full[val_mask],
        )

        dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
        dval = lgb.Dataset(
            X_val, label=y_val, reference=dtrain, categorical_feature=cat_cols
        )

        model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=lgb_params["n_estimators"],
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=60, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        val_preds = model.predict(X_val)
        oof_lgb[val_mask] = val_preds
        test_lgb += model.predict(test_df[feature_cols]) / n_folds

    # ==========================================================================
    # Model 2: XGBoost
    # ==========================================================================
    xgb_params = get_xgb_params(seed=42)
    dtest = xgb.DMatrix(test_df[feature_cols], enable_categorical=True)

    for fold in range(n_folds):
        tr_mask = train_df[fold_col] != fold
        val_mask = train_df[fold_col] == fold

        X_tr, y_tr = train_df.loc[tr_mask, feature_cols], y_train_full[tr_mask]
        X_val, y_val = (
            train_df.loc[val_mask, feature_cols],
            y_train_full[val_mask],
        )

        dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
        dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=xgb_params["n_estimators"],
            evals=[(dval, "val")],
            early_stopping_rounds=60,
            verbose_eval=False,
        )

        val_preds = model.predict(dval)
        oof_xgb[val_mask] = val_preds
        test_xgb += model.predict(dtest) / n_folds

    # ==========================================================================
    # Model 3: CatBoost
    # ==========================================================================
    cb_params = get_cb_params(seed=42)
    X_test_cb = test_df[feature_cols].copy()
    for c in cat_cols:
        X_test_cb[c] = X_test_cb[c].astype(str)

    for fold in range(n_folds):
        tr_mask = train_df[fold_col] != fold
        val_mask = train_df[fold_col] == fold

        X_tr = train_df.loc[tr_mask, feature_cols].copy()
        y_tr = y_train_full[tr_mask]
        X_val = train_df.loc[val_mask, feature_cols].copy()
        y_val = y_train_full[val_mask]

        for c in cat_cols:
            X_tr[c] = X_tr[c].astype(str)
            X_val[c] = X_val[c].astype(str)

        cb_model = CatBoostRegressor(**cb_params)
        cb_model.fit(
            X_tr,
            y_tr,
            eval_set=(X_val, y_val),
            cat_features=cat_cols,
            use_best_model=True,
            verbose=False,
        )

        oof_cb[val_mask] = cb_model.predict(X_val)
        test_cb += cb_model.predict(X_test_cb) / n_folds

    # ==========================================================================
    # Model 4: TabularResNet PyTorch
    # ==========================================================================
    epochs_nn = 35
    batch_size = 256

    for fold in range(n_folds):
        tr_mask = train_df[fold_col] != fold
        val_mask = train_df[fold_col] == fold

        scaler = StandardScaler()
        X_tr_num = scaler.fit_transform(
            np.nan_to_num(train_df.loc[tr_mask, num_cols].values, nan=0.0)
        )
        X_val_num = scaler.transform(
            np.nan_to_num(train_df.loc[val_mask, num_cols].values, nan=0.0)
        )
        X_test_num = scaler.transform(np.nan_to_num(test_df[num_cols].values, nan=0.0))

        cardinalities = []
        X_tr_cat_list, X_val_cat_list, X_test_cat_list = [], [], []

        for col in cat_cols:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            tr_cat = encoder.fit_transform(
                train_df.loc[tr_mask, [col]].astype(str)
            ).astype(np.int64)
            val_cat = encoder.transform(
                train_df.loc[val_mask, [col]].astype(str)
            ).astype(np.int64)
            test_cat = encoder.transform(test_df[[col]].astype(str)).astype(np.int64)

            tr_cat = tr_cat + 1
            val_cat = np.clip(val_cat + 1, 0, len(encoder.categories_[0]))
            test_cat = np.clip(test_cat + 1, 0, len(encoder.categories_[0]))

            cardinalities.append(len(encoder.categories_[0]) + 1)
            X_tr_cat_list.append(tr_cat)
            X_val_cat_list.append(val_cat)
            X_test_cat_list.append(test_cat)

        X_tr_cat = (
            np.hstack(X_tr_cat_list)
            if X_tr_cat_list
            else np.empty((np.sum(tr_mask), 0))
        )
        X_val_cat = (
            np.hstack(X_val_cat_list)
            if X_val_cat_list
            else np.empty((np.sum(val_mask), 0))
        )
        X_test_cat = (
            np.hstack(X_test_cat_list) if X_test_cat_list else np.empty((n_test, 0))
        )

        y_tr_t = torch.tensor(y_train_full[tr_mask], dtype=torch.float32)
        y_val_t = torch.tensor(y_train_full[val_mask], dtype=torch.float32)

        tr_dataset = TensorDataset(
            torch.tensor(X_tr_num, dtype=torch.float32),
            torch.tensor(X_tr_cat, dtype=torch.long),
            y_tr_t,
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_num, dtype=torch.float32),
            torch.tensor(X_val_cat, dtype=torch.long),
            y_val_t,
        )

        tr_loader = DataLoader(
            tr_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

        model_nn = TabularResNet(
            num_continuous_features=X_tr_num.shape[1],
            categorical_cardinalities=cardinalities,
            hidden_dims=[256, 256, 128, 64],
            dropout_rate=0.15,
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = AdamW(model_nn.parameters(), lr=3e-3, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs_nn, eta_min=1e-6)

        best_val_loss = float("inf")
        best_preds = None

        for epoch in range(epochs_nn):
            model_nn.train()
            for b_num, b_cat, b_y in tr_loader:
                b_num, b_cat, b_y = (
                    b_num.to(device),
                    b_cat.to(device),
                    b_y.to(device),
                )
                optimizer.zero_grad()
                pred = model_nn(b_num, b_cat)
                loss = criterion(pred, b_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_nn.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            model_nn.eval()
            val_losses = []
            val_preds_list = []
            with torch.no_grad():
                for b_num, b_cat, b_y in val_loader:
                    b_num, b_cat, b_y = (
                        b_num.to(device),
                        b_cat.to(device),
                        b_y.to(device),
                    )
                    pred = model_nn(b_num, b_cat)
                    val_losses.append(criterion(pred, b_y).item())
                    val_preds_list.append(pred.cpu().numpy().flatten())

            epoch_val_rmse = np.sqrt(np.mean(val_losses))
            if epoch_val_rmse < best_val_loss:
                best_val_loss = epoch_val_rmse
                best_preds = np.concatenate(val_preds_list)

        oof_nn[val_mask] = best_preds

        model_nn.eval()
        with torch.no_grad():
            t_num = torch.tensor(X_test_num, dtype=torch.float32).to(device)
            t_cat = torch.tensor(X_test_cat, dtype=torch.long).to(device)
            test_nn += model_nn(t_num, t_cat).cpu().numpy().flatten() / n_folds

    # ==========================================================================
    # Ensemble Optimization & Output
    # ==========================================================================
    oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cb, oof_nn])
    test_matrix = np.column_stack([test_lgb, test_xgb, test_cb, test_nn])
    num_models = oof_matrix.shape[1]

    def objective(weights):
        pred = np.dot(oof_matrix, weights)
        pred = np.clip(pred, a_min=0.0, a_max=None)
        return np.sqrt(mean_squared_error(y_train_full, pred))

    init_weights = np.ones(num_models) / num_models
    bounds = [(0.0, 1.0) for _ in range(num_models)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    opt_res = minimize(
        objective,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    if opt_res.success:
        best_weights = opt_res.x
        best_weights = np.maximum(0, best_weights)
        best_weights = best_weights / np.sum(best_weights)
    else:
        best_weights = init_weights

    oof_ensemble = np.dot(oof_matrix, best_weights)
    oof_ensemble = np.clip(oof_ensemble, a_min=0.0, a_max=None)

    final_rmse = np.sqrt(mean_squared_error(y_train_full, oof_ensemble))

    test_ensemble = np.dot(test_matrix, best_weights)
    test_ensemble = np.clip(test_ensemble, a_min=0.0, a_max=None)

    os.makedirs("./submission", exist_ok=True)
    submission_df = pd.DataFrame(
        {"id": np.arange(len(test_ensemble)), "capacity_mw": test_ensemble}
    )
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_rmse}")


if __name__ == "__main__":
    run_pipeline()
