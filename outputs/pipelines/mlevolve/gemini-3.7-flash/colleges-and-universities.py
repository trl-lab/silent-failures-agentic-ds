import copy
import json
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

try:
    import catboost as cb

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings("ignore")


# ==============================================================================
# 1. Feature Engineering and Preprocessing
# ==============================================================================


def extract_text_and_svd_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_cols: List[str],
    n_components: int = 12,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract length statistics and TF-IDF TruncatedSVD semantic embeddings from text columns."""
    train_dict = {}
    val_dict = {}
    test_dict = {}

    keywords = [
        "university", "college", "institute", "academy", "school", "community",
        "tech", "polytechnic", "seminary", "theological", "medical", "health",
        "nursing", "law", "art", "design", "beauty", "barber", "cosmetology",
        "aviation", "flight", "state", "national", "catholic", "baptist",
        "christian", "graduate", "center", "campus", "system", "vocational",
    ]

    for col in text_cols:
        if col in train_df.columns:
            s_train = train_df[col].astype(str).fillna("")
            s_val = val_df[col].astype(str).fillna("")
            s_test = test_df[col].astype(str).fillna("")

            train_dict[f"{col}_len"] = s_train.str.len().values
            val_dict[f"{col}_len"] = s_val.str.len().values
            test_dict[f"{col}_len"] = s_test.str.len().values

            train_dict[f"{col}_word_count"] = s_train.apply(lambda x: len(x.split())).values
            val_dict[f"{col}_word_count"] = s_val.apply(lambda x: len(x.split())).values
            test_dict[f"{col}_word_count"] = s_test.apply(lambda x: len(x.split())).values

            s_train_lower = s_train.str.lower()
            s_val_lower = s_val.str.lower()
            s_test_lower = s_test.str.lower()

            for kw in keywords:
                train_dict[f"{col}_has_{kw}"] = s_train_lower.apply(lambda x: int(kw in x)).astype(np.int8).values
                val_dict[f"{col}_has_{kw}"] = s_val_lower.apply(lambda x: int(kw in x)).astype(np.int8).values
                test_dict[f"{col}_has_{kw}"] = s_test_lower.apply(lambda x: int(kw in x)).astype(np.int8).values

            # TF-IDF + TruncatedSVD
            try:
                tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
                train_tfidf = tfidf.fit_transform(s_train)
                val_tfidf = tfidf.transform(s_val)
                test_tfidf = tfidf.transform(s_test)

                n_comp = min(n_components, max(1, train_tfidf.shape[1] - 1))
                if n_comp >= 2:
                    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
                    train_svd = svd.fit_transform(train_tfidf)
                    val_svd = svd.transform(val_tfidf)
                    test_svd = svd.transform(test_tfidf)

                    for i in range(n_comp):
                        train_dict[f"{col}_svd_{i}"] = train_svd[:, i]
                        val_dict[f"{col}_svd_{i}"] = val_svd[:, i]
                        test_dict[f"{col}_svd_{i}"] = test_svd[:, i]
            except Exception:
                pass

    return (
        pd.DataFrame(train_dict, index=train_df.index),
        pd.DataFrame(val_dict, index=val_df.index),
        pd.DataFrame(test_dict, index=test_df.index),
    )


def engineer_spatial_and_knn_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    lat_col: Optional[str],
    lon_col: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Engineer rotated coordinates, polar representations, and KNN spatial density metrics."""
    train_dict = {}
    val_dict = {}
    test_dict = {}

    if lat_col and lon_col and lat_col in train_df.columns and lon_col in train_df.columns:
        train_lat = train_df[lat_col].astype(float).fillna(train_df[lat_col].median()).values
        train_lon = train_df[lon_col].astype(float).fillna(train_df[lon_col].median()).values

        val_lat = val_df[lat_col].astype(float).fillna(train_df[lat_col].median()).values
        val_lon = val_df[lon_col].astype(float).fillna(train_df[lon_col].median()).values

        test_lat = test_df[lat_col].astype(float).fillna(train_df[lat_col].median()).values
        test_lon = test_df[lon_col].astype(float).fillna(train_df[lon_col].median()).values

        for angle_deg in [30, 45, 60]:
            rad = np.radians(angle_deg)
            train_dict[f"rot_{angle_deg}_x"] = np.cos(rad) * train_lon - np.sin(rad) * train_lat
            train_dict[f"rot_{angle_deg}_y"] = np.sin(rad) * train_lon + np.cos(rad) * train_lat
            val_dict[f"rot_{angle_deg}_x"] = np.cos(rad) * val_lon - np.sin(rad) * val_lat
            val_dict[f"rot_{angle_deg}_y"] = np.sin(rad) * val_lon + np.cos(rad) * val_lat
            test_dict[f"rot_{angle_deg}_x"] = np.cos(rad) * test_lon - np.sin(rad) * test_lat
            test_dict[f"rot_{angle_deg}_y"] = np.sin(rad) * test_lon + np.cos(rad) * test_lat

        center_lat, center_lon = 39.8283, -98.5795
        for d, lat_arr, lon_arr in [
            (train_dict, train_lat, train_lon),
            (val_dict, val_lat, val_lon),
            (test_dict, test_lat, test_lon),
        ]:
            d_lat = lat_arr - center_lat
            d_lon = lon_arr - center_lon
            d["dist_from_us_center"] = np.sqrt(d_lat**2 + d_lon**2)
            d["angle_from_us_center"] = np.arctan2(d_lat, d_lon)

        # KNN spatial density features
        train_coords = np.column_stack([train_lat, train_lon])
        val_coords = np.column_stack([val_lat, val_lon])
        test_coords = np.column_stack([test_lat, test_lon])

        n_neighbors = min(16, len(train_coords))
        if n_neighbors >= 5:
            knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
            knn.fit(train_coords)

            train_distances, _ = knn.kneighbors(train_coords)
            val_distances, _ = knn.kneighbors(val_coords)
            test_distances, _ = knn.kneighbors(test_coords)

            for k in [3, 5, 10]:
                if k < n_neighbors:
                    train_dict[f"knn_dist_mean_{k}"] = train_distances[:, 1 : k + 1].mean(axis=1)
                    val_dict[f"knn_dist_mean_{k}"] = val_distances[:, :k].mean(axis=1)
                    test_dict[f"knn_dist_mean_{k}"] = test_distances[:, :k].mean(axis=1)

                    train_dict[f"knn_density_{k}"] = 1.0 / (train_dict[f"knn_dist_mean_{k}"] + 1e-4)
                    val_dict[f"knn_density_{k}"] = 1.0 / (val_dict[f"knn_dist_mean_{k}"] + 1e-4)
                    test_dict[f"knn_density_{k}"] = 1.0 / (test_dict[f"knn_dist_mean_{k}"] + 1e-4)

    return (
        pd.DataFrame(train_dict, index=train_df.index),
        pd.DataFrame(val_dict, index=val_df.index),
        pd.DataFrame(test_dict, index=test_df.index),
    )


def engineer_numerical_features_dict(df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
    """Generate log transforms, missing indicators, and interaction ratios without fragmentation."""
    data_dict = {}

    for col in num_cols:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            data_dict[f"{col}_isna"] = vals.isna().astype(np.int8).values

            min_val = vals.min()
            if pd.notna(min_val) and min_val >= 0:
                data_dict[f"{col}_log1p"] = np.log1p(np.maximum(vals.values, 0))

    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            col1 = num_cols[i]
            col2 = num_cols[j]
            if col1 in df.columns and col2 in df.columns:
                v1 = pd.to_numeric(df[col1], errors="coerce").values
                v2 = pd.to_numeric(df[col2], errors="coerce").values
                data_dict[f"ratio_{col1}_div_{col2}"] = v1 / (np.abs(v2) + 1e-5)
                data_dict[f"ratio_{col2}_div_{col1}"] = v2 / (np.abs(v1) + 1e-5)
                data_dict[f"diff_{col1}_minus_{col2}"] = v1 - v2
                data_dict[f"sum_{col1}_plus_{col2}"] = v1 + v2

    return pd.DataFrame(data_dict, index=df.index)


def compute_oof_target_encodings(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: pd.Series,
    cat_cols: List[str],
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute out-of-fold target statistics (mean, median, std) for categoricals to prevent leakage."""
    train_dict = {}
    val_dict = {}
    test_dict = {}

    global_mean = float(y_train.mean())
    global_median = float(y_train.median())
    global_std = float(y_train.std()) if len(y_train) > 1 else 0.0

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for col in cat_cols:
        if col in train_df.columns:
            oof_mean = np.full(len(train_df), global_mean, dtype=float)
            oof_median = np.full(len(train_df), global_median, dtype=float)
            oof_std = np.full(len(train_df), global_std, dtype=float)

            # Inner K-Fold for Train OOF
            train_s = train_df[col].astype(str).values
            for inner_tr_idx, inner_va_idx in kf.split(train_df):
                inner_tr_s = train_df[col].iloc[inner_tr_idx].astype(str)
                inner_tr_y = y_train.iloc[inner_tr_idx]

                grp = inner_tr_y.groupby(inner_tr_s)
                mean_map = grp.mean().to_dict()
                median_map = grp.median().to_dict()
                std_map = grp.std().fillna(0.0).to_dict()

                val_cats = train_df[col].iloc[inner_va_idx].astype(str)
                oof_mean[inner_va_idx] = val_cats.map(mean_map).fillna(global_mean).values
                oof_median[inner_va_idx] = val_cats.map(median_map).fillna(global_median).values
                oof_std[inner_va_idx] = val_cats.map(std_map).fillna(global_std).values

            train_dict[f"{col}_te_mean"] = oof_mean
            train_dict[f"{col}_te_median"] = oof_median
            train_dict[f"{col}_te_std"] = oof_std

            # Full train encoding for Val and Test
            full_grp = y_train.groupby(train_df[col].astype(str))
            full_mean_map = full_grp.mean().to_dict()
            full_median_map = full_grp.median().to_dict()
            full_std_map = full_grp.std().fillna(0.0).to_dict()

            val_cats = val_df[col].astype(str)
            val_dict[f"{col}_te_mean"] = val_cats.map(full_mean_map).fillna(global_mean).values
            val_dict[f"{col}_te_median"] = val_cats.map(full_median_map).fillna(global_median).values
            val_dict[f"{col}_te_std"] = val_cats.map(full_std_map).fillna(global_std).values

            test_cats = test_df[col].astype(str)
            test_dict[f"{col}_te_mean"] = test_cats.map(full_mean_map).fillna(global_mean).values
            test_dict[f"{col}_te_median"] = test_cats.map(full_median_map).fillna(global_median).values
            test_dict[f"{col}_te_std"] = test_cats.map(full_std_map).fillna(global_std).values

    return (
        pd.DataFrame(train_dict, index=train_df.index),
        pd.DataFrame(val_dict, index=val_df.index),
        pd.DataFrame(test_dict, index=test_df.index),
    )


def build_group_aggregations(
    train_df: pd.DataFrame, num_cols: List[str], group_cols: List[str]
) -> Dict[str, pd.DataFrame]:
    """Compute state and category level statistical aggregations solely on train split."""
    agg_tables = {}
    for g_col in group_cols:
        if g_col in train_df.columns:
            agg_dict = {}
            for num_col in num_cols:
                if num_col in train_df.columns:
                    agg_dict[num_col] = ["mean", "std", "min", "max", "median"]
            if agg_dict:
                grouped = train_df.groupby(g_col).agg(agg_dict)
                grouped.columns = [f"{g_col}_{c[0]}_{c[1]}" for c in grouped.columns]
                agg_tables[g_col] = grouped
    return agg_tables


def apply_group_aggregations(
    df: pd.DataFrame,
    group_cols: List[str],
    agg_tables: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge train-derived group aggregation statistics onto any dataset split without fragmentation."""
    cols_dict = {}
    for g_col in group_cols:
        if g_col in df.columns and g_col in agg_tables:
            merged = df[[g_col]].merge(agg_tables[g_col], on=g_col, how="left")
            merged.index = df.index
            cols_to_add = [c for c in merged.columns if c != g_col]
            for c in cols_to_add:
                cols_dict[c] = merged[c].values
    return pd.DataFrame(cols_dict, index=df.index)


# ==============================================================================
# 2. Deep Tabular Neural Network Architecture (Tabular ResNet)
# ==============================================================================


class ResidualBlock(nn.Module):
    """Residual Block for tabular feature representations with LayerNorm and Dropout."""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):
    """Deep Tabular ResNet architecture designed for continuous tabular regression tasks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.2,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )

        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        out = self.head(h)
        return out.squeeze(-1)


def get_loss_function() -> nn.Module:
    """Define the official regression loss function directly aligned with RMSE metric."""
    return nn.MSELoss()


def get_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 100,
) -> Tuple[torch.optim.Optimizer, LRScheduler]:
    """Configure AdamW optimizer with weight decay regularization and Cosine Annealing schedule."""
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "LayerNorm" in name or "layer_norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    return optimizer, scheduler


# ==============================================================================
# 3. Main Execution Pipeline
# ==============================================================================


def main():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # 1. Load Raw Datasets
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_data = pd.read_parquet(train_path)
    test_data = pd.read_parquet(test_path)

    target_col = "RATIO_ENROLL_EMP"
    if target_col not in train_data.columns:
        raise ValueError(f"Target column '{target_col}' not found in train data.")

    y = train_data[target_col].copy()
    X = train_data.drop(columns=[target_col]).copy()
    X_test_raw = test_data.copy()

    id_col = "id" if "id" in X.columns else None

    # 2. Strict Train/Validation Split (80/20)
    SEED = 42
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=SEED, shuffle=True
    )

    # 3. Identify Modalities
    spatial_lat_candidates = [
        c for c in X.columns if c.upper() in ["LATITUDE", "LAT", "Y"]
    ]
    spatial_lon_candidates = [
        c for c in X.columns if c.upper() in ["LONGITUDE", "LON", "X"]
    ]
    lat_col = spatial_lat_candidates[0] if spatial_lat_candidates else None
    lon_col = spatial_lon_candidates[0] if spatial_lon_candidates else None

    text_candidates = [
        c
        for c in X.columns
        if any(
            kw in c.upper()
            for kw in [
                "NAME",
                "ADDRESS",
                "CITY",
                "STREET",
                "WEBSITE",
                "URL",
                "DESCRIPTION",
                "NAICS_DESC",
            ]
        )
    ]

    group_candidates = [
        c
        for c in X.columns
        if any(
            kw in c.upper()
            for kw in [
                "STATE",
                "COUNTY",
                "TYPE",
                "STATUS",
                "DEGREE",
                "LEVEL",
                "SECTOR",
                "LOCALE",
            ]
        )
    ]

    numeric_cols = []
    categorical_cols = []
    for col in X.columns:
        if id_col and col == id_col:
            continue
        if pd.api.types.is_numeric_dtype(X[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    # 4. Feature Transformations
    train_text_feat, val_text_feat, test_text_feat = extract_text_and_svd_features(
        X_train_raw, X_val_raw, X_test_raw, text_candidates
    )

    train_spatial_feat, val_spatial_feat, test_spatial_feat = engineer_spatial_and_knn_features(
        X_train_raw, X_val_raw, X_test_raw, lat_col, lon_col
    )

    train_num_feat = engineer_numerical_features_dict(X_train_raw, numeric_cols)
    val_num_feat = engineer_numerical_features_dict(X_val_raw, numeric_cols)
    test_num_feat = engineer_numerical_features_dict(X_test_raw, numeric_cols)

    agg_tables = build_group_aggregations(X_train_raw, numeric_cols, group_candidates)
    train_grp_feat = apply_group_aggregations(X_train_raw, group_candidates, agg_tables)
    val_grp_feat = apply_group_aggregations(X_val_raw, group_candidates, agg_tables)
    test_grp_feat = apply_group_aggregations(X_test_raw, group_candidates, agg_tables)

    train_cat_feat = pd.DataFrame(index=X_train_raw.index)
    val_cat_feat = pd.DataFrame(index=X_val_raw.index)
    test_cat_feat = pd.DataFrame(index=X_test_raw.index)

    for c_col in categorical_cols:
        freq_map = X_train_raw[c_col].value_counts(normalize=True).to_dict()
        train_cat_feat[f"{c_col}_freq"] = (
            X_train_raw[c_col].map(freq_map).fillna(0).astype(float)
        )
        val_cat_feat[f"{c_col}_freq"] = (
            X_val_raw[c_col].map(freq_map).fillna(0).astype(float)
        )
        test_cat_feat[f"{c_col}_freq"] = (
            X_test_raw[c_col].map(freq_map).fillna(0).astype(float)
        )

        ord_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        train_str = X_train_raw[[c_col]].astype(str).fillna("__MISSING__")
        val_str = X_val_raw[[c_col]].astype(str).fillna("__MISSING__")
        test_str = X_test_raw[[c_col]].astype(str).fillna("__MISSING__")

        train_cat_feat[f"{c_col}_ord"] = ord_encoder.fit_transform(train_str).ravel()
        val_cat_feat[f"{c_col}_ord"] = ord_encoder.transform(val_str).ravel()
        test_cat_feat[f"{c_col}_ord"] = ord_encoder.transform(test_str).ravel()

    train_raw_num = pd.DataFrame(index=X_train_raw.index)
    val_raw_num = pd.DataFrame(index=X_val_raw.index)
    test_raw_num = pd.DataFrame(index=X_test_raw.index)

    for n_col in numeric_cols:
        t_vals = pd.to_numeric(X_train_raw[n_col], errors="coerce")
        median_val = t_vals.median() if not np.isnan(t_vals.median()) else 0.0

        train_raw_num[n_col] = t_vals.fillna(median_val)
        val_raw_num[n_col] = pd.to_numeric(X_val_raw[n_col], errors="coerce").fillna(
            median_val
        )
        test_raw_num[n_col] = pd.to_numeric(X_test_raw[n_col], errors="coerce").fillna(
            median_val
        )

    # 5. Assemble Feature Matrices
    X_train_proc = pd.concat(
        [
            train_raw_num,
            train_cat_feat,
            train_text_feat,
            train_spatial_feat,
            train_num_feat,
            train_grp_feat,
        ],
        axis=1,
    )
    X_val_proc = pd.concat(
        [
            val_raw_num,
            val_cat_feat,
            val_text_feat,
            val_spatial_feat,
            val_num_feat,
            val_grp_feat,
        ],
        axis=1,
    )
    X_test_proc = pd.concat(
        [
            test_raw_num,
            test_cat_feat,
            test_text_feat,
            test_spatial_feat,
            test_num_feat,
            test_grp_feat,
        ],
        axis=1,
    )

    X_train_proc = X_train_proc.loc[:, ~X_train_proc.columns.duplicated()].copy()
    X_val_proc = X_val_proc.loc[:, ~X_val_proc.columns.duplicated()].copy()
    X_test_proc = X_test_proc.loc[:, ~X_test_proc.columns.duplicated()].copy()

    common_features = [
        c
        for c in X_train_proc.columns
        if c in X_val_proc.columns and c in X_test_proc.columns
    ]
    X_train_proc = X_train_proc[common_features].copy()
    X_val_proc = X_val_proc[common_features].copy()
    X_test_proc = X_test_proc[common_features].copy()

    for df in [X_train_proc, X_val_proc, X_test_proc]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in df.columns:
            if df[col].isna().any():
                fill_val = X_train_proc[col].median()
                if pd.isna(fill_val):
                    fill_val = 0.0
                df[col].fillna(fill_val, inplace=True)

    models_val_preds = []
    models_test_preds = []
    model_names = []

    # =========================================================================
    # 6. Train LightGBM
    # =========================================================================
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "num_leaves": 45,
        "max_depth": 7,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_train_proc,
        y_train,
        eval_set=[(X_val_proc, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_pred_lgb = lgb_model.predict(X_val_proc)
    test_pred_lgb = lgb_model.predict(X_test_proc)
    val_rmse_lgb = root_mean_squared_error(y_val, val_pred_lgb)
    print(f"LightGBM Validation RMSE: {val_rmse_lgb:.5f}")

    models_val_preds.append(val_pred_lgb)
    models_test_preds.append(test_pred_lgb)
    model_names.append("LightGBM")

    # =========================================================================
    # 7. Train XGBoost
    # =========================================================================
    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "colsample_bylevel": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.5,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
    }
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(
        X_train_proc,
        y_train,
        eval_set=[(X_val_proc, y_val)],
        verbose=False,
    )

    val_pred_xgb = xgb_model.predict(X_val_proc)
    test_pred_xgb = xgb_model.predict(X_test_proc)
    val_rmse_xgb = root_mean_squared_error(y_val, val_pred_xgb)
    print(f"XGBoost Validation RMSE: {val_rmse_xgb:.5f}")

    models_val_preds.append(val_pred_xgb)
    models_test_preds.append(test_pred_xgb)
    model_names.append("XGBoost")

    # =========================================================================
    # 8. Train CatBoost (if available)
    # =========================================================================
    if HAS_CATBOOST:
        cb_params = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "iterations": 2500,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "subsample": 0.8,
            "random_seed": 42,
            "verbose": 0,
            "thread_count": -1,
            "early_stopping_rounds": 50,
        }
        cb_model = cb.CatBoostRegressor(**cb_params)
        cb_model.fit(
            X_train_proc,
            y_train,
            eval_set=(X_val_proc, y_val),
            verbose=False,
        )

        val_pred_cb = cb_model.predict(X_val_proc)
        test_pred_cb = cb_model.predict(X_test_proc)
        val_rmse_cb = root_mean_squared_error(y_val, val_pred_cb)
        print(f"CatBoost Validation RMSE: {val_rmse_cb:.5f}")

        models_val_preds.append(val_pred_cb)
        models_test_preds.append(test_pred_cb)
        model_names.append("CatBoost")

    # =========================================================================
    # 9. Train TabularResNet
    # =========================================================================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_proc)
    X_val_scaled = scaler.transform(X_val_proc)
    X_test_scaled = scaler.transform(X_test_proc)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = X_train_proc.shape[1]
    nn_model = TabularResNet(
        input_dim=input_dim, hidden_dim=256, num_blocks=3, dropout_rate=0.2
    ).to(device)
    criterion = get_loss_function()

    EPOCHS = 70
    BATCH_SIZE = 128
    optimizer, scheduler = get_optimizer_and_scheduler(
        nn_model, lr=1e-3, weight_decay=1e-4, epochs=EPOCHS
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val.values, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )

    best_nn_val_rmse = float("inf")
    best_nn_state = None

    for epoch in range(1, EPOCHS + 1):
        nn_model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds = nn_model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_y)

        scheduler.step()
        train_loss = train_loss / len(train_dataset)

        nn_model.eval()
        val_preds_list = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                batch_preds = nn_model(batch_x)
                val_preds_list.append(batch_preds.cpu().numpy())

        current_val_preds = np.concatenate(val_preds_list)
        current_val_rmse = root_mean_squared_error(y_val, current_val_preds)

        if current_val_rmse < best_nn_val_rmse:
            best_nn_val_rmse = current_val_rmse
            best_nn_state = copy.deepcopy(nn_model.state_dict())

        if epoch % 10 == 0 or epoch == EPOCHS:
            print(
                f"Epoch {epoch:03d}/{EPOCHS:03d} | Train MSE: {train_loss:.4f} | Val RMSE: {current_val_rmse:.4f} | Best Val RMSE: {best_nn_val_rmse:.4f}"
            )

    nn_model.load_state_dict(best_nn_state)
    nn_model.eval()

    with torch.no_grad():
        val_pred_nn = (
            nn_model(torch.tensor(X_val_scaled, dtype=torch.float32).to(device))
            .cpu()
            .numpy()
        )
        test_pred_nn = (
            nn_model(torch.tensor(X_test_scaled, dtype=torch.float32).to(device))
            .cpu()
            .numpy()
        )

    val_rmse_nn = root_mean_squared_error(y_val, val_pred_nn)
    print(f"TabularResNet Best Validation RMSE: {val_rmse_nn:.5f}")

    models_val_preds.append(val_pred_nn)
    models_test_preds.append(test_pred_nn)
    model_names.append("TabularResNet")

    # =========================================================================
    # 10. Ensemble Optimization and Submission
    # =========================================================================
    val_matrix = np.column_stack(models_val_preds)
    test_matrix = np.column_stack(models_test_preds)
    num_models = val_matrix.shape[1]

    def ensemble_objective(weights):
        w_pred = np.dot(val_matrix, weights)
        return root_mean_squared_error(y_val, np.clip(w_pred, 0, None))

    initial_weights = np.ones(num_models) / num_models
    bounds = [(0.0, 1.0) for _ in range(num_models)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    opt_result = minimize(
        ensemble_objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    best_weights = opt_result.x
    best_weights = np.maximum(best_weights, 0)
    best_weights /= np.sum(best_weights)

    for name, w in zip(model_names, best_weights):
        print(f"Model: {name:15s} | Weight: {w:.4f}")

    val_ensemble_pred = np.clip(np.dot(val_matrix, best_weights), 0, None)
    test_ensemble_pred = np.clip(np.dot(test_matrix, best_weights), 0, None)

    final_val_score = root_mean_squared_error(y_val, val_ensemble_pred)

    submission_path = "./submission/submission.csv"
    submission_df = pd.DataFrame(
        {
            "id": np.arange(len(test_ensemble_pred), dtype=int),
            "RATIO_ENROLL_EMP": test_ensemble_pred,
        }
    )
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved to: {submission_path} with shape {submission_df.shape}")
    print(f"Final Validation Score: {final_val_score}")


if __name__ == "__main__":
    main()