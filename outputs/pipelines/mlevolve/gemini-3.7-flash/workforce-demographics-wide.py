import json
import math
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls
from catboost import CatBoostRegressor
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PowerTransformer, StandardScaler
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")

# Set global deterministic random seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# 1. Feature Engineering & Leakage-Free Preprocessing
# ==============================================================================
def sanitize_column_name(col_name: str) -> str:
    """Sanitize column names for compatibility with tree-based models."""
    cleaned = re.sub(r'[\[\]<>\s,:{}"\'/\\+]+', "_", str(col_name)).strip("_")
    return cleaned if cleaned else "feature"


def calculate_shannon_entropy(probs: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Compute normalized Shannon entropy across a set of demographic percentage features."""
    p_sum = np.sum(probs, axis=1, keepdims=True) + eps
    p_normalized = probs / p_sum
    p_normalized = np.clip(p_normalized, eps, 1.0)
    raw_entropy = -np.sum(p_normalized * np.log(p_normalized), axis=1)
    k = max(2, probs.shape[1])
    return raw_entropy / np.log(k)


def calculate_simpson_diversity(probs: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Compute Simpson Diversity Index (1 - sum(p_i^2))."""
    p_sum = np.sum(probs, axis=1, keepdims=True) + eps
    p_normalized = probs / p_sum
    return 1.0 - np.sum(p_normalized ** 2, axis=1)


def engineer_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "Female (%)",
    n_splits: int = 10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Comprehensive feature engineering pipeline with strict out-of-fold isolation.
    """
    os.makedirs("./working", exist_ok=True)

    df_train = train_df.copy()
    df_test = test_df.copy()

    # Separate target
    y_train = df_train[target_col].values
    df_train = df_train.drop(columns=[target_col], errors="ignore")
    if target_col in df_test.columns:
        df_test = df_test.drop(columns=[target_col], errors="ignore")

    n_train = len(df_train)
    n_test = len(df_test)

    # Identify column types
    categorical_cols = [
        c
        for c in df_train.columns
        if df_train[c].dtype == "object" or df_train[c].dtype.name == "category"
    ]
    numeric_cols = [c for c in df_train.columns if c not in categorical_cols]

    # 1. Row-level demographic, compositional, and statistical summary features
    for df in [df_train, df_test]:
        df["num_missing_values"] = df[numeric_cols].isna().sum(axis=1)

        if len(numeric_cols) > 1:
            num_data = df[numeric_cols].values
            df["num_row_mean"] = np.nanmean(num_data, axis=1)
            df["num_row_std"] = np.nanstd(num_data, axis=1)
            df["num_row_min"] = np.nanmin(num_data, axis=1)
            df["num_row_max"] = np.nanmax(num_data, axis=1)
            df["num_row_median"] = np.nanmedian(num_data, axis=1)
            df["num_row_skew"] = pd.DataFrame(num_data).skew(axis=1).values
            df["num_row_zeros_count"] = (num_data == 0).sum(axis=1)

        # Log1p transforms for positive numerical features
        for num_col in numeric_cols:
            min_val = df[num_col].min()
            if min_val >= 0:
                df[f"{num_col}_log1p"] = np.log1p(df[num_col].fillna(0.0))

        pct_cols = [
            c
            for c in numeric_cols
            if "%" in c or "pct" in c.lower() or "percent" in c.lower()
        ]
        if len(pct_cols) >= 2:
            pct_vals = np.nan_to_num(df[pct_cols].values, nan=0.0)
            df["pct_sum_total"] = np.sum(pct_vals, axis=1)
            df["pct_residual_to_100"] = np.clip(100.0 - df["pct_sum_total"], 0.0, 100.0)
            df["pct_diversity_entropy"] = calculate_shannon_entropy(pct_vals)
            df["pct_simpson_diversity"] = calculate_simpson_diversity(pct_vals)
            df["pct_hhi_index"] = np.sum((pct_vals / 100.0) ** 2, axis=1)
            df["pct_max_dominant"] = np.max(pct_vals, axis=1)
            df["pct_min_dominant"] = np.min(pct_vals, axis=1)
            df["pct_dominant_spread"] = df["pct_max_dominant"] - df["pct_min_dominant"]
            df["pct_dominant_ratio"] = (df["pct_max_dominant"] + 1e-4) / (
                df["pct_min_dominant"] + 1e-4
            )

            # Compositional Data Analysis (CoDa): Logit transforms
            for p_col in pct_cols:
                p_bounded = np.clip(df[p_col].fillna(0.0), 0.01, 99.99)
                df[f"{p_col}_logit"] = np.log(p_bounded / (100.0 - p_bounded))

            # Compositional Data Analysis (CoDa): Centered Log-Ratio (CLR)
            clamped_pcts = np.clip(pct_vals, 0.01, 100.0)
            log_pcts = np.log(clamped_pcts)
            geom_mean_log = np.mean(log_pcts, axis=1, keepdims=True)
            clr_matrix = log_pcts - geom_mean_log
            for idx, p_col in enumerate(pct_cols):
                df[f"{p_col}_clr"] = clr_matrix[:, idx]

            # Additive Log-Ratio (ALR) with residual reference component
            ref_log = np.log(np.clip(df["pct_residual_to_100"].values, 0.01, 100.0))
            for idx, p_col in enumerate(pct_cols):
                df[f"{p_col}_alr"] = log_pcts[:, idx] - ref_log

            # Isometric Log-Ratio (ILR) transforms
            D = log_pcts.shape[1]
            for k_idx in range(1, D):
                cum_mean = np.mean(log_pcts[:, :k_idx], axis=1)
                ilr_factor = np.sqrt(k_idx / (k_idx + 1.0))
                df[f"pct_ilr_{k_idx}"] = ilr_factor * (cum_mean - log_pcts[:, k_idx])

            # Sub-compositional demographic shares & pairwise differences
            for p_col in pct_cols:
                df[f"{p_col}_share_of_total"] = (df[p_col].fillna(0.0) + 1e-5) / (
                    df["pct_sum_total"] + 1e-5
                )

            for i in range(len(pct_cols)):
                for j in range(i + 1, len(pct_cols)):
                    c1, c2 = pct_cols[i], pct_cols[j]
                    df[f"{c1}_diff_{c2}"] = df[c1].fillna(0.0) - df[c2].fillna(0.0)
                    df[f"{c1}_ratio_{c2}"] = (df[c1].fillna(0.0) + 1e-5) / (
                        df[c2].fillna(0.0) + 1e-5
                    )

    # Apply Yeo-Johnson Power Transformations to skewed numeric features
    for num_col in numeric_cols:
        col_skew = abs(df_train[num_col].dropna().skew())
        if col_skew > 0.4:
            try:
                pt = PowerTransformer(method="yeo-johnson", standardize=True)
                train_vals = df_train[[num_col]].fillna(df_train[num_col].median()).values
                test_vals = df_test[[num_col]].fillna(df_train[num_col].median()).values
                df_train[f"{num_col}_yj"] = pt.fit_transform(train_vals).flatten()
                df_test[f"{num_col}_yj"] = pt.transform(test_vals).flatten()
            except Exception:
                pass

        # Pairwise interaction features for numerical signals
        for i in range(min(len(numeric_cols), 6)):
            for j in range(i + 1, min(len(numeric_cols), 6)):
                c1, c2 = numeric_cols[i], numeric_cols[j]
                v1, v2 = df[c1].fillna(0), df[c2].fillna(0)
                df[f"{c1}_plus_{c2}"] = v1 + v2
                df[f"{c1}_minus_{c2}"] = v1 - v2
                df[f"{c1}_mult_{c2}"] = v1 * v2
                df[f"{c1}_ratio_{c2}"] = (v1 + 1e-5) / (v2 + 1e-5)

    # Latent Demographic Decomposition Components (PCA & TruncatedSVD)
    decomp_features = [
        c
        for c in numeric_cols
        if "%" in c or "pct" in c.lower() or "percent" in c.lower()
    ]
    if len(decomp_features) < 2:
        decomp_features = numeric_cols

    if len(decomp_features) >= 2:
        scaler_decomp = StandardScaler()
        train_decomp_in = scaler_decomp.fit_transform(
            df_train[decomp_features].fillna(0.0)
        )
        test_decomp_in = scaler_decomp.transform(
            df_test[decomp_features].fillna(0.0)
        )

        n_comps = min(4, len(decomp_features))
        pca = PCA(n_components=n_comps, random_state=seed)
        train_pca = pca.fit_transform(train_decomp_in)
        test_pca = pca.transform(test_decomp_in)
        for k in range(n_comps):
            df_train[f"latent_pca_{k}"] = train_pca[:, k]
            df_test[f"latent_pca_{k}"] = test_pca[:, k]

        svd = TruncatedSVD(n_components=n_comps, random_state=seed)
        train_svd = svd.fit_transform(train_decomp_in)
        test_svd = svd.transform(test_decomp_in)
        for k in range(n_comps):
            df_train[f"latent_svd_{k}"] = train_svd[:, k]
            df_test[f"latent_svd_{k}"] = test_svd[:, k]

    # 2. Categorical Frequency & Pairwise Cross-Frequency Encoding
    for col in categorical_cols:
        freq_map = df_train[col].value_counts(normalize=True).to_dict()
        df_train[f"{col}_freq"] = df_train[col].map(freq_map).fillna(0.0)
        df_test[f"{col}_freq"] = df_test[col].map(freq_map).fillna(0.0)

    for i in range(len(categorical_cols)):
        for j in range(i + 1, len(categorical_cols)):
            c1, c2 = categorical_cols[i], categorical_cols[j]
            pair_train = (
                df_train[c1].astype(str) + "_" + df_train[c2].astype(str)
            )
            pair_test = df_test[c1].astype(str) + "_" + df_test[c2].astype(str)
            pair_freq = pair_train.value_counts(normalize=True).to_dict()
            df_train[f"{c1}_{c2}_cross_freq"] = (
                pair_train.map(pair_freq).fillna(0.0)
            )
            df_test[f"{c1}_{c2}_cross_freq"] = (
                pair_test.map(pair_freq).fillna(0.0)
            )

    # 3. Create Stratified Binned Folds for local CV
    n_bins = min(10, max(2, len(y_train) // n_splits))
    target_bins = pd.qcut(y_train, q=n_bins, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = np.zeros(n_train, dtype=int)
    for fold_idx, (_, val_idx) in enumerate(skf.split(df_train, target_bins)):
        folds[val_idx] = fold_idx

    df_train["fold"] = folds

    # 4. Out-of-Fold Target Encoding (Strict Leak-Free)
    for col in categorical_cols:
        oof_te = np.zeros(n_train)

        for fold_idx in range(n_splits):
            tr_mask = folds != fold_idx
            val_mask = folds == fold_idx

            fold_tr_cats = df_train.loc[tr_mask, col]
            fold_tr_y = y_train[tr_mask]

            global_mean = np.mean(fold_tr_y)
            cat_stats = (
                pd.DataFrame({"cat": fold_tr_cats, "target": fold_tr_y})
                .groupby("cat")["target"]
                .agg(["mean", "count"])
            )

            smoothing_m = 10.0
            smooth_te = (
                cat_stats["count"] * cat_stats["mean"] + smoothing_m * global_mean
            ) / (cat_stats["count"] + smoothing_m)
            te_dict = smooth_te.to_dict()

            oof_te[val_mask] = (
                df_train.loc[val_mask, col].map(te_dict).fillna(global_mean)
            )

        global_mean_full = np.mean(y_train)
        full_cat_stats = (
            pd.DataFrame({"cat": df_train[col], "target": y_train})
            .groupby("cat")["target"]
            .agg(["mean", "count"])
        )
        full_smooth_te = (
            full_cat_stats["count"] * full_cat_stats["mean"] + 10.0 * global_mean_full
        ) / (full_cat_stats["count"] + 10.0)
        full_te_dict = full_smooth_te.to_dict()
        test_te = df_test[col].map(full_te_dict).fillna(global_mean_full)

        df_train[f"{col}_target_enc"] = oof_te
        df_test[f"{col}_target_enc"] = test_te

    # 5. Out-of-Fold Categorical Group Multi-Moment Aggregations
    top_num_cols = numeric_cols[: min(6, len(numeric_cols))]
    for cat_col in categorical_cols:
        for num_col in top_num_cols:
            mean_oof = np.zeros(n_train)
            median_oof = np.zeros(n_train)
            std_oof = np.zeros(n_train)
            min_oof = np.zeros(n_train)
            max_oof = np.zeros(n_train)

            for fold_idx in range(n_splits):
                tr_mask = folds != fold_idx
                val_mask = folds == fold_idx

                fold_df = df_train.loc[tr_mask, [cat_col, num_col]]
                stats = fold_df.groupby(cat_col)[num_col].agg(
                    ["mean", "median", "std", "min", "max"]
                )

                mean_oof[val_mask] = (
                    df_train.loc[val_mask, cat_col]
                    .map(stats["mean"].to_dict())
                    .fillna(df_train.loc[tr_mask, num_col].mean())
                )
                median_oof[val_mask] = (
                    df_train.loc[val_mask, cat_col]
                    .map(stats["median"].to_dict())
                    .fillna(df_train.loc[tr_mask, num_col].median())
                )
                std_oof[val_mask] = (
                    df_train.loc[val_mask, cat_col]
                    .map(stats["std"].to_dict())
                    .fillna(df_train.loc[tr_mask, num_col].std() or 0.0)
                )
                min_oof[val_mask] = (
                    df_train.loc[val_mask, cat_col]
                    .map(stats["min"].to_dict())
                    .fillna(df_train.loc[tr_mask, num_col].min())
                )
                max_oof[val_mask] = (
                    df_train.loc[val_mask, cat_col]
                    .map(stats["max"].to_dict())
                    .fillna(df_train.loc[tr_mask, num_col].max())
                )

            full_stats = (
                df_train[[cat_col, num_col]]
                .groupby(cat_col)[num_col]
                .agg(["mean", "median", "std", "min", "max"])
            )
            df_train[f"{cat_col}_{num_col}_grp_mean"] = mean_oof
            df_train[f"{cat_col}_{num_col}_grp_median"] = median_oof
            df_train[f"{cat_col}_{num_col}_grp_std"] = std_oof
            df_train[f"{cat_col}_{num_col}_grp_min"] = min_oof
            df_train[f"{cat_col}_{num_col}_grp_max"] = max_oof
            df_train[f"{cat_col}_{num_col}_grp_zscore"] = (
                df_train[num_col].fillna(0.0) - mean_oof
            ) / (std_oof + 1e-5)

            test_mean = (
                df_test[cat_col]
                .map(full_stats["mean"].to_dict())
                .fillna(df_train[num_col].mean())
            )
            test_std = (
                df_test[cat_col]
                .map(full_stats["std"].to_dict())
                .fillna(df_train[num_col].std() or 0.0)
            )
            df_test[f"{cat_col}_{num_col}_grp_mean"] = test_mean
            df_test[f"{cat_col}_{num_col}_grp_median"] = (
                df_test[cat_col]
                .map(full_stats["median"].to_dict())
                .fillna(df_train[num_col].median())
            )
            df_test[f"{cat_col}_{num_col}_grp_std"] = test_std
            df_test[f"{cat_col}_{num_col}_grp_min"] = (
                df_test[cat_col]
                .map(full_stats["min"].to_dict())
                .fillna(df_train[num_col].min())
            )
            df_test[f"{cat_col}_{num_col}_grp_max"] = (
                df_test[cat_col]
                .map(full_stats["max"].to_dict())
                .fillna(df_train[num_col].max())
            )
            df_test[f"{cat_col}_{num_col}_grp_zscore"] = (
                df_test[num_col].fillna(0.0) - test_mean
            ) / (test_std + 1e-5)

    # 6. Convert Categorical Columns to Category Dtype
    for col in categorical_cols:
        df_train[col] = df_train[col].astype("category")
        df_test[col] = pd.Categorical(
            df_test[col], categories=df_train[col].cat.categories
        )

    # 7. Clean and Sanitize All Column Names
    col_mapping = {
        col: sanitize_column_name(col)
        for col in df_train.columns
        if col not in ["fold"]
    }
    seen = {}
    final_mapping = {}
    for orig, clean in col_mapping.items():
        if clean in seen:
            seen[clean] += 1
            final_mapping[orig] = f"{clean}_{seen[clean]}"
        else:
            seen[clean] = 0
            final_mapping[orig] = clean

    df_train = df_train.rename(columns=final_mapping)
    df_test = df_test.rename(columns=final_mapping)

    df_train["target"] = y_train

    feature_cols = [c for c in df_train.columns if c not in ["target", "fold"]]
    categorical_feature_cols = [
        final_mapping.get(c, c)
        for c in categorical_cols
        if final_mapping.get(c, c) in feature_cols
    ]

    metadata = {
        "target_col": "target",
        "raw_target_name": target_col,
        "feature_cols": feature_cols,
        "categorical_cols": categorical_feature_cols,
        "n_splits": n_splits,
        "n_train": n_train,
        "n_test": n_test,
    }

    df_train.to_parquet("./working/train_processed.parquet", index=False)
    df_test.to_parquet("./working/test_processed.parquet", index=False)
    with open("./working/feature_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    return df_train, df_test, metadata


# ==============================================================================
# 2. Model Architectures & Loss Functions
# ==============================================================================
class TabularResidualBlock(nn.Module):
    """Residual dense block with Layer Normalization, SiLU, and Dropout."""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.block(x))


class TabularResMLP(nn.Module):
    """Multi-Head Deep Tabular Residual Network for regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [96, 64, 32],
        dropout_rate: float = 0.12,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        layers: List[nn.Module] = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.append(
                TabularResidualBlock(hidden_dim=in_dim, dropout_rate=dropout_rate)
            )
            if in_dim != out_dim:
                layers.append(
                    nn.Sequential(
                        nn.Linear(in_dim, out_dim),
                        nn.LayerNorm(out_dim),
                        nn.SiLU(),
                        nn.Dropout(dropout_rate),
                    )
                )

        layers.append(
            TabularResidualBlock(hidden_dim=hidden_dims[-1], dropout_rate=dropout_rate)
        )
        self.backbone = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dims[-1] // 2, output_dim),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.input_proj(x)
        features = self.backbone(features)
        out = self.head(features)
        return out.squeeze(-1)


class RMSELoss(nn.Module):
    """Task-faithful Root Mean Squared Error Loss."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.mse = nn.MSELoss(reduction="mean")
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(y_pred, y_true) + self.eps)


def create_optimizer_and_scheduler(
    model: nn.Module,
    learning_rate: float = 3e-3,
    weight_decay: float = 1e-4,
    epochs: int = 45,
    scheduler_type: str = "cosine",
) -> Tuple[torch.optim.Optimizer, Any]:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "LayerNorm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )

    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=learning_rate * 0.01
        )
    else:
        scheduler = None

    return optimizer, scheduler


def get_lgb_model_params(
    num_leaves: int = 31,
    learning_rate: float = 0.03,
    n_estimators: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": num_leaves,
        "max_depth": -1,
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_model_params(
    max_depth: int = 6,
    learning_rate: float = 0.03,
    n_estimators: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
    }


# ==============================================================================
# 3. Main Training, Ensembling, and Submission Pipeline
# ==============================================================================
def main():
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_raw = pd.read_parquet(train_path)
    test_raw = pd.read_parquet(test_path)

    train_df, test_df, metadata = engineer_features(
        train_raw, test_raw, target_col="Female (%)", n_splits=10, seed=SEED
    )

    target_col = metadata["target_col"]
    feature_cols = metadata["feature_cols"]
    categorical_cols = metadata["categorical_cols"]
    n_splits = metadata["n_splits"]

    n_train = len(train_df)
    n_test = len(test_df)
    y_all = train_df[target_col].values

    for col in categorical_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype("category")
            test_df[col] = pd.Categorical(
                test_df[col], categories=train_df[col].cat.categories
            )

    # Helper to convert categorical to numeric codes for bagging & neural networks
    def prepare_numeric_matrix(df_source: pd.DataFrame, cols: list) -> pd.DataFrame:
        df_num = df_source[cols].copy()
        for c in cols:
            if df_num[c].dtype.name == "category" or df_num[c].dtype == "object":
                df_num[c] = (
                    df_num[c].cat.codes
                    if hasattr(df_num[c], "cat")
                    else pd.factorize(df_num[c])[0]
                )
        return df_num.fillna(0.0)

    X_num_train = prepare_numeric_matrix(train_df, feature_cols)
    X_num_test = prepare_numeric_matrix(test_df, feature_cols)

    # --------------------------------------------------------------------------
    # Model 1: LightGBM (Deep L2 Objective with Multi-Seed Averaging)
    # --------------------------------------------------------------------------
    oof_preds_lgb_deep = np.zeros(n_train)
    test_preds_lgb_deep = np.zeros(n_test)
    seeds_lgb = [42, 2024]

    for seed in seeds_lgb:
        lgb_params_deep = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 45,
            "max_depth": 7,
            "learning_rate": 0.025,
            "n_estimators": 2500,
            "subsample": 0.85,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.5,
            "random_state": seed,
            "n_jobs": -1,
            "verbose": -1,
        }
        for fold in range(n_splits):
            tr_idx = train_df["fold"] != fold
            val_idx = train_df["fold"] == fold

            X_tr, y_tr = train_df.loc[tr_idx, feature_cols], train_df.loc[tr_idx, target_col]
            X_val, y_val = train_df.loc[val_idx, feature_cols], train_df.loc[val_idx, target_col]
            X_te = test_df[feature_cols]

            model_lgb = lgb.LGBMRegressor(**lgb_params_deep)
            model_lgb.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=60, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            oof_preds_lgb_deep[val_idx] += model_lgb.predict(X_val) / len(seeds_lgb)
            test_preds_lgb_deep += model_lgb.predict(X_te) / (n_splits * len(seeds_lgb))

    # --------------------------------------------------------------------------
    # Model 2: LightGBM (Robust Huber Objective)
    # --------------------------------------------------------------------------
    oof_preds_lgb_huber = np.zeros(n_train)
    test_preds_lgb_huber = np.zeros(n_test)
    lgb_params_huber = {
        "objective": "huber",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 25,
        "max_depth": 5,
        "learning_rate": 0.035,
        "n_estimators": 2000,
        "alpha": 0.9,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.2,
        "reg_lambda": 2.0,
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }
    for fold in range(n_splits):
        tr_idx = train_df["fold"] != fold
        val_idx = train_df["fold"] == fold

        X_tr, y_tr = train_df.loc[tr_idx, feature_cols], train_df.loc[tr_idx, target_col]
        X_val, y_val = train_df.loc[val_idx, feature_cols], train_df.loc[val_idx, target_col]
        X_te = test_df[feature_cols]

        model_lgb_h = lgb.LGBMRegressor(**lgb_params_huber)
        model_lgb_h.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=60, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        oof_preds_lgb_huber[val_idx] = model_lgb_h.predict(X_val)
        test_preds_lgb_huber += model_lgb_h.predict(X_te) / n_splits

    # --------------------------------------------------------------------------
    # Model 3: XGBoost (Shallow Tree Variant)
    # --------------------------------------------------------------------------
    oof_preds_xgb_shallow = np.zeros(n_train)
    test_preds_xgb_shallow = np.zeros(n_test)
    xgb_params_shallow = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 4,
        "learning_rate": 0.03,
        "n_estimators": 2500,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.5,
        "random_state": SEED,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    for fold in range(n_splits):
        tr_idx = train_df["fold"] != fold
        val_idx = train_df["fold"] == fold

        X_tr, y_tr = train_df.loc[tr_idx, feature_cols], train_df.loc[tr_idx, target_col]
        X_val, y_val = train_df.loc[val_idx, feature_cols], train_df.loc[val_idx, target_col]
        X_te = test_df[feature_cols]

        model_xgb_s = xgb.XGBRegressor(
            **xgb_params_shallow, early_stopping_rounds=60, enable_categorical=True
        )
        model_xgb_s.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_preds_xgb_shallow[val_idx] = model_xgb_s.predict(X_val)
        test_preds_xgb_shallow += model_xgb_s.predict(X_te) / n_splits

    # --------------------------------------------------------------------------
    # Model 4: XGBoost (Deep Tree Variant with Multi-Seed Averaging)
    # --------------------------------------------------------------------------
    oof_preds_xgb_deep = np.zeros(n_train)
    test_preds_xgb_deep = np.zeros(n_test)
    seeds_xgb = [42, 2024]

    for seed in seeds_xgb:
        xgb_params_deep = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": 7,
            "learning_rate": 0.02,
            "n_estimators": 2500,
            "subsample": 0.9,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.2,
            "reg_lambda": 2.0,
            "random_state": seed,
            "n_jobs": -1,
            "tree_method": "hist",
        }
        for fold in range(n_splits):
            tr_idx = train_df["fold"] != fold
            val_idx = train_df["fold"] == fold

            X_tr, y_tr = train_df.loc[tr_idx, feature_cols], train_df.loc[tr_idx, target_col]
            X_val, y_val = train_df.loc[val_idx, feature_cols], train_df.loc[val_idx, target_col]
            X_te = test_df[feature_cols]

            model_xgb_d = xgb.XGBRegressor(
                **xgb_params_deep, early_stopping_rounds=60, enable_categorical=True
            )
            model_xgb_d.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            oof_preds_xgb_deep[val_idx] += model_xgb_d.predict(X_val) / len(seeds_xgb)
            test_preds_xgb_deep += model_xgb_d.predict(X_te) / (n_splits * len(seeds_xgb))

    # --------------------------------------------------------------------------
    # Model 5: ExtraTrees Regressor (Bagging Tree Ensemble)
    # --------------------------------------------------------------------------
    oof_preds_et = np.zeros(n_train)
    test_preds_et = np.zeros(n_test)

    for fold in range(n_splits):
        tr_idx = train_df["fold"] != fold
        val_idx = train_df["fold"] == fold

        X_tr = X_num_train.loc[tr_idx]
        y_tr = y_all[tr_idx]
        X_val = X_num_train.loc[val_idx]
        X_te = X_num_test

        et = ExtraTreesRegressor(
            n_estimators=350,
            max_depth=16,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=SEED,
            n_jobs=-1,
        )
        et.fit(X_tr, y_tr)
        oof_preds_et[val_idx] = et.predict(X_val)
        test_preds_et += et.predict(X_te) / n_splits

    # --------------------------------------------------------------------------
    # Model 6: Random Forest Regressor (Bagging Tree Ensemble)
    # --------------------------------------------------------------------------
    oof_preds_rf = np.zeros(n_train)
    test_preds_rf = np.zeros(n_test)

    for fold in range(n_splits):
        tr_idx = train_df["fold"] != fold
        val_idx = train_df["fold"] == fold

        X_tr = X_num_train.loc[tr_idx]
        y_tr = y_all[tr_idx]
        X_val = X_num_train.loc[val_idx]
        X_te = X_num_test

        rf = RandomForestRegressor(
            n_estimators=350,
            max_depth=16,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=SEED,
            n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)
        oof_preds_rf[val_idx] = rf.predict(X_val)
        test_preds_rf += rf.predict(X_te) / n_splits

    # --------------------------------------------------------------------------
    # Model 7: CatBoost Regressor (RMSE Objective with Symmetric Splits)
    # --------------------------------------------------------------------------
    oof_preds_cb_rmse = np.zeros(n_train)
    test_preds_cb_rmse = np.zeros(n_test)

    X_cb_train = train_df[feature_cols].copy()
    X_cb_test = test_df[feature_cols].copy()
    for col in categorical_cols:
        if col in X_cb_train.columns:
            X_cb_train[col] = X_cb_train[col].astype(str).fillna("missing")
            X_cb_test[col] = X_cb_test[col].astype(str).fillna("missing")

    cb_params_rmse = {
        "iterations": 2500,
        "learning_rate": 0.03,
        "depth": 6,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": SEED,
        "verbose": 0,
    }

    for fold in range(n_splits):
        tr_idx = train_df["fold"] != fold
        val_idx = train_df["fold"] == fold

        X_tr = X_cb_train.loc[tr_idx]
        y_tr = y_all[tr_idx]
        X_val = X_cb_train.loc[val_idx]
        y_val = y_all[val_idx]
        X_te = X_cb_test

        cb_model = CatBoostRegressor(**cb_params_rmse, cat_features=categorical_cols)
        cb_model.fit(
            X_tr,
            y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=60,
            verbose=False,
        )
        oof_preds_cb_rmse[val_idx] = cb_model.predict(X_val)
        test_preds_cb_rmse += cb_model.predict(X_te) / n_splits

    # --------------------------------------------------------------------------
    # Model 8: CatBoost Regressor (Robust Huber / MAE Objective)
    # --------------------------------------------------------------------------
    oof_preds_cb_huber = np.zeros(n_train)
    test_preds_cb_huber = np.zeros(n_test)

    cb_params_huber = {
        "iterations": 2200,
        "learning_rate": 0.035,
        "depth": 5,
        "loss_function": "MAE",
        "eval_metric": "RMSE",
        "random_seed": SEED,
        "verbose": 0,
    }

    for fold in range(n_splits):
        tr_idx = train_df["fold"] != fold
        val_idx = train_df["fold"] == fold

        X_tr = X_cb_train.loc[tr_idx]
        y_tr = y_all[tr_idx]
        X_val = X_cb_train.loc[val_idx]
        y_val = y_all[val_idx]
        X_te = X_cb_test

        cb_huber_model = CatBoostRegressor(
            **cb_params_huber, cat_features=categorical_cols
        )
        cb_huber_model.fit(
            X_tr,
            y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=60,
            verbose=False,
        )
        oof_preds_cb_huber[val_idx] = cb_huber_model.predict(X_val)
        test_preds_cb_huber += cb_huber_model.predict(X_te) / n_splits

    # --------------------------------------------------------------------------
    # Model 9: Regularized Ridge Linear Baseline
    # --------------------------------------------------------------------------
    oof_preds_ridge = np.zeros(n_train)
    test_preds_ridge = np.zeros(n_test)

    for fold in range(n_splits):
        tr_idx = train_df["fold"] != fold
        val_idx = train_df["fold"] == fold

        scaler_r = StandardScaler()
        X_tr_r = scaler_r.fit_transform(X_num_train.loc[tr_idx])
        X_val_r = scaler_r.transform(X_num_train.loc[val_idx])
        X_te_r = scaler_r.transform(X_num_test)

        ridge = Ridge(alpha=10.0, random_state=SEED)
        ridge.fit(X_tr_r, y_all[tr_idx])
        oof_preds_ridge[val_idx] = ridge.predict(X_val_r)
        test_preds_ridge += ridge.predict(X_te_r) / n_splits

    # --------------------------------------------------------------------------
    # Model 10: Tabular Residual MLP (PyTorch)
    # --------------------------------------------------------------------------
    oof_preds_nn = np.zeros(n_train)
    test_preds_nn = np.zeros(n_test)

    for fold in range(n_splits):
        tr_idx = train_df["fold"] != fold
        val_idx = train_df["fold"] == fold

        scaler = StandardScaler()
        X_tr_fold = scaler.fit_transform(X_num_train.loc[tr_idx])
        X_val_fold = scaler.transform(X_num_train.loc[val_idx])
        X_test_fold = scaler.transform(X_num_test)

        y_tr_fold = y_all[tr_idx]
        y_val_fold = y_all[val_idx]

        train_dataset = TensorDataset(
            torch.tensor(X_tr_fold, dtype=torch.float32),
            torch.tensor(y_tr_fold, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_fold, dtype=torch.float32),
            torch.tensor(y_val_fold, dtype=torch.float32),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=64, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

        nn_model = TabularResMLP(
            input_dim=X_tr_fold.shape[1],
            hidden_dims=[128, 64, 32],
            dropout_rate=0.15,
        ).to(device)
        optimizer, scheduler = create_optimizer_and_scheduler(
            nn_model,
            learning_rate=3e-3,
            weight_decay=1e-4,
            epochs=50,
            scheduler_type="cosine",
        )
        criterion = RMSELoss()

        best_val_loss = float("inf")
        best_weights = None
        patience = 14
        patience_counter = 0

        for epoch in range(1, 51):
            nn_model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                preds = nn_model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            nn_model.eval()
            val_preds_list = []
            with torch.no_grad():
                for batch_x, _ in val_loader:
                    batch_x = batch_x.to(device)
                    batch_pred = nn_model(batch_x)
                    val_preds_list.append(batch_pred.cpu().numpy())

            current_val_preds = np.concatenate(val_preds_list)
            current_val_rmse = np.sqrt(
                mean_squared_error(y_val_fold, current_val_preds)
            )

            if current_val_rmse < best_val_loss:
                best_val_loss = current_val_rmse
                best_weights = {
                    k: v.cpu().clone() for k, v in nn_model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        nn_model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
        nn_model.eval()

        with torch.no_grad():
            val_preds = (
                nn_model(torch.tensor(X_val_fold, dtype=torch.float32).to(device))
                .cpu()
                .numpy()
            )
            test_preds = (
                nn_model(torch.tensor(X_test_fold, dtype=torch.float32).to(device))
                .cpu()
                .numpy()
            )

        oof_preds_nn[val_idx] = val_preds
        test_preds_nn += test_preds / n_splits

    # --------------------------------------------------------------------------
    # Diverse Multi-Model Optimization & Constrained Blending
    # --------------------------------------------------------------------------
    oof_matrix = np.column_stack(
        [
            oof_preds_lgb_deep,
            oof_preds_lgb_huber,
            oof_preds_xgb_shallow,
            oof_preds_xgb_deep,
            oof_preds_cb_rmse,
            oof_preds_cb_huber,
            oof_preds_et,
            oof_preds_rf,
            oof_preds_ridge,
            oof_preds_nn,
        ]
    )
    test_matrix = np.column_stack(
        [
            test_preds_lgb_deep,
            test_preds_lgb_huber,
            test_preds_xgb_shallow,
            test_preds_xgb_deep,
            test_preds_cb_rmse,
            test_preds_cb_huber,
            test_preds_et,
            test_preds_rf,
            test_preds_ridge,
            test_preds_nn,
        ]
    )

    n_models = oof_matrix.shape[1]

    # Non-Negative Least Squares initialization & L2 Regularized SLSQP
    nnls_weights, _ = nnls(oof_matrix, y_all)
    if np.sum(nnls_weights) > 0:
        init_weights = nnls_weights / np.sum(nnls_weights)
    else:
        init_weights = np.ones(n_models) / n_models

    def regularized_objective(weights):
        w = np.array(weights)
        w = w / (np.sum(w) + 1e-9)
        pred = np.dot(oof_matrix, w)
        rmse = np.sqrt(mean_squared_error(y_all, pred))
        l2_reg = 1e-4 * np.sum(w ** 2)
        return rmse + l2_reg

    bounds = [(0.0, 1.0) for _ in range(n_models)]
    constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}

    opt_result = minimize(
        regularized_objective,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    best_weights = opt_result.x / np.sum(opt_result.x)

    final_oof = np.dot(oof_matrix, best_weights)
    final_test_pred = np.dot(test_matrix, best_weights)

    # Strict domain boundary clipping for percentages [0.0, 100.0]
    final_oof = np.clip(final_oof, 0.0, 100.0)
    final_test_pred = np.clip(final_test_pred, 0.0, 100.0)

    final_validation_score = float(np.sqrt(mean_squared_error(y_all, final_oof)))

    # --------------------------------------------------------------------------
    # Format and Save Submission File
    # --------------------------------------------------------------------------
    os.makedirs("./submission", exist_ok=True)
    submission_path = "./submission/submission.csv"

    submission_df = pd.DataFrame(
        {"id": np.arange(len(final_test_pred)), "Female (%)": final_test_pred}
    )
    submission_df.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {final_validation_score}")


if __name__ == "__main__":
    main()
