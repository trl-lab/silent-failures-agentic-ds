import json
import math
import os
import random
import warnings
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge, RidgeCV
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings("ignore")


# ==============================================================================
# 0. Reproducibility & Environment Setup
# ==============================================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# 1. Feature Engineering & Preprocessing Pipeline
# ==============================================================================
def detect_column_types(
    df: pd.DataFrame, target_col: Optional[str] = None, id_cols: Optional[list] = None
):
    if id_cols is None:
        id_cols = []

    exclude_cols = set(id_cols + ([target_col] if target_col else []))
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    datetime_cols = []
    numeric_cols = []
    categorical_cols = []

    for col in feature_cols:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
        elif df[col].dtype == "object" or df[col].dtype.name == "category":
            sample_non_null = df[col].dropna().head(100)
            is_date = False
            col_lower = str(col).lower()
            has_date_name = any(
                k in col_lower
                for k in ["date", "time", "year", "month", "day", "dt", "period", "timestamp"]
            )
            if len(sample_non_null) > 0 and has_date_name:
                try:
                    parsed = pd.to_datetime(sample_non_null, errors="coerce")
                    if pd.api.types.is_datetime64_any_dtype(parsed.dtype) and parsed.notnull().mean() > 0.6:
                        is_date = True
                except Exception:
                    is_date = False

            if is_date:
                datetime_cols.append(col)
            else:
                categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() < 10 and not any(
                term in str(col).lower()
                for term in ["rate", "percent", "amount", "ratio", "cost", "score", "credit"]
            ):
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols, datetime_cols


def extract_datetime_features(df: pd.DataFrame, datetime_cols: list) -> pd.DataFrame:
    df_out = pd.DataFrame(index=df.index)
    for col in datetime_cols:
        try:
            dt_series = pd.to_datetime(df[col], errors="coerce")
            if not pd.api.types.is_datetime64_any_dtype(dt_series.dtype) or dt_series.notnull().sum() == 0:
                continue
            prefix = f"dt_{col}"
            df_out[f"{prefix}_year"] = dt_series.dt.year.fillna(-1).astype(np.float32)
            df_out[f"{prefix}_month"] = dt_series.dt.month.fillna(-1).astype(np.float32)
            df_out[f"{prefix}_quarter"] = dt_series.dt.quarter.fillna(-1).astype(np.float32)
            df_out[f"{prefix}_day"] = dt_series.dt.day.fillna(-1).astype(np.float32)
            df_out[f"{prefix}_dayofweek"] = dt_series.dt.dayofweek.fillna(-1).astype(
                np.float32
            )
            df_out[f"{prefix}_is_month_end"] = dt_series.dt.is_month_end.fillna(
                False
            ).astype(np.float32)
            df_out[f"{prefix}_is_quarter_end"] = dt_series.dt.is_quarter_end.fillna(
                False
            ).astype(np.float32)

            min_date = dt_series.min()
            if pd.notnull(min_date):
                df_out[f"{prefix}_days_elapsed"] = (
                    (dt_series - min_date).dt.total_seconds() / 86400.0
                ).fillna(-1).astype(np.float32)
            else:
                df_out[f"{prefix}_days_elapsed"] = np.float32(-1)
        except Exception:
            continue

    return df_out


def create_numeric_interaction_features(
    df: pd.DataFrame, numeric_cols: list
) -> pd.DataFrame:
    df_out = pd.DataFrame(index=df.index)
    eps = 1e-5

    for col in numeric_cols:
        col_val = df[col].fillna(0)
        min_val = df[col].min()
        if pd.notnull(min_val) and min_val >= 0:
            df_out[f"num_{col}_log1p"] = np.log1p(col_val.clip(lower=0)).astype(
                np.float32
            )
            df_out[f"num_{col}_sqrt"] = np.sqrt(col_val.clip(lower=0)).astype(
                np.float32
            )
        else:
            df_out[f"num_{col}_log1p"] = (
                np.sign(col_val) * np.log1p(np.abs(col_val))
            ).astype(np.float32)

    n_num = len(numeric_cols)
    for i in range(n_num):
        c1 = numeric_cols[i]
        v1 = df[c1].fillna(0).astype(np.float32)
        for j in range(i + 1, n_num):
            c2 = numeric_cols[j]
            v2 = df[c2].fillna(0).astype(np.float32)
            df_out[f"num_diff_{c1}_{c2}"] = (v1 - v2).astype(np.float32)
            df_out[f"num_ratio_{c1}_{c2}"] = (v1 / (v2.abs() + eps)).astype(np.float32)
            df_out[f"num_ratio_{c2}_{c1}"] = (v2 / (v1.abs() + eps)).astype(np.float32)
            df_out[f"num_prod_{c1}_{c2}"] = (v1 * v2).astype(np.float32)

    if len(numeric_cols) > 1:
        num_mat = df[numeric_cols].fillna(0).values.astype(np.float32)
        df_out["num_row_sum"] = np.sum(num_mat, axis=1).astype(np.float32)
        df_out["num_row_mean"] = np.mean(num_mat, axis=1).astype(np.float32)
        df_out["num_row_std"] = np.std(num_mat, axis=1).astype(np.float32)
        df_out["num_row_max"] = np.max(num_mat, axis=1).astype(np.float32)
        df_out["num_row_min"] = np.min(num_mat, axis=1).astype(np.float32)

    return df_out


def compute_group_aggregations(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cat_cols: list, num_cols: list
):
    train_agg_dict = {}
    test_agg_dict = {}

    for cat in cat_cols:
        for num in num_cols:
            grouped = train_df.groupby(cat)[num]
            mean_map = grouped.mean().to_dict()
            std_map = grouped.std().to_dict()
            min_map = grouped.min().to_dict()
            max_map = grouped.max().to_dict()
            cnt_map = grouped.count().to_dict()

            prefix = f"grp_{cat}_{num}"
            train_agg_dict[f"{prefix}_mean"] = (
                train_df[cat].map(mean_map).fillna(-1).astype(np.float32)
            )
            train_agg_dict[f"{prefix}_std"] = (
                train_df[cat].map(std_map).fillna(-1).astype(np.float32)
            )
            train_agg_dict[f"{prefix}_min"] = (
                train_df[cat].map(min_map).fillna(-1).astype(np.float32)
            )
            train_agg_dict[f"{prefix}_max"] = (
                train_df[cat].map(max_map).fillna(-1).astype(np.float32)
            )
            train_agg_dict[f"{prefix}_count"] = (
                train_df[cat].map(cnt_map).fillna(0).astype(np.float32)
            )

            test_agg_dict[f"{prefix}_mean"] = (
                test_df[cat].map(mean_map).fillna(-1).astype(np.float32)
            )
            test_agg_dict[f"{prefix}_std"] = (
                test_df[cat].map(std_map).fillna(-1).astype(np.float32)
            )
            test_agg_dict[f"{prefix}_min"] = (
                test_df[cat].map(min_map).fillna(-1).astype(np.float32)
            )
            test_agg_dict[f"{prefix}_max"] = (
                test_df[cat].map(max_map).fillna(-1).astype(np.float32)
            )
            test_agg_dict[f"{prefix}_count"] = (
                test_df[cat].map(cnt_map).fillna(0).astype(np.float32)
            )

    return pd.DataFrame(train_agg_dict, index=train_df.index), pd.DataFrame(
        test_agg_dict, index=test_df.index
    )


def compute_target_and_freq_encodings(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_cols: list,
    target_col: str,
    n_splits: int = 5,
    seed: int = 42,
):
    train_encoded = pd.DataFrame(index=train_df.index)
    test_encoded = pd.DataFrame(index=test_df.index)

    global_mean = (
        float(train_df[target_col].mean()) if target_col in train_df.columns else 0.0
    )
    smoothing = 10.0

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for cat in cat_cols:
        freq_map = train_df[cat].value_counts(normalize=True).to_dict()
        train_encoded[f"freq_{cat}"] = (
            train_df[cat].map(freq_map).fillna(0).astype(np.float32)
        )
        test_encoded[f"freq_{cat}"] = (
            test_df[cat].map(freq_map).fillna(0).astype(np.float32)
        )

        train_te = np.full(len(train_df), global_mean, dtype=np.float32)
        if "fold" in train_df.columns:
            for f in train_df["fold"].unique():
                val_mask = (train_df["fold"] == f).values
                tr_mask = ~val_mask
                fold_tr = train_df.loc[tr_mask]
                fold_stats = fold_tr.groupby(cat)[target_col].agg(["count", "mean"])
                smoothed_te = (
                    (fold_stats["count"] * fold_stats["mean"]) + (smoothing * global_mean)
                ) / (fold_stats["count"] + smoothing)
                te_dict = smoothed_te.to_dict()
                train_te[val_mask] = (
                    train_df.loc[val_mask, cat].map(te_dict).fillna(global_mean).values
                )
        else:
            for tr_idx, oof_idx in kf.split(train_df):
                fold_tr = train_df.iloc[tr_idx]
                fold_stats = fold_tr.groupby(cat)[target_col].agg(["count", "mean"])
                smoothed_te = (
                    (fold_stats["count"] * fold_stats["mean"]) + (smoothing * global_mean)
                ) / (fold_stats["count"] + smoothing)
                te_dict = smoothed_te.to_dict()
                train_te[oof_idx] = (
                    train_df.iloc[oof_idx][cat].map(te_dict).fillna(global_mean).values
                )

        train_encoded[f"te_{cat}"] = train_te

        full_stats = train_df.groupby(cat)[target_col].agg(["count", "mean"])
        full_smoothed_te = (
            (full_stats["count"] * full_stats["mean"]) + (smoothing * global_mean)
        ) / (full_stats["count"] + smoothing)
        full_te_dict = full_smoothed_te.to_dict()

        test_encoded[f"te_{cat}"] = (
            test_df[cat].map(full_te_dict).fillna(global_mean).astype(np.float32)
        )

        cat_levels = {
            val: idx
            for idx, val in enumerate(train_df[cat].astype(str).unique(), start=1)
        }
        train_encoded[f"cat_code_{cat}"] = (
            train_df[cat].astype(str).map(cat_levels).fillna(0).astype(np.int32)
        )
        test_encoded[f"cat_code_{cat}"] = (
            test_df[cat].astype(str).map(cat_levels).fillna(0).astype(np.int32)
        )

    return train_encoded, test_encoded


def preprocess_datasets():
    input_dir = "./input"
    train_df = pd.read_parquet(os.path.join(input_dir, "train.parquet"))
    test_df = pd.read_parquet(os.path.join(input_dir, "test.parquet"))

    target_col = "credit_amount"
    id_cols = [
        col
        for col in ["id", "ID", "index"]
        if col in train_df.columns or col in test_df.columns
    ]

    numeric_cols, categorical_cols, datetime_cols = detect_column_types(
        train_df, target_col=target_col, id_cols=id_cols
    )

    n_splits = 10
    seed = 42
    strat_bins = pd.qcut(train_df[target_col], q=min(10, train_df[target_col].nunique()), labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(train_df, strat_bins)):
        train_df.loc[train_df.index[val_idx], "fold"] = fold

    missing_cols = [
        col
        for col in train_df.columns
        if col not in [target_col, "fold"] + id_cols
        and (train_df[col].isnull().any() or test_df[col].isnull().any())
    ]
    for col in missing_cols:
        train_df[f"missing_{col}"] = train_df[col].isnull().astype(np.int8)
        test_df[f"missing_{col}"] = test_df[col].isnull().astype(np.int8)

    cross_cat_cols = []
    for i in range(len(categorical_cols)):
        for j in range(i + 1, min(len(categorical_cols), i + 4)):
            c1, c2 = categorical_cols[i], categorical_cols[j]
            cross_col = f"cross_{c1}_{c2}"
            train_df[cross_col] = (
                train_df[c1].astype(str) + "__" + train_df[c2].astype(str)
            )
            test_df[cross_col] = (
                test_df[c1].astype(str) + "__" + test_df[c2].astype(str)
            )
            cross_cat_cols.append(cross_col)
    all_cat_cols = categorical_cols + cross_cat_cols

    if datetime_cols:
        tr_dt = extract_datetime_features(train_df, datetime_cols)
        ts_dt = extract_datetime_features(test_df, datetime_cols)
    else:
        tr_dt = pd.DataFrame(index=train_df.index)
        ts_dt = pd.DataFrame(index=test_df.index)

    if numeric_cols:
        tr_num = create_numeric_interaction_features(train_df, numeric_cols)
        ts_num = create_numeric_interaction_features(test_df, numeric_cols)
        for col in numeric_cols:
            median_val = train_df[col].median()
            train_df[f"raw_num_{col}"] = (
                train_df[col].fillna(median_val).astype(np.float32)
            )
            test_df[f"raw_num_{col}"] = (
                test_df[col].fillna(median_val).astype(np.float32)
            )

        scaler_geo = StandardScaler()
        num_scaled_tr = scaler_geo.fit_transform(train_df[numeric_cols].fillna(0))
        num_scaled_ts = scaler_geo.transform(test_df[numeric_cols].fillna(0))

        n_pca = min(5, len(numeric_cols))
        if n_pca >= 2:
            pca = PCA(n_components=n_pca, random_state=seed)
            tr_pca = pca.fit_transform(num_scaled_tr)
            ts_pca = pca.transform(num_scaled_ts)
            for p in range(n_pca):
                train_df[f"geo_pca_{p}"] = tr_pca[:, p].astype(np.float32)
                test_df[f"geo_pca_{p}"] = ts_pca[:, p].astype(np.float32)

        n_clusters = min(6, max(2, len(train_df) // 50))
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        tr_dist = kmeans.fit_transform(num_scaled_tr)
        ts_dist = kmeans.transform(num_scaled_ts)
        for k in range(n_clusters):
            train_df[f"geo_kmeans_dist_{k}"] = tr_dist[:, k].astype(np.float32)
            test_df[f"geo_kmeans_dist_{k}"] = ts_dist[:, k].astype(np.float32)
    else:
        tr_num = pd.DataFrame(index=train_df.index)
        ts_num = pd.DataFrame(index=test_df.index)

    tr_cat_all, ts_cat_all = compute_target_and_freq_encodings(
        train_df, test_df, all_cat_cols, target_col, n_splits=n_splits, seed=seed
    )

    tr_grp_all, ts_grp_all = compute_group_aggregations(
        train_df, test_df, categorical_cols[:6], numeric_cols[:6]
    )

    raw_num_cols = [f"raw_num_{c}" for c in numeric_cols]
    missing_flag_cols = [f"missing_{c}" for c in missing_cols]
    geo_cols = [c for c in train_df.columns if c.startswith("geo_")]

    train_processed = pd.concat(
        [
            train_df[["fold", target_col] + id_cols],
            train_df[raw_num_cols],
            train_df[missing_flag_cols],
            train_df[geo_cols],
            tr_dt,
            tr_num,
            tr_cat_all,
            tr_grp_all,
        ],
        axis=1,
    )

    test_processed = pd.concat(
        [
            test_df[id_cols] if id_cols else pd.DataFrame(index=test_df.index),
            test_df[raw_num_cols],
            test_df[missing_flag_cols],
            test_df[geo_cols],
            ts_dt,
            ts_num,
            ts_cat_all,
            ts_grp_all,
        ],
        axis=1,
    )

    train_processed = train_processed.loc[:, ~train_processed.columns.duplicated()]
    test_processed = test_processed.loc[:, ~test_processed.columns.duplicated()]

    meta_cols = set(["fold", target_col] + id_cols)
    feature_cols = [c for c in train_processed.columns if c not in meta_cols]

    for col in feature_cols:
        if (
            train_processed[col].dtype == "object"
            or train_processed[col].dtype.name == "category"
        ):
            train_processed[col] = (
                pd.to_numeric(train_processed[col], errors="coerce")
                .fillna(0)
                .astype(np.float32)
            )
            test_processed[col] = (
                pd.to_numeric(test_processed[col], errors="coerce")
                .fillna(0)
                .astype(np.float32)
            )
        else:
            train_processed[col] = (
                train_processed[col]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
                .astype(np.float32)
            )
            test_processed[col] = (
                test_processed[col]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
                .astype(np.float32)
            )

    return train_processed, test_processed, feature_cols, target_col, id_cols


# ==============================================================================
# 2. Neural Architecture & Metric Definitions
# ==============================================================================
class TabularResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.activation = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        n_res_blocks: int = 2,
        dropout: float = 0.15,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.Mish(),
            nn.Dropout(dropout),
        )

        self.res_blocks = nn.ModuleList(
            [
                TabularResidualBlock(hidden_dims[0], dropout=dropout)
                for _ in range(n_res_blocks)
            ]
        )

        transition_layers = []
        in_dim = hidden_dims[0]
        for out_dim in hidden_dims[1:]:
            transition_layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.Mish(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = out_dim
        self.transition = (
            nn.Sequential(*transition_layers) if transition_layers else nn.Identity()
        )
        self.head = nn.Linear(in_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for res_block in self.res_blocks:
            h = res_block(h)
        h = self.transition(h)
        return self.head(h).squeeze(-1)


# ==============================================================================
# 3. Model Training, Ensemble Blending & Evaluation Pipeline
# ==============================================================================
def run_pipeline():
    train_df, test_df, feature_cols, target_col, id_cols = preprocess_datasets()
    n_folds = int(train_df["fold"].nunique())
    n_samples = len(train_df)
    n_test = len(test_df)

    oof_preds_lgb = np.zeros(n_samples, dtype=np.float32)
    test_preds_lgb = np.zeros(n_test, dtype=np.float32)

    oof_preds_lgb_log = np.zeros(n_samples, dtype=np.float32)
    test_preds_lgb_log = np.zeros(n_test, dtype=np.float32)

    oof_preds_xgb = np.zeros(n_samples, dtype=np.float32)
    test_preds_xgb = np.zeros(n_test, dtype=np.float32)

    oof_preds_cat = np.zeros(n_samples, dtype=np.float32)
    test_preds_cat = np.zeros(n_test, dtype=np.float32)

    oof_preds_hgbr = np.zeros(n_samples, dtype=np.float32)
    test_preds_hgbr = np.zeros(n_test, dtype=np.float32)

    oof_preds_et = np.zeros(n_samples, dtype=np.float32)
    test_preds_et = np.zeros(n_test, dtype=np.float32)

    oof_preds_rf = np.zeros(n_samples, dtype=np.float32)
    test_preds_rf = np.zeros(n_test, dtype=np.float32)

    oof_preds_ridge = np.zeros(n_samples, dtype=np.float32)
    test_preds_ridge = np.zeros(n_test, dtype=np.float32)

    oof_preds_nn = np.zeros(n_samples, dtype=np.float32)
    test_preds_nn = np.zeros(n_test, dtype=np.float32)

    y_true = train_df[target_col].values.astype(np.float32)
    X_test_raw = test_df[feature_cols].values.astype(np.float32)

    for fold in range(n_folds):
        train_idx = (train_df["fold"] != fold).values
        val_idx = (train_df["fold"] == fold).values

        X_train = train_df.loc[train_idx, feature_cols].values.astype(np.float32)
        y_train = train_df.loc[train_idx, target_col].values.astype(np.float32)
        X_val = train_df.loc[val_idx, feature_cols].values.astype(np.float32)
        y_val = train_df.loc[val_idx, target_col].values.astype(np.float32)

        # ----------------------------------------------------------------------
        # Model 1: LightGBM Regressor
        # ----------------------------------------------------------------------
        lgb_params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.85,
            "subsample_freq": 1,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42 + fold,
            "n_jobs": -1,
            "verbose": -1,
        }

        trn_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=trn_data)

        lgb_model = lgb.train(
            lgb_params,
            trn_data,
            num_boost_round=1500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        val_pred_lgb = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
        oof_preds_lgb[val_idx] = val_pred_lgb
        test_preds_lgb += (
            lgb_model.predict(X_test_raw, num_iteration=lgb_model.best_iteration)
            / n_folds
        )

        # ----------------------------------------------------------------------
        # Model 1b: LightGBM Log-Transformed Regressor
        # ----------------------------------------------------------------------
        y_train_log = np.log1p(np.clip(y_train, 0, None))
        y_val_log = np.log1p(np.clip(y_val, 0, None))

        trn_data_log = lgb.Dataset(X_train, label=y_train_log)
        val_data_log = lgb.Dataset(X_val, label=y_val_log, reference=trn_data_log)

        lgb_log_model = lgb.train(
            lgb_params,
            trn_data_log,
            num_boost_round=1500,
            valid_sets=[val_data_log],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        val_pred_lgb_log = np.expm1(
            lgb_log_model.predict(X_val, num_iteration=lgb_log_model.best_iteration)
        )
        oof_preds_lgb_log[val_idx] = val_pred_lgb_log
        test_preds_lgb_log += (
            np.expm1(
                lgb_log_model.predict(
                    X_test_raw, num_iteration=lgb_log_model.best_iteration
                )
            )
            / n_folds
        )

        # ----------------------------------------------------------------------
        # Model 2: XGBoost Regressor
        # ----------------------------------------------------------------------
        xgb_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_estimators=1500,
            early_stopping_rounds=50,
            random_state=42 + fold,
            n_jobs=-1,
            tree_method="hist",
        )

        xgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_pred_xgb = xgb_model.predict(X_val)
        oof_preds_xgb[val_idx] = val_pred_xgb
        test_preds_xgb += xgb_model.predict(X_test_raw) / n_folds

        # ----------------------------------------------------------------------
        # Model 3: CatBoost Regressor
        # ----------------------------------------------------------------------
        if HAS_CATBOOST:
            cb_model = CatBoostRegressor(
                loss_function="RMSE",
                eval_metric="RMSE",
                learning_rate=0.03,
                depth=6,
                l2_leaf_reg=3.0,
                iterations=1500,
                early_stopping_rounds=50,
                random_seed=42 + fold,
                verbose=False,
                thread_count=-1,
            )
            cb_model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                verbose=False,
            )
            val_pred_cat = cb_model.predict(X_val)
            oof_preds_cat[val_idx] = val_pred_cat
            test_preds_cat += cb_model.predict(X_test_raw) / n_folds
        else:
            oof_preds_cat[val_idx] = val_pred_lgb
            test_preds_cat += (
                lgb_model.predict(X_test_raw, num_iteration=lgb_model.best_iteration)
                / n_folds
            )

        # ----------------------------------------------------------------------
        # Model 3b: HistGradientBoosting Regressor
        # ----------------------------------------------------------------------
        hgbr_model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.03,
            max_iter=500,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=True,
            n_iter_no_change=25,
            random_state=42 + fold,
        )
        hgbr_model.fit(X_train, y_train)
        val_pred_hgbr = hgbr_model.predict(X_val)
        oof_preds_hgbr[val_idx] = val_pred_hgbr
        test_preds_hgbr += hgbr_model.predict(X_test_raw) / n_folds

        # ----------------------------------------------------------------------
        # Model 4: ExtraTrees Regressor
        # ----------------------------------------------------------------------
        et_model = ExtraTreesRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42 + fold,
            n_jobs=-1,
        )
        et_model.fit(X_train, y_train)
        val_pred_et = et_model.predict(X_val)
        oof_preds_et[val_idx] = val_pred_et
        test_preds_et += et_model.predict(X_test_raw) / n_folds

        # ----------------------------------------------------------------------
        # Model 5: RandomForest Regressor
        # ----------------------------------------------------------------------
        rf_model = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42 + fold,
            n_jobs=-1,
        )
        rf_model.fit(X_train, y_train)
        val_pred_rf = rf_model.predict(X_val)
        oof_preds_rf[val_idx] = val_pred_rf
        test_preds_rf += rf_model.predict(X_test_raw) / n_folds

        # ----------------------------------------------------------------------
        # Model 6: RidgeCV Regressor (with Robust Scaling)
        # ----------------------------------------------------------------------
        scaler_ridge = RobustScaler()
        X_train_r = scaler_ridge.fit_transform(X_train)
        X_val_r = scaler_ridge.transform(X_val)
        X_test_r = scaler_ridge.transform(X_test_raw)

        ridge_model = RidgeCV(alphas=np.logspace(-3, 3, 20))
        ridge_model.fit(X_train_r, y_train)
        val_pred_ridge = ridge_model.predict(X_val_r)
        oof_preds_ridge[val_idx] = val_pred_ridge
        test_preds_ridge += ridge_model.predict(X_test_r) / n_folds

        # ----------------------------------------------------------------------
        # Model 7: Tabular Residual Neural Network (PyTorch)
        # ----------------------------------------------------------------------
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
        X_val_scaled = scaler.transform(X_val).astype(np.float32)
        X_test_scaled = scaler.transform(X_test_raw).astype(np.float32)

        train_dataset = TensorDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_scaled, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=128, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        nn_model = TabularResNet(
            input_dim=len(feature_cols),
            hidden_dims=(256, 128, 64),
            dropout=0.15,
        ).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=40, eta_min=1e-6
        )

        best_val_loss = float("inf")
        best_nn_preds = None
        best_state = None
        patience = 10
        patience_counter = 0

        for epoch in range(40):
            nn_model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                preds = nn_model(bx)
                loss = criterion(preds, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(nn_model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            nn_model.eval()
            val_preds_list = []
            with torch.no_grad():
                for bx, _ in val_loader:
                    bx = bx.to(device)
                    val_preds_list.append(nn_model(bx).cpu().numpy())
            current_val_preds = np.concatenate(val_preds_list)
            val_rmse = root_mean_squared_error(
                y_val, np.clip(current_val_preds, 0, None)
            )

            if val_rmse < best_val_loss:
                best_val_loss = val_rmse
                best_nn_preds = current_val_preds
                best_state = {
                    k: v.cpu().clone() for k, v in nn_model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        oof_preds_nn[val_idx] = best_nn_preds

        if best_state is not None:
            nn_model.load_state_dict(best_state)
        nn_model.eval()
        with torch.no_grad():
            t_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
            nn_test_pred = nn_model(t_tensor).cpu().numpy()
            test_preds_nn += nn_test_pred / n_folds

    # ==============================================================================
    # 4. Out-of-Fold Blending Optimization
    # ==============================================================================
    models_oof = np.column_stack(
        [
            oof_preds_lgb,
            oof_preds_lgb_log,
            oof_preds_xgb,
            oof_preds_cat,
            oof_preds_hgbr,
            oof_preds_et,
            oof_preds_rf,
            oof_preds_ridge,
            oof_preds_nn,
        ]
    )
    models_test = np.column_stack(
        [
            test_preds_lgb,
            test_preds_lgb_log,
            test_preds_xgb,
            test_preds_cat,
            test_preds_hgbr,
            test_preds_et,
            test_preds_rf,
            test_preds_ridge,
            test_preds_nn,
        ]
    )
    n_models = models_oof.shape[1]

    def blend_objective(weights):
        w = np.array(weights)
        w = w / (np.sum(w) + 1e-12)
        pred = np.dot(models_oof, w)
        rmse = root_mean_squared_error(y_true, np.clip(pred, 0, None))
        l2_reg = 1e-4 * np.sum(w**2)
        return rmse + l2_reg

    initial_weights = [1.0 / n_models] * n_models
    bounds = [(0, 1)] * n_models
    constraints = {"type": "eq", "fun": lambda w: 1.0 - sum(w)}
    res = minimize(
        blend_objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )
    opt_weights = np.maximum(0, res.x)
    opt_weights = opt_weights / np.sum(opt_weights)

    oof_final = np.dot(models_oof, opt_weights)
    oof_final_clipped = np.clip(oof_final, 0, None)
    final_validation_score = float(root_mean_squared_error(y_true, oof_final_clipped))

    test_final = np.dot(models_test, opt_weights)
    test_final_clipped = np.clip(test_final, 0, None)

    # ==============================================================================
    # 5. Generate Submission File
    # ==============================================================================
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    test_ids = (
        test_df["id"].values if "id" in test_df.columns else np.arange(len(test_df))
    )
    sub = pd.DataFrame({"id": test_ids, "credit_amount": test_final_clipped})
    sub["id"] = sub["id"].astype(int)
    sub.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {final_validation_score}")


if __name__ == "__main__":
    run_pipeline()
