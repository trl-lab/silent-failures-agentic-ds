import copy
import os
import re
import warnings
from typing import Any, Dict, Optional, Tuple

from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")

# =========================================================================
# 1. Configuration and Data Loading
# =========================================================================
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

train_path = os.path.join(INPUT_DIR, "train.parquet")
test_path = os.path.join(INPUT_DIR, "test.parquet")
submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")

print("Loading raw datasets...")
df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

target_col = "New Commitments (US$)"
assert (
    target_col in df_train.columns
), f"Target column '{target_col}' not found in training data!"

print(f"Raw Train Shape: {df_train.shape}, Raw Test Shape: {df_test.shape}")

y_train = df_train[target_col].copy()
X_train_raw = df_train.drop(columns=[target_col]).copy()
X_test_raw = df_test.copy()

X_train_raw.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
X_test_raw.reset_index(drop=True, inplace=True)

# =========================================================================
# 2. Column Identification & Feature Engineering
# =========================================================================
datetime_cols = []
numeric_cols = []
categorical_cols = []
text_cols = []

for col in X_train_raw.columns:
    if pd.api.types.is_datetime64_any_dtype(X_train_raw[col]):
        datetime_cols.append(col)
    elif X_train_raw[col].dtype == object or isinstance(
        X_train_raw[col].dtype, pd.CategoricalDtype
    ):
        sample = X_train_raw[col].dropna().astype(str).head(50)
        is_date = False
        if len(sample) > 0:
            try:
                pd.to_datetime(sample, format="mixed", errors="raise")
                is_date = True
            except Exception:
                is_date = False

        if is_date:
            datetime_cols.append(col)
        else:
            unique_cnt = X_train_raw[col].nunique()
            avg_len = (
                X_train_raw[col].dropna().astype(str).str.len().mean()
                if unique_cnt > 0
                else 0
            )
            if avg_len > 100 or unique_cnt > 0.8 * len(X_train_raw):
                text_cols.append(col)
            else:
                categorical_cols.append(col)
    elif pd.api.types.is_numeric_dtype(X_train_raw[col]):
        if X_train_raw[col].nunique() <= 10 and not pd.api.types.is_float_dtype(
            X_train_raw[col]
        ):
            categorical_cols.append(col)
        numeric_cols.append(col)
    else:
        categorical_cols.append(col)

print(
    f"Identified Columns - Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}, "
    f"Datetime: {len(datetime_cols)}, Text: {len(text_cols)}"
)


def extract_base_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)

    # Datetime decomposition & cyclical harmonics
    for col in datetime_cols:
        dt_series = pd.to_datetime(df[col], errors="coerce")
        feats[f"{col}_year"] = dt_series.dt.year.fillna(-1).astype(np.float32)
        feats[f"{col}_month"] = dt_series.dt.month.fillna(-1).astype(np.float32)
        feats[f"{col}_quarter"] = dt_series.dt.quarter.fillna(-1).astype(np.float32)
        feats[f"{col}_day"] = dt_series.dt.day.fillna(-1).astype(np.float32)
        feats[f"{col}_dayofweek"] = dt_series.dt.dayofweek.fillna(-1).astype(np.float32)
        feats[f"{col}_dayofyear"] = dt_series.dt.dayofyear.fillna(-1).astype(np.float32)
        feats[f"{col}_is_month_end"] = dt_series.dt.is_month_end.astype(float).fillna(0)
        feats[f"{col}_is_quarter_end"] = dt_series.dt.is_quarter_end.astype(
            float
        ).fillna(0)

        month_rad = 2 * np.pi * dt_series.dt.month / 12.0
        feats[f"{col}_month_sin"] = np.sin(month_rad).fillna(0).astype(np.float32)
        feats[f"{col}_month_cos"] = np.cos(month_rad).fillna(0).astype(np.float32)

        ref_date = pd.Timestamp("2000-01-01")
        feats[f"{col}_days_from_ref"] = (
            (dt_series - ref_date).dt.days.fillna(-999).astype(np.float32)
        )

    # Datetime pairwise difference
    if len(datetime_cols) >= 2:
        for i in range(len(datetime_cols)):
            for j in range(i + 1, len(datetime_cols)):
                col_a = datetime_cols[i]
                col_b = datetime_cols[j]
                dt_a = pd.to_datetime(df[col_a], errors="coerce")
                dt_b = pd.to_datetime(df[col_b], errors="coerce")
                feats[f"diff_days_{col_a}_{col_b}"] = (
                    (dt_b - dt_a).dt.days.fillna(-9999).astype(np.float32)
                )

    # Numeric transformations and missingness indicators
    for col in numeric_cols:
        col_vals = pd.to_numeric(df[col], errors="coerce")
        feats[f"{col}_isna"] = col_vals.isna().astype(np.float32)
        feats[col] = col_vals.astype(np.float32)
        feats[f"{col}_signed_log"] = np.sign(col_vals) * np.log1p(np.abs(col_vals))

    # Basic text statistics
    for col in text_cols:
        str_series = df[col].fillna("").astype(str)
        feats[f"{col}_char_len"] = str_series.str.len().astype(np.float32)
        feats[f"{col}_word_count"] = str_series.apply(lambda s: len(s.split())).astype(
            np.float32
        )
        feats[f"{col}_digit_count"] = str_series.apply(
            lambda s: sum(c.isdigit() for c in s)
        ).astype(np.float32)
        feats[f"{col}_uppercase_count"] = str_series.apply(
            lambda s: sum(c.isupper() for c in s)
        ).astype(np.float32)

    # Categorical string normalization
    for col in categorical_cols:
        feats[f"cat_{col}"] = (
            df[col].fillna("MISSING").astype(str).str.strip().str.lower()
        )

    return feats


print("Extracting base features...")
X_train_base = extract_base_features(X_train_raw)
X_test_base = extract_base_features(X_test_raw)

# Frequency encodings & interactions
clean_cat_cols = [
    f"cat_{col}" for col in categorical_cols if f"cat_{col}" in X_train_base.columns
]

for ccol in clean_cat_cols:
    freq_map = X_train_base[ccol].value_counts(normalize=True).to_dict()
    X_train_base[f"{ccol}_freq"] = (
        X_train_base[ccol].map(freq_map).fillna(0).astype(np.float32)
    )
    X_test_base[f"{ccol}_freq"] = (
        X_test_base[ccol].map(freq_map).fillna(0).astype(np.float32)
    )

if len(clean_cat_cols) >= 2:
    for i in range(min(4, len(clean_cat_cols))):
        for j in range(i + 1, min(4, len(clean_cat_cols))):
            col_1 = clean_cat_cols[i]
            col_2 = clean_cat_cols[j]
            comb_train = X_train_base[col_1] + "__" + X_train_base[col_2]
            comb_test = X_test_base[col_1] + "__" + X_test_base[col_2]
            freq_comb = comb_train.value_counts(normalize=True).to_dict()
            X_train_base[f"inter_{col_1}_{col_2}_freq"] = (
                comb_train.map(freq_comb).fillna(0).astype(np.float32)
            )
            X_test_base[f"inter_{col_1}_{col_2}_freq"] = (
                comb_test.map(freq_comb).fillna(0).astype(np.float32)
            )

# Out-Of-Fold Cross-Validation splits
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
fold_indices = np.zeros(len(df_train), dtype=int)
for fold, (trn_idx, val_idx) in enumerate(kf.split(df_train)):
    fold_indices[val_idx] = fold
X_train_base["fold"] = fold_indices

# TF-IDF + TruncatedSVD and OOF Ridge Regressors on text columns
for tcol in text_cols:
    train_text = X_train_raw[tcol].fillna("").astype(str)
    test_text = X_test_raw[tcol].fillna("").astype(str)

    try:
        tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            token_pattern=r"(?u)\b\w+\b",
            sublinear_tf=True,
        )
        X_tr_tfidf = tfidf.fit_transform(train_text)
        X_te_tfidf = tfidf.transform(test_text)
        n_vocab = X_tr_tfidf.shape[1]
    except Exception:
        n_vocab = 0

    if n_vocab > 0:
        n_comp = min(12, max(1, n_vocab - 1)) if n_vocab > 1 else 1
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        tr_svd = svd.fit_transform(X_tr_tfidf)
        te_svd = svd.transform(X_te_tfidf)

        for i in range(n_comp):
            X_train_base[f"{tcol}_svd_{i}"] = tr_svd[:, i].astype(np.float32)
            X_test_base[f"{tcol}_svd_{i}"] = te_svd[:, i].astype(np.float32)

        ridge_oof = np.zeros(len(df_train), dtype=np.float32)
        ridge_test = np.zeros(len(df_test), dtype=np.float32)
        for fold in range(N_SPLITS):
            trn_idx = np.where(fold_indices != fold)[0]
            val_idx = np.where(fold_indices == fold)[0]
            ridge = Ridge(alpha=10.0, random_state=42)
            ridge.fit(X_tr_tfidf[trn_idx], y_train.iloc[trn_idx])
            ridge_oof[val_idx] = ridge.predict(X_tr_tfidf[val_idx]).astype(np.float32)
            ridge_test += (
                ridge.predict(X_te_tfidf).astype(np.float32) / N_SPLITS
            )
        X_train_base[f"{tcol}_ridge_pred"] = ridge_oof
        X_test_base[f"{tcol}_ridge_pred"] = ridge_test

# Categorical group aggregation statistics
for ccol in clean_cat_cols:
    for ncol in numeric_cols:
        if ncol in X_train_base.columns:
            group_stats = (
                X_train_base.groupby(ccol)[ncol]
                .agg(["mean", "std", "min", "max"])
                .reset_index()
            )
            group_stats.columns = [
                ccol,
                f"{ccol}_{ncol}_grp_mean",
                f"{ccol}_{ncol}_grp_std",
                f"{ccol}_{ncol}_grp_min",
                f"{ccol}_{ncol}_grp_max",
            ]
            X_train_base = X_train_base.merge(group_stats, on=ccol, how="left")
            X_test_base = X_test_base.merge(group_stats, on=ccol, how="left")

# Multi-level out-of-fold smoothed target encodings
target_encoded_cols = clean_cat_cols
global_target_mean = float(y_train.mean())
smooth_weight = 10.0

for ccol in target_encoded_cols:
    oof_target_enc = np.full(len(df_train), global_target_mean, dtype=np.float32)
    for fold in range(N_SPLITS):
        trn_idx = np.where(fold_indices != fold)[0]
        val_idx = np.where(fold_indices == fold)[0]

        cats_trn = X_train_base.iloc[trn_idx][ccol]
        y_trn = y_train.iloc[trn_idx]
        stats = y_trn.groupby(cats_trn).agg(["count", "mean"])
        smoothed = (
            stats["count"] * stats["mean"] + smooth_weight * global_target_mean
        ) / (stats["count"] + smooth_weight)
        oof_target_enc[val_idx] = (
            X_train_base.iloc[val_idx][ccol]
            .map(smoothed.to_dict())
            .fillna(global_target_mean)
            .values
        )

    X_train_base[f"{ccol}_target_enc"] = oof_target_enc

    stats_full = y_train.groupby(X_train_base[ccol]).agg(["count", "mean"])
    smoothed_full = (
        stats_full["count"] * stats_full["mean"] + smooth_weight * global_target_mean
    ) / (stats_full["count"] + smooth_weight)
    X_test_base[f"{ccol}_target_enc"] = (
        X_test_base[ccol]
        .map(smoothed_full.to_dict())
        .fillna(global_target_mean)
        .values.astype(np.float32)
    )

if len(clean_cat_cols) >= 2:
    for i in range(min(3, len(clean_cat_cols))):
        for j in range(i + 1, min(3, len(clean_cat_cols))):
            col1 = clean_cat_cols[i]
            col2 = clean_cat_cols[j]
            pair_name = f"{col1}__{col2}"
            pair_trn = (
                X_train_base[col1].astype(str) + "__" + X_train_base[col2].astype(str)
            )
            pair_tst = (
                X_test_base[col1].astype(str) + "__" + X_test_base[col2].astype(str)
            )

            oof_target_enc = np.full(
                len(df_train), global_target_mean, dtype=np.float32
            )
            for fold in range(N_SPLITS):
                trn_idx = np.where(fold_indices != fold)[0]
                val_idx = np.where(fold_indices == fold)[0]
                stats = (
                    y_train.iloc[trn_idx]
                    .groupby(pair_trn.iloc[trn_idx])
                    .agg(["count", "mean"])
                )
                smoothed = (
                    stats["count"] * stats["mean"] + smooth_weight * global_target_mean
                ) / (stats["count"] + smooth_weight)
                oof_target_enc[val_idx] = (
                    pair_trn.iloc[val_idx]
                    .map(smoothed.to_dict())
                    .fillna(global_target_mean)
                    .values
                )
            X_train_base[f"{pair_name}_target_enc"] = oof_target_enc

            stats_full = y_train.groupby(pair_trn).agg(["count", "mean"])
            smoothed_full = (
                stats_full["count"] * stats_full["mean"]
                + smooth_weight * global_target_mean
            ) / (stats_full["count"] + smooth_weight)
            X_test_base[f"{pair_name}_target_enc"] = (
                pair_tst.map(smoothed_full.to_dict())
                .fillna(global_target_mean)
                .values.astype(np.float32)
            )

# Label encode categories to integer codes
for ccol in clean_cat_cols:
    all_unique_cats = sorted(
        list(set(X_train_base[ccol].unique()).union(set(X_test_base[ccol].unique())))
    )
    cat_to_code = {cat: idx for idx, cat in enumerate(all_unique_cats)}
    X_train_base[f"{ccol}_code"] = (
        X_train_base[ccol].map(cat_to_code).fillna(-1).astype(np.int32)
    )
    X_test_base[f"{ccol}_code"] = (
        X_test_base[ccol].map(cat_to_code).fillna(-1).astype(np.int32)
    )
    X_train_base.drop(columns=[ccol], inplace=True)
    X_test_base.drop(columns=[ccol], inplace=True)

# Sanitize column names and impute medians
X_train_base.columns = [re.sub(r"[^\w_]", "_", col) for col in X_train_base.columns]
X_test_base.columns = [re.sub(r"[^\w_]", "_", col) for col in X_test_base.columns]

feature_columns = [col for col in X_train_base.columns if col != "fold"]

for col in feature_columns:
    median_val = X_train_base[col].replace([np.inf, -np.inf], np.nan).median()
    if pd.isna(median_val):
        median_val = 0.0
    X_train_base[col] = (
        X_train_base[col]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(median_val)
        .astype(np.float32)
    )
    X_test_base[col] = (
        X_test_base[col]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(median_val)
        .astype(np.float32)
    )

print(
    f"Feature Engineering complete. Features: {len(feature_columns)}, Train: {X_train_base.shape}, Test: {X_test_base.shape}"
)


# =========================================================================
# 3. Model Architecture Specifications
# =========================================================================
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.15):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
        output_dim: int = 1,
    ):
        super(TabularResNet, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim, output_dim),
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
        return self.head(h)


def get_lgb_params(seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "min_child_samples": 20,
        "verbosity": -1,
        "n_jobs": -1,
        "random_state": seed,
    }


def get_xgb_params(seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_weight": 3,
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": seed,
    }


def get_catboost_params(seed: int = 42) -> Dict[str, Any]:
    return {
        "iterations": 2500,
        "learning_rate": 0.04,
        "depth": 6,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": seed,
        "verbose": False,
        "early_stopping_rounds": 60,
    }


def build_pytorch_model(
    input_dim: int,
    hidden_dim: int = 256,
    num_blocks: int = 3,
    dropout_rate: float = 0.15,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, nn.Module, optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularResNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout_rate=dropout_rate,
        output_dim=1,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-6)
    return model, criterion, optimizer, scheduler


def calc_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# =========================================================================
# 4. Model Training & Validation
# =========================================================================
fold_series = X_train_base["fold"].values
X_tr_full = X_train_base[feature_columns].values
X_te_full = X_test_base[feature_columns].values
y_tr_full = y_train.values

n_samples = len(X_tr_full)
n_test = len(X_te_full)
n_features = len(feature_columns)
unique_folds = sorted(list(np.unique(fold_series)))

oof_lgb = np.zeros(n_samples, dtype=np.float64)
test_lgb = np.zeros(n_test, dtype=np.float64)

oof_xgb = np.zeros(n_samples, dtype=np.float64)
test_xgb = np.zeros(n_test, dtype=np.float64)

oof_cat = np.zeros(n_samples, dtype=np.float64)
test_cat = np.zeros(n_test, dtype=np.float64)

oof_nn = np.zeros(n_samples, dtype=np.float64)
test_nn = np.zeros(n_test, dtype=np.float64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model 1: LightGBM ---
print("\n>>> Training LightGBM...")
lgb_params = get_lgb_params(seed=42)

for fold in unique_folds:
    trn_mask = fold_series != fold
    val_mask = fold_series == fold

    X_trn, y_trn = X_tr_full[trn_mask], y_tr_full[trn_mask]
    X_val, y_val = X_tr_full[val_mask], y_tr_full[val_mask]

    trn_data = lgb.Dataset(
        X_trn, label=y_trn, feature_name=feature_columns, free_raw_data=False
    )
    val_data = lgb.Dataset(
        X_val,
        label=y_val,
        feature_name=feature_columns,
        reference=trn_data,
        free_raw_data=False,
    )

    model_lgb = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=2500,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=60, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_preds = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)
    oof_lgb[val_mask] = val_preds
    test_lgb += (
        model_lgb.predict(X_te_full, num_iteration=model_lgb.best_iteration) / N_SPLITS
    )
    print(
        f"[LightGBM] Fold {fold} | Best Iter: {model_lgb.best_iteration} | Val RMSE: {calc_rmse(y_val, val_preds):.4f}"
    )

print(f"--> [LightGBM] Out-Of-Fold RMSE: {calc_rmse(y_tr_full, oof_lgb):.4f}")

# --- Model 2: XGBoost ---
print("\n>>> Training XGBoost...")
xgb_params = get_xgb_params(seed=42)
dtest = xgb.DMatrix(X_te_full, feature_names=feature_columns)

for fold in unique_folds:
    trn_mask = fold_series != fold
    val_mask = fold_series == fold

    X_trn, y_trn = X_tr_full[trn_mask], y_tr_full[trn_mask]
    X_val, y_val = X_tr_full[val_mask], y_tr_full[val_mask]

    dtrain = xgb.DMatrix(X_trn, label=y_trn, feature_names=feature_columns)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_columns)

    model_xgb = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=2500,
        evals=[(dval, "val")],
        early_stopping_rounds=60,
        verbose_eval=False,
    )

    val_preds = model_xgb.predict(dval)
    oof_xgb[val_mask] = val_preds
    test_xgb += model_xgb.predict(dtest) / N_SPLITS
    print(
        f"[XGBoost] Fold {fold} | Best Iter: {model_xgb.best_iteration} | Val RMSE: {calc_rmse(y_val, val_preds):.4f}"
    )

print(f"--> [XGBoost] Out-Of-Fold RMSE: {calc_rmse(y_tr_full, oof_xgb):.4f}")

# --- Model 3: CatBoost ---
print("\n>>> Training CatBoost...")
cat_params = get_catboost_params(seed=42)

for fold in unique_folds:
    trn_mask = fold_series != fold
    val_mask = fold_series == fold

    X_trn, y_trn = X_tr_full[trn_mask], y_tr_full[trn_mask]
    X_val, y_val = X_tr_full[val_mask], y_tr_full[val_mask]

    model_cat = CatBoostRegressor(**cat_params)
    model_cat.fit(
        X_trn,
        y_trn,
        eval_set=(X_val, y_val),
        use_best_model=True,
        verbose=False,
    )

    val_preds = model_cat.predict(X_val)
    oof_cat[val_mask] = val_preds
    test_cat += model_cat.predict(X_te_full) / N_SPLITS
    print(
        f"[CatBoost] Fold {fold} | Best Iter: {model_cat.get_best_iteration()} | Val RMSE: {calc_rmse(y_val, val_preds):.4f}"
    )

print(f"--> [CatBoost] Out-Of-Fold RMSE: {calc_rmse(y_tr_full, oof_cat):.4f}")

# --- Model 4: PyTorch Tabular ResNet ---
print("\n>>> Training PyTorch Tabular ResNet...")
NN_EPOCHS = 35
NN_BATCH_SIZE = 256
NN_PATIENCE = 8

for fold in unique_folds:
    trn_mask = fold_series != fold
    val_mask = fold_series == fold

    X_trn, y_trn = X_tr_full[trn_mask], y_tr_full[trn_mask]
    X_val, y_val = X_tr_full[val_mask], y_tr_full[val_mask]

    scaler = StandardScaler()
    X_trn_s = scaler.fit_transform(X_trn)
    X_val_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_te_full)

    train_dataset = TensorDataset(
        torch.tensor(X_trn_s, dtype=torch.float32),
        torch.tensor(y_trn, dtype=torch.float32).unsqueeze(1),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_s, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1),
    )
    test_dataset = TensorDataset(torch.tensor(X_te_s, dtype=torch.float32))

    train_loader = DataLoader(
        train_dataset, batch_size=NN_BATCH_SIZE, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=NN_BATCH_SIZE * 2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=NN_BATCH_SIZE * 2, shuffle=False)

    model, criterion, optimizer, scheduler = build_pytorch_model(
        input_dim=n_features,
        hidden_dim=256,
        num_blocks=3,
        dropout_rate=0.15,
        device=device,
    )

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    for epoch in range(1, NN_EPOCHS + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                preds = model(batch_x)
                val_preds_list.append(preds.cpu().numpy())

        current_val_preds = np.vstack(val_preds_list).flatten()
        val_rmse = calc_rmse(y_val, current_val_preds)

        if val_rmse < best_val_loss:
            best_val_loss = val_rmse
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= NN_PATIENCE:
                break

    model.load_state_dict(best_weights)
    model.eval()

    val_preds_list = []
    with torch.no_grad():
        for batch_x, _ in val_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x)
            val_preds_list.append(preds.cpu().numpy())
    fold_val_preds = np.vstack(val_preds_list).flatten()
    oof_nn[val_mask] = fold_val_preds

    test_preds_list = []
    with torch.no_grad():
        for (batch_x,) in test_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x)
            test_preds_list.append(preds.cpu().numpy())
    fold_test_preds = np.vstack(test_preds_list).flatten()
    test_nn += fold_test_preds / N_SPLITS

    print(
        f"[TabularResNet] Fold {fold} | Val RMSE: {calc_rmse(y_val, fold_val_preds):.4f}"
    )

print(f"--> [TabularResNet] Out-Of-Fold RMSE: {calc_rmse(y_tr_full, oof_nn):.4f}")

# =========================================================================
# 5. Out-Of-Fold Optimal Ensemble Optimization
# =========================================================================
print("\n>>> Optimizing Ensemble Weights...")
candidate_oofs = [oof_lgb, oof_xgb, oof_cat, oof_nn]
candidate_tests = [test_lgb, test_xgb, test_cat, test_nn]
n_models = len(candidate_oofs)


def ensemble_objective(weights):
    w_norm = weights / np.sum(weights)
    pred = sum(w * oof for w, oof in zip(w_norm, candidate_oofs))
    return calc_rmse(y_tr_full, pred)


initial_weights = [1.0 / n_models] * n_models
bounds = [(0.0, 1.0) for _ in range(n_models)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    ensemble_objective,
    initial_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
best_weights = opt_res.x / np.sum(opt_res.x)

print(
    f"Ensemble Weights -> LightGBM: {best_weights[0]:.4f}, XGBoost: {best_weights[1]:.4f}, CatBoost: {best_weights[2]:.4f}, TabularResNet: {best_weights[3]:.4f}"
)

oof_ensemble = sum(w * oof for w, oof in zip(best_weights, candidate_oofs))
test_ensemble = sum(w * test for w, test in zip(best_weights, candidate_tests))

if (y_tr_full >= 0).all():
    clipped_oof = np.clip(oof_ensemble, 0, None)
    if calc_rmse(y_tr_full, clipped_oof) <= calc_rmse(y_tr_full, oof_ensemble):
        oof_ensemble = clipped_oof
        test_ensemble = np.clip(test_ensemble, 0, None)

final_validation_score = calc_rmse(y_tr_full, oof_ensemble)

# =========================================================================
# 6. Submission Generation & Verification
# =========================================================================
submission_df = pd.DataFrame(
    {"id": np.arange(n_test, dtype=int), target_col: test_ensemble}
)

submission_df.to_csv(submission_path, index=False)

assert os.path.exists(submission_path), "Submission file not found!"
assert (
    len(submission_df) == n_test
), f"Submission row mismatch: expected {n_test}, got {len(submission_df)}"
assert list(submission_df.columns) == [
    "id",
    target_col,
], f"Incorrect columns: {submission_df.columns}"
assert not submission_df[target_col].isna().any(), "Submission contains NaN values!"
assert not np.isinf(
    submission_df[target_col]
).any(), "Submission contains Infinite values!"

print(f"Submission generated successfully at {submission_path}.")
print(f"Final Validation Score: {final_validation_score}")
