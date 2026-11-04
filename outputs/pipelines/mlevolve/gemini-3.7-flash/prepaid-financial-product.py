import copy
import os
import random
import re
import warnings
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")


# ==============================================================================
# 0. Seed & Environment Setup
# ==============================================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


seed_everything(42)


# ==============================================================================
# 1. Feature Engineering Utilities
# ==============================================================================
def parse_currency_value(val):
    """Robustly parse currency strings, fee ranges, or numeric values into float."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).strip().lower()
    if val_str in ["", "none", "nan", "null", "n/a", "na", "not applicable", "varies"]:
        return np.nan
    if "free" in val_str or "no fee" in val_str or "$0" in val_str or val_str == "0":
        return 0.0

    numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", val_str)
    if not numbers:
        return np.nan
    try:
        float_nums = [float(n) for n in numbers]
        return float(np.mean(float_nums))
    except Exception:
        return np.nan


def extract_date_features(df, col):
    """Extract temporal signals from datetime or date string column."""
    dt_series = pd.to_datetime(df[col], errors="coerce")
    feats = pd.DataFrame(index=df.index)
    feats[f"{col}_year"] = dt_series.dt.year.astype("float32")
    feats[f"{col}_month"] = dt_series.dt.month.astype("float32")
    feats[f"{col}_day"] = dt_series.dt.day.astype("float32")
    feats[f"{col}_dayofweek"] = dt_series.dt.dayofweek.astype("float32")
    feats[f"{col}_quarter"] = dt_series.dt.quarter.astype("float32")
    feats[f"{col}_is_month_end"] = dt_series.dt.is_month_end.astype("float32")

    epoch = pd.Timestamp("2010-01-01")
    feats[f"{col}_days_since_epoch"] = (dt_series - epoch).dt.total_seconds() / (
        24 * 3600
    )
    feats[f"{col}_is_null"] = dt_series.isna().astype("float32")
    return feats, dt_series


# ==============================================================================
# 2. Deep Tabular Neural Network Architecture (TabularResNet)
# ==============================================================================
class ResidualBlock(nn.Module):
    """Pre-LayerNorm Residual Block with Swish (SiLU) activation and Dropout."""

    def __init__(self, d_in: int, d_hidden: int, dropout_rate: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_in)
        self.linear1 = nn.Linear(d_in, d_hidden)
        self.act = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_in)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm1(x)
        out = self.linear1(out)
        out = self.act(out)
        out = self.dropout1(out)
        out = self.norm2(out)
        out = self.linear2(out)
        out = self.dropout2(out)
        return residual + out


class TabularResNet(nn.Module):
    """Deep Tabular Residual Network for Binary Classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    d_in=hidden_dim,
                    d_hidden=hidden_dim * 2,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        for block in self.blocks:
            out = block(out)
        logits = self.head(out).squeeze(-1)
        return logits


def get_lgb_params(random_seed: int = 42) -> dict:
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "n_jobs": -1,
        "random_state": random_seed,
        "verbose": -1,
    }


def get_xgb_params(random_seed: int = 42) -> dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.03,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 1500,
        "tree_method": "hist",
        "random_state": random_seed,
        "n_jobs": -1,
    }


def get_cb_params(random_seed: int = 42) -> dict:
    return {
        "iterations": 1500,
        "learning_rate": 0.03,
        "depth": 6,
        "eval_metric": "AUC",
        "random_seed": random_seed,
        "verbose": False,
        "l2_leaf_reg": 3.0,
    }


# ==============================================================================
# 3. Main End-to-End Pipeline
# ==============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    target_col = "current_status"
    id_col = "id" if "id" in df_train.columns else None

    if target_col not in df_train.columns:
        raise ValueError(f"Target column '{target_col}' not found in train.parquet")

    # Normalize target to binary 0/1 integers
    y_raw = df_train[target_col]
    if y_raw.dtype == "object" or y_raw.dtype.name == "category":
        unique_vals = sorted(list(y_raw.dropna().unique()))
        target_map = {val: i for i, val in enumerate(unique_vals)}
        y_train_series = y_raw.map(target_map).astype(int)
    else:
        y_train_series = y_raw.astype(int)

    n_train = len(df_train)
    n_test = len(df_test)

    if id_col is not None and id_col in df_test.columns:
        test_ids = df_test[id_col].values
    else:
        test_ids = np.arange(n_test)

    # Combine datasets for consistent tabular feature parsing
    df_all = pd.concat(
        [
            df_train.drop(columns=[target_col], errors="ignore"),
            df_test.drop(columns=[target_col], errors="ignore"),
        ],
        axis=0,
        ignore_index=True,
    )

    if id_col in df_all.columns:
        df_all = df_all.drop(columns=[id_col])

    engineered_df = pd.DataFrame(index=df_all.index)

    # 2. Identify Modalities
    candidate_cols = df_all.columns.tolist()
    date_cols = []
    numeric_cols = []
    text_categorical_cols = []

    for col in candidate_cols:
        col_lower = col.lower()
        if (
            "date" in col_lower
            or "time" in col_lower
            or "period" in col_lower
            or "effective" in col_lower
        ):
            date_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df_all[col]):
            numeric_cols.append(col)
        else:
            sample_non_null = df_all[col].dropna().astype(str).head(30)
            if (
                len(sample_non_null) > 0
                and sample_non_null.str.contains(r"[\$€£0-9]").mean() > 0.6
            ):
                parsed_sample = sample_non_null.apply(parse_currency_value)
                if parsed_sample.notna().mean() > 0.4:
                    numeric_cols.append(col)
                else:
                    text_categorical_cols.append(col)
            else:
                text_categorical_cols.append(col)

    # 3. Process Dates & Pairwise Durations
    parsed_datetimes = {}
    for col in date_cols:
        d_feats, dt_obj = extract_date_features(df_all, col)
        for c in d_feats.columns:
            engineered_df[c] = d_feats[c]
        parsed_datetimes[col] = dt_obj

    date_keys = list(parsed_datetimes.keys())
    for i in range(len(date_keys)):
        for j in range(i + 1, len(date_keys)):
            col1, col2 = date_keys[i], date_keys[j]
            diff_days = (
                parsed_datetimes[col1] - parsed_datetimes[col2]
            ).dt.total_seconds() / (24 * 3600)
            engineered_df[f"duration_{col1}_minus_{col2}_days"] = diff_days.astype(
                "float32"
            )
            engineered_df[f"duration_{col1}_minus_{col2}_abs"] = diff_days.abs().astype(
                "float32"
            )

    # 4. Numeric and Fee Aggregation Features
    parsed_num_cols = []
    for col in numeric_cols:
        if pd.api.types.is_numeric_dtype(df_all[col]):
            val = df_all[col].astype("float32")
        else:
            val = df_all[col].apply(parse_currency_value).astype("float32")

        col_clean_name = f"num_{col}"
        engineered_df[col_clean_name] = val
        engineered_df[f"{col_clean_name}_is_null"] = val.isna().astype("float32")
        engineered_df[f"{col_clean_name}_is_zero"] = (val == 0.0).astype("float32")
        engineered_df[f"{col_clean_name}_log1p"] = np.log1p(
            np.maximum(val.fillna(0.0), 0.0)
        ).astype("float32")
        parsed_num_cols.append(col_clean_name)

    if len(parsed_num_cols) > 1:
        num_sub = engineered_df[parsed_num_cols]
        engineered_df["fee_sum"] = num_sub.sum(axis=1, skipna=True).astype("float32")
        engineered_df["fee_mean"] = num_sub.mean(axis=1, skipna=True).astype("float32")
        engineered_df["fee_std"] = (
            num_sub.std(axis=1, skipna=True).fillna(0.0).astype("float32")
        )
        engineered_df["fee_max"] = num_sub.max(axis=1, skipna=True).astype("float32")
        engineered_df["fee_min"] = num_sub.min(axis=1, skipna=True).astype("float32")
        engineered_df["fee_active_count"] = (num_sub > 0).sum(axis=1).astype("float32")
        engineered_df["fee_null_count"] = num_sub.isna().sum(axis=1).astype("float32")
        engineered_df["fee_range"] = (
            engineered_df["fee_max"] - engineered_df["fee_min"]
        ).astype("float32")

        # Pairwise fee interactions: differences and normalized ratios
        for fi in range(min(5, len(parsed_num_cols))):
            for fj in range(fi + 1, min(5, len(parsed_num_cols))):
                col1_n, col2_n = parsed_num_cols[fi], parsed_num_cols[fj]
                val1 = engineered_df[col1_n]
                val2 = engineered_df[col2_n]
                engineered_df[f"{col1_n}_minus_{col2_n}"] = (val1 - val2).astype("float32")
                engineered_df[f"{col1_n}_ratio_{col2_n}"] = (
                    (val1 + 1.0) / (val2.abs() + 1.0)
                ).astype("float32")

    # Fee-to-time duration rates
    duration_cols = [c for c in engineered_df.columns if "duration_" in c and "_days" in c]
    if duration_cols and len(parsed_num_cols) > 0 and "fee_sum" in engineered_df.columns:
        for dur_c in duration_cols[:3]:
            dur_val = engineered_df[dur_c].abs() + 1.0
            engineered_df[f"fee_sum_per_{dur_c}"] = (engineered_df["fee_sum"] / dur_val).astype("float32")
            engineered_df[f"{parsed_num_cols[0]}_per_{dur_c}"] = (
                engineered_df[parsed_num_cols[0]] / dur_val
            ).astype("float32")

    # 5. Frequency & Categorical Features and Missingness Signatures
    engineered_df["row_missing_count"] = df_all.isna().sum(axis=1).astype("float32")
    engineered_df["row_missing_ratio"] = df_all.isna().mean(axis=1).astype("float32")

    for col in text_categorical_cols:
        norm_series = (
            df_all[col].astype(str).str.strip().str.lower().fillna("missing_val")
        )
        freq_map = norm_series.value_counts(normalize=True).to_dict()
        engineered_df[f"{col}_freq"] = norm_series.map(freq_map).astype("float32")
        engineered_df[f"{col}_cat_code"] = norm_series.astype(
            "category"
        ).cat.codes.astype("int32")
        engineered_df[f"{col}_strlen"] = norm_series.str.len().astype("float32")
        engineered_df[f"{col}_wordcnt"] = (
            norm_series.str.split()
            .apply(lambda x: len(x) if isinstance(x, list) else 0)
            .astype("float32")
        )

    # 2-way categorical interaction frequency counts
    for i in range(len(text_categorical_cols)):
        for j in range(i + 1, min(i + 4, len(text_categorical_cols))):
            c1, c2 = text_categorical_cols[i], text_categorical_cols[j]
            pair_series = (
                df_all[c1].astype(str).str.strip().str.lower()
                + "_"
                + df_all[c2].astype(str).str.strip().str.lower()
            )
            pair_freq = pair_series.value_counts(normalize=True).to_dict()
            engineered_df[f"pair_{c1}_{c2}_freq"] = pair_series.map(pair_freq).astype("float32")

    # Field-specific character & word TF-IDF with TruncatedSVD for distinct text fields
    for col in text_categorical_cols:
        col_text = df_all[col].fillna("missing").astype(str).str.strip().str.lower()
        if col_text.nunique() > 5:
            try:
                col_tfidf = TfidfVectorizer(
                    ngram_range=(1, 3), analyzer="char_wb", max_features=2500, min_df=2, sublinear_tf=True
                )
                col_mat = col_tfidf.fit_transform(col_text)
                n_comp = min(6, col_mat.shape[1] - 1)
                if n_comp >= 2:
                    col_svd = TruncatedSVD(n_components=n_comp, random_state=42)
                    col_svd_feats = col_svd.fit_transform(col_mat)
                    for k in range(col_svd_feats.shape[1]):
                        engineered_df[f"{col}_char_svd_{k}"] = col_svd_feats[:, k].astype("float32")
            except Exception:
                pass

    # Global text semantic representation via TF-IDF + TruncatedSVD
    combined_text = pd.Series("", index=df_all.index)
    for col in text_categorical_cols:
        col_str = df_all[col].fillna("").astype(str).str.strip()
        combined_text = combined_text + " " + col_str
    combined_text = combined_text.str.strip().fillna("none")

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2), max_features=10000, min_df=2, sublinear_tf=True
    )
    tfidf_matrix = tfidf.fit_transform(combined_text)
    svd = TruncatedSVD(n_components=16, random_state=42)
    svd_feats = svd.fit_transform(tfidf_matrix)
    for k in range(svd_feats.shape[1]):
        engineered_df[f"tfidf_svd_{k}"] = svd_feats[:, k].astype("float32")

    # 6. Multi-group Hierarchical Aggregations
    high_card_cols = [
        col for col in text_categorical_cols if df_all[col].nunique() > 2
    ]
    target_num_cols = parsed_num_cols[:6]
    for grp_col in high_card_cols[:4]:
        group_norm = df_all[grp_col].astype(str).str.strip().str.lower()
        for num_c in target_num_cols:
            grp_mean = engineered_df[num_c].groupby(group_norm).transform("mean")
            grp_std = (
                engineered_df[num_c]
                .groupby(group_norm)
                .transform("std")
                .fillna(0.0)
            )
            grp_min = engineered_df[num_c].groupby(group_norm).transform("min")
            grp_max = engineered_df[num_c].groupby(group_norm).transform("max")

            engineered_df[f"{grp_col}_{num_c}_mean"] = grp_mean.astype("float32")
            engineered_df[f"{grp_col}_{num_c}_std"] = grp_std.astype("float32")
            engineered_df[f"{grp_col}_{num_c}_min"] = grp_min.astype("float32")
            engineered_df[f"{grp_col}_{num_c}_max"] = grp_max.astype("float32")
            engineered_df[f"{grp_col}_{num_c}_diff_mean"] = (
                engineered_df[num_c] - grp_mean
            ).astype("float32")
            engineered_df[f"{grp_col}_{num_c}_ratio_mean"] = (
                engineered_df[num_c] / (grp_mean + 1e-4)
            ).astype("float32")
            engineered_df[f"{grp_col}_{num_c}_zscore"] = (
                (engineered_df[num_c] - grp_mean) / (grp_std + 1e-4)
            ).astype("float32")

    # Split into Train and Test
    X_train = engineered_df.iloc[:n_train].copy().reset_index(drop=True)
    X_test = engineered_df.iloc[n_train:].copy().reset_index(drop=True)

    # 7. Stratified 10-Fold Setup & Leak-Free Out-of-Fold Target Encodings
    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = np.zeros(n_train, dtype=int)
    y_train = y_train_series.values

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        folds[val_idx] = fold

    smoothing_weight = 10.0
    global_target_mean = float(y_train.mean())

    for col in text_categorical_cols[:8]:
        te_train_col = np.zeros(n_train, dtype=np.float32)
        test_cat_series = df_all[col].iloc[n_train:].astype(str).str.strip().str.lower()

        for fold in range(n_splits):
            tr_mask = folds != fold
            va_mask = folds == fold

            tr_cat = (
                df_all[col]
                .iloc[:n_train]
                .iloc[tr_mask]
                .astype(str)
                .str.strip()
                .str.lower()
            )
            tr_y = y_train_series.iloc[tr_mask]

            stats = tr_y.groupby(tr_cat).agg(["count", "mean"])
            smoothed_te = (
                stats["count"] * stats["mean"] + smoothing_weight * global_target_mean
            ) / (stats["count"] + smoothing_weight)
            te_map = smoothed_te.to_dict()

            va_cat = (
                df_all[col]
                .iloc[:n_train]
                .iloc[va_mask]
                .astype(str)
                .str.strip()
                .str.lower()
            )
            te_train_col[va_mask] = (
                va_cat.map(te_map).fillna(global_target_mean).astype(np.float32)
            )

        X_train[f"{col}_target_enc"] = te_train_col

        full_tr_cat = df_all[col].iloc[:n_train].astype(str).str.strip().str.lower()
        stats_full = y_train_series.groupby(full_tr_cat).agg(["count", "mean"])
        smoothed_te_full = (
            stats_full["count"] * stats_full["mean"]
            + smoothing_weight * global_target_mean
        ) / (stats_full["count"] + smoothing_weight)
        te_map_full = smoothed_te_full.to_dict()
        X_test[f"{col}_target_enc"] = (
            test_cat_series.map(te_map_full)
            .fillna(global_target_mean)
            .astype(np.float32)
        )

    # 8. Clean infinite and missing values
    feature_cols = [c for c in X_train.columns]
    for col in feature_cols:
        X_train[col] = X_train[col].replace([np.inf, -np.inf], np.nan)
        X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan)
        col_median = X_train[col].median()
        if pd.isna(col_median):
            col_median = 0.0
        X_train[col] = X_train[col].fillna(col_median)
        X_test[col] = X_test[col].fillna(col_median)

    # ==============================================================================
    # 4. Stratified 10-Fold Training Loop (LightGBM, XGBoost, CatBoost, TabularResNet)
    # ==============================================================================
    num_features = len(feature_cols)
    y_train_float = y_train.astype(np.float32)

    oof_lgb = np.zeros(n_train, dtype=np.float32)
    oof_xgb = np.zeros(n_train, dtype=np.float32)
    oof_cb = np.zeros(n_train, dtype=np.float32)
    oof_nn = np.zeros(n_train, dtype=np.float32)

    test_preds_lgb = np.zeros(n_test, dtype=np.float32)
    test_preds_xgb = np.zeros(n_test, dtype=np.float32)
    test_preds_cb = np.zeros(n_test, dtype=np.float32)
    test_preds_nn = np.zeros(n_test, dtype=np.float32)

    X_train_mat = X_train.values.astype(np.float32)
    X_test_mat = X_test.values.astype(np.float32)

    for fold in range(n_splits):
        tr_idx = np.where(folds != fold)[0]
        va_idx = np.where(folds == fold)[0]

        X_tr = X_train_mat[tr_idx]
        y_tr = y_train_float[tr_idx]
        X_va = X_train_mat[va_idx]
        y_va = y_train_float[va_idx]
        X_te = X_test_mat

        # --- 1. LightGBM ---
        lgb_dtrain = lgb.Dataset(X_tr, label=y_tr)
        lgb_dval = lgb.Dataset(X_va, label=y_va, reference=lgb_dtrain)
        lgb_params = get_lgb_params(random_seed=42 + fold)
        lgb_model = lgb.train(
            lgb_params,
            lgb_dtrain,
            num_boost_round=1500,
            valid_sets=[lgb_dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        oof_lgb[va_idx] = lgb_model.predict(
            X_va, num_iteration=lgb_model.best_iteration
        )
        test_preds_lgb += (
            lgb_model.predict(X_te, num_iteration=lgb_model.best_iteration) / n_splits
        )

        # --- 2. XGBoost ---
        xgb_params = get_xgb_params(random_seed=42 + fold)
        xgb_model = xgb.XGBClassifier(
            **xgb_params,
            early_stopping_rounds=50,
        )
        xgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
        oof_xgb[va_idx] = xgb_model.predict_proba(X_va)[:, 1]
        test_preds_xgb += xgb_model.predict_proba(X_te)[:, 1] / n_splits

        # --- 3. CatBoost ---
        cb_params = get_cb_params(random_seed=42 + fold)
        cb_model = CatBoostClassifier(**cb_params)
        cb_model.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=50,
            verbose=False,
        )
        oof_cb[va_idx] = cb_model.predict_proba(X_va)[:, 1]
        test_preds_cb += cb_model.predict_proba(X_te)[:, 1] / n_splits

        # --- 4. TabularResNet ---
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)
        X_te_scaled = scaler.transform(X_te)

        train_dataset = TensorDataset(
            torch.tensor(X_tr_scaled, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_va_scaled, dtype=torch.float32),
            torch.tensor(y_va, dtype=torch.float32),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=256, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=512, shuffle=False, drop_last=False
        )

        nn_model = TabularResNet(
            input_dim=num_features,
            hidden_dim=128,
            num_blocks=3,
            dropout_rate=0.15,
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4, eps=1e-8)
        epochs = 25
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        best_val_loss = float("inf")
        best_nn_weights = None

        for epoch in range(epochs):
            nn_model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits = nn_model(bx)
                loss = criterion(logits, by)
                loss.backward()
                nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

            nn_model.eval()
            val_loss = 0.0
            val_samples = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    v_logits = nn_model(bx)
                    v_loss = criterion(v_logits, by)
                    val_loss += v_loss.item() * len(by)
                    val_samples += len(by)

            avg_val_loss = val_loss / max(val_samples, 1)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_nn_weights = copy.deepcopy(nn_model.state_dict())

        nn_model.load_state_dict(best_nn_weights)
        nn_model.eval()
        with torch.no_grad():
            va_tensor = torch.tensor(X_va_scaled, dtype=torch.float32).to(device)
            te_tensor = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)
            oof_nn[va_idx] = torch.sigmoid(nn_model(va_tensor)).cpu().numpy()
            test_preds_nn += torch.sigmoid(nn_model(te_tensor)).cpu().numpy() / n_splits

    # ==============================================================================
    # 5. Out-of-Fold AUC Optimization & Submission Generation
    # ==============================================================================
    oof_lgb_rank = (rankdata(oof_lgb) - 0.5) / len(oof_lgb)
    oof_xgb_rank = (rankdata(oof_xgb) - 0.5) / len(oof_xgb)
    oof_cb_rank = (rankdata(oof_cb) - 0.5) / len(oof_cb)
    oof_nn_rank = (rankdata(oof_nn) - 0.5) / len(oof_nn)

    test_lgb_rank = (rankdata(test_preds_lgb) - 0.5) / len(test_preds_lgb)
    test_xgb_rank = (rankdata(test_preds_xgb) - 0.5) / len(test_preds_xgb)
    test_cb_rank = (rankdata(test_preds_cb) - 0.5) / len(test_preds_cb)
    test_nn_rank = (rankdata(test_preds_nn) - 0.5) / len(test_preds_nn)

    model_oofs_rank = [oof_lgb_rank, oof_xgb_rank, oof_cb_rank, oof_nn_rank]
    model_test_ranks = [test_lgb_rank, test_xgb_rank, test_cb_rank, test_nn_rank]
    num_models = len(model_oofs_rank)

    def auc_loss(weights):
        weights = np.array(weights)
        weights = weights / (np.sum(weights) + 1e-12)
        blend = np.zeros_like(oof_lgb_rank)
        for w, pred in zip(weights, model_oofs_rank):
            blend += w * pred
        return -roc_auc_score(y_train, blend)

    init_weights = [1.0 / num_models] * num_models
    bounds = [(0.0, 1.0) for _ in range(num_models)]
    constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}

    res = minimize(
        auc_loss,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-8},
    )

    opt_weights = res.x / np.sum(res.x)

    oof_blend = np.zeros_like(oof_lgb_rank)
    test_blend = np.zeros_like(test_lgb_rank)
    for w, oof_p, test_p in zip(opt_weights, model_oofs_rank, model_test_ranks):
        oof_blend += w * oof_p
        test_blend += w * test_p

    final_score = roc_auc_score(y_train, oof_blend)

    os.makedirs("./submission", exist_ok=True)
    submission_df = pd.DataFrame(
        {
            "id": test_ids.astype(int),
            "current_status": np.clip(test_blend, 0.0, 1.0),
        }
    )
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()
