import gc
import json
import math
import os
import warnings
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

torch.manual_seed(42)

warnings.filterwarnings("ignore")

# =========================================================================
# 1. Directory Setup & Data Ingestion
# =========================================================================
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
MODELS_DIR = os.path.join(WORKING_DIR, "models")

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

train_path = os.path.join(INPUT_DIR, "train.parquet")
test_path = os.path.join(INPUT_DIR, "test.parquet")

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

TARGET_COL = "amountUSD"
if TARGET_COL not in train_df.columns:
    possible_targets = [c for c in train_df.columns if "amount" in c.lower()]
    TARGET_COL = possible_targets[0] if possible_targets else train_df.columns[-1]

# Ensure valid target numeric values
train_df[TARGET_COL] = pd.to_numeric(train_df[TARGET_COL], errors="coerce").fillna(0.0)
train_df[TARGET_COL] = np.clip(train_df[TARGET_COL], a_min=0.0, a_max=None)

id_candidates = [c for c in ["id", "ID", "Id", "index"] if c in train_df.columns]
ID_COL = id_candidates[0] if id_candidates else None

# =========================================================================
# 2. Modality Detection & Feature Engineering
# =========================================================================
feature_cols = [c for c in train_df.columns if c not in [TARGET_COL, ID_COL]]

datetime_cols = []
text_cols = []
categorical_cols = []
numerical_cols = []

for col in feature_cols:
    col_dtype = train_df[col].dtype
    if (
        pd.api.types.is_datetime64_any_dtype(col_dtype)
        or "date" in col.lower()
        or "time" in col.lower()
        or "year" in col.lower()
    ):
        datetime_cols.append(col)
    elif pd.api.types.is_numeric_dtype(col_dtype):
        if 1 < train_df[col].nunique() < 20:
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)
    else:
        sample_vals = train_df[col].dropna().astype(str)
        if len(sample_vals) > 0 and sample_vals.str.len().mean() > 50:
            text_cols.append(col)
        else:
            categorical_cols.append(col)


def engineer_base_features(df):
    res = df.copy()

    # Datetime features
    for col in datetime_cols:
        parsed_dt = pd.to_datetime(res[col], errors="coerce")
        if parsed_dt.notna().sum() > 0:
            prefix = col.replace(" ", "_").replace("/", "_")
            res[f"{prefix}_year"] = parsed_dt.dt.year.fillna(-1).astype(int)
            res[f"{prefix}_month"] = parsed_dt.dt.month.fillna(-1).astype(int)
            res[f"{prefix}_day"] = parsed_dt.dt.day.fillna(-1).astype(int)
            res[f"{prefix}_dayofweek"] = parsed_dt.dt.dayofweek.fillna(-1).astype(int)
            res[f"{prefix}_quarter"] = parsed_dt.dt.quarter.fillna(-1).astype(int)
            res[f"{prefix}_is_month_end"] = parsed_dt.dt.is_month_end.fillna(
                False
            ).astype(int)
            res[f"{prefix}_is_year_end"] = (parsed_dt.dt.month == 12).astype(int)

            month_rad = 2 * np.pi * parsed_dt.dt.month.fillna(1) / 12.0
            res[f"{prefix}_month_sin"] = np.sin(month_rad)
            res[f"{prefix}_month_cos"] = np.cos(month_rad)

            baseline = pd.Timestamp("2000-01-01")
            res[f"{prefix}_days_since_2000"] = (
                (parsed_dt - baseline).dt.days.fillna(-1).astype(float)
            )

            if col in res.columns and col not in [ID_COL, TARGET_COL]:
                res.drop(columns=[col], inplace=True)

    # Categorical normalization
    for col in categorical_cols:
        if col in res.columns:
            res[col] = (
                res[col].astype(str).fillna("__MISSING__").str.strip().str.lower()
            )
            res[f"{col}_is_missing"] = (
                res[col].isin(["__missing__", "nan", "none", ""])
            ).astype(int)

    # Numerical normalization
    for col in numerical_cols:
        if col in res.columns:
            res[col] = pd.to_numeric(res[col], errors="coerce")
            res[f"{col}_is_na"] = res[col].isna().astype(int)
            res[col] = res[col].fillna(0.0)

    # Relational interaction pairs
    cat_keys = [c for c in categorical_cols if c in res.columns]
    if len(cat_keys) >= 2:
        for i in range(min(5, len(cat_keys))):
            for j in range(i + 1, min(6, len(cat_keys))):
                c1, c2 = cat_keys[i], cat_keys[j]
                res[f"{c1}__x__{c2}"] = res[c1].astype(str) + "__" + res[c2].astype(str)

    return res


train_eng = engineer_base_features(train_df)
test_eng = engineer_base_features(test_df)

# Multi-Scale Text Feature Extraction (Word + Char Subword TF-IDF + SVD)
for col in text_cols:
    train_texts = train_df[col].fillna("").astype(str).values
    test_texts = test_df[col].fillna("").astype(str).values

    train_eng[f"{col}_char_len"] = [len(t) for t in train_texts]
    train_eng[f"{col}_word_count"] = [len(t.split()) for t in train_texts]
    train_eng[f"{col}_avg_word_len"] = [
        np.mean([len(w) for w in t.split()]) if len(t.split()) > 0 else 0.0
        for t in train_texts
    ]
    train_eng[f"{col}_digit_count"] = [sum(c.isdigit() for c in t) for t in train_texts]
    train_eng[f"{col}_upper_count"] = [sum(c.isupper() for c in t) for t in train_texts]

    test_eng[f"{col}_char_len"] = [len(t) for t in test_texts]
    test_eng[f"{col}_word_count"] = [len(t.split()) for t in test_texts]
    test_eng[f"{col}_avg_word_len"] = [
        np.mean([len(w) for w in t.split()]) if len(t.split()) > 0 else 0.0
        for t in test_texts
    ]
    test_eng[f"{col}_digit_count"] = [sum(c.isdigit() for c in t) for t in test_texts]
    test_eng[f"{col}_upper_count"] = [sum(c.isupper() for c in t) for t in test_texts]

    n_word_comp = min(24, max(2, len(train_df) // 60))
    tfidf_word = TfidfVectorizer(
        max_features=12000, ngram_range=(1, 3), stop_words="english", min_df=2
    )
    svd_word = TruncatedSVD(n_components=n_word_comp, random_state=42)

    train_tfidf_w = tfidf_word.fit_transform(train_texts)
    test_tfidf_w = tfidf_word.transform(test_texts)

    train_svd_w = svd_word.fit_transform(train_tfidf_w)
    test_svd_w = svd_word.transform(test_tfidf_w)

    for k in range(n_word_comp):
        train_eng[f"{col}_word_svd_{k}"] = train_svd_w[:, k]
        test_eng[f"{col}_word_svd_{k}"] = test_svd_w[:, k]

    n_char_comp = min(16, max(2, len(train_df) // 80))
    tfidf_char = TfidfVectorizer(
        analyzer="char_wb", max_features=10000, ngram_range=(3, 5), min_df=2
    )
    svd_char = TruncatedSVD(n_components=n_char_comp, random_state=42)

    train_tfidf_c = tfidf_char.fit_transform(train_texts)
    test_tfidf_c = tfidf_char.transform(test_texts)

    train_svd_c = svd_char.fit_transform(train_tfidf_c)
    test_svd_c = svd_char.transform(test_tfidf_c)

    for k in range(n_char_comp):
        train_eng[f"{col}_char_svd_{k}"] = train_svd_c[:, k]
        test_eng[f"{col}_char_svd_{k}"] = test_svd_c[:, k]

    if col in train_eng.columns:
        train_eng.drop(columns=[col], inplace=True)
    if col in test_eng.columns:
        test_eng.drop(columns=[col], inplace=True)

# Out-of-Fold 10-Fold Stratified Split Definition
N_SPLITS = 10
train_eng["fold"] = -1
target_vals = train_eng[TARGET_COL].values
log_target = np.log1p(np.maximum(0, target_vals))

n_bins = min(15, len(np.unique(log_target)))
try:
    strat_bins = pd.qcut(log_target, q=n_bins, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    split_gen = skf.split(train_eng, strat_bins)
except Exception:
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    split_gen = kf.split(train_eng)

for fold_idx, (trn_idx, val_idx) in enumerate(split_gen):
    train_eng.loc[val_idx, "fold"] = fold_idx

# Leak-Free Frequency & Multi-Statistic Target Encoding
all_cat_cols = [
    c
    for c in train_eng.columns
    if (train_eng[c].dtype == "object" or str(train_eng[c].dtype) == "category")
    and c not in [TARGET_COL, "fold", ID_COL]
]

SMOOTHING_WEIGHT = 10.0
global_mean = float(np.mean(target_vals))
global_median = float(np.median(target_vals))
global_min = float(np.min(target_vals))
global_max = float(np.max(target_vals))

for cat in all_cat_cols:
    train_eng[f"{cat}_freq"] = 0.0
    test_eng[f"{cat}_freq"] = 0.0

    freq_map = train_eng[cat].value_counts(normalize=True).to_dict()
    test_eng[f"{cat}_freq"] = test_eng[cat].map(freq_map).fillna(0.0).astype(float)

    for fold in range(N_SPLITS):
        trn_mask = train_eng["fold"] != fold
        val_mask = train_eng["fold"] == fold
        fold_freq = train_eng.loc[trn_mask, cat].value_counts(normalize=True).to_dict()
        train_eng.loc[val_mask, f"{cat}_freq"] = (
            train_eng.loc[val_mask, cat].map(fold_freq).fillna(0.0).astype(float)
        )

    # Hierarchical Target Statistics (Mean, Median, Std, Count, Min, Max, Sum)
    cat_stats = (
        train_eng.groupby(cat)[TARGET_COL]
        .agg(count="count", mean="mean", median="median", std="std", min="min", max="max", sum="sum")
        .reset_index()
    )
    cat_stats["smoothed_mean"] = (
        cat_stats["count"] * cat_stats["mean"] + SMOOTHING_WEIGHT * global_mean
    ) / (cat_stats["count"] + SMOOTHING_WEIGHT)

    mean_dict = dict(zip(cat_stats[cat], cat_stats["smoothed_mean"]))
    med_dict = dict(zip(cat_stats[cat], cat_stats["median"].fillna(global_median)))
    std_dict = dict(zip(cat_stats[cat], cat_stats["std"].fillna(0.0)))
    cnt_dict = dict(zip(cat_stats[cat], np.log1p(cat_stats["count"])))
    min_dict = dict(zip(cat_stats[cat], cat_stats["min"].fillna(global_min)))
    max_dict = dict(zip(cat_stats[cat], cat_stats["max"].fillna(global_max)))
    sum_dict = dict(zip(cat_stats[cat], cat_stats["sum"].fillna(0.0)))

    test_eng[f"{cat}_target_mean"] = test_eng[cat].map(mean_dict).fillna(global_mean).astype(float)
    test_eng[f"{cat}_target_median"] = test_eng[cat].map(med_dict).fillna(global_median).astype(float)
    test_eng[f"{cat}_target_std"] = test_eng[cat].map(std_dict).fillna(0.0).astype(float)
    test_eng[f"{cat}_target_count"] = test_eng[cat].map(cnt_dict).fillna(0.0).astype(float)
    test_eng[f"{cat}_target_min"] = test_eng[cat].map(min_dict).fillna(global_min).astype(float)
    test_eng[f"{cat}_target_max"] = test_eng[cat].map(max_dict).fillna(global_max).astype(float)
    test_eng[f"{cat}_target_sum"] = test_eng[cat].map(sum_dict).fillna(0.0).astype(float)

    train_eng[f"{cat}_target_mean"] = 0.0
    train_eng[f"{cat}_target_median"] = 0.0
    train_eng[f"{cat}_target_std"] = 0.0
    train_eng[f"{cat}_target_count"] = 0.0
    train_eng[f"{cat}_target_min"] = 0.0
    train_eng[f"{cat}_target_max"] = 0.0
    train_eng[f"{cat}_target_sum"] = 0.0

    for fold in range(N_SPLITS):
        trn_mask = train_eng["fold"] != fold
        val_mask = train_eng["fold"] == fold

        trn_data = train_eng.loc[trn_mask]
        trn_target = trn_data[TARGET_COL].values
        trn_global_mean = float(np.mean(trn_target))
        trn_global_median = float(np.median(trn_target))
        trn_global_min = float(np.min(trn_target))
        trn_global_max = float(np.max(trn_target))

        fold_stats = (
            trn_data.groupby(cat)[TARGET_COL]
            .agg(count="count", mean="mean", median="median", std="std", min="min", max="max", sum="sum")
            .reset_index()
        )
        fold_stats["smoothed_mean"] = (
            fold_stats["count"] * fold_stats["mean"]
            + SMOOTHING_WEIGHT * trn_global_mean
        ) / (fold_stats["count"] + SMOOTHING_WEIGHT)

        f_mean_dict = dict(zip(fold_stats[cat], fold_stats["smoothed_mean"]))
        f_med_dict = dict(zip(fold_stats[cat], fold_stats["median"].fillna(trn_global_median)))
        f_std_dict = dict(zip(fold_stats[cat], fold_stats["std"].fillna(0.0)))
        f_cnt_dict = dict(zip(fold_stats[cat], np.log1p(fold_stats["count"])))
        f_min_dict = dict(zip(fold_stats[cat], fold_stats["min"].fillna(trn_global_min)))
        f_max_dict = dict(zip(fold_stats[cat], fold_stats["max"].fillna(trn_global_max)))
        f_sum_dict = dict(zip(fold_stats[cat], fold_stats["sum"].fillna(0.0)))

        train_eng.loc[val_mask, f"{cat}_target_mean"] = (
            train_eng.loc[val_mask, cat].map(f_mean_dict).fillna(trn_global_mean).astype(float)
        )
        train_eng.loc[val_mask, f"{cat}_target_median"] = (
            train_eng.loc[val_mask, cat].map(f_med_dict).fillna(trn_global_median).astype(float)
        )
        train_eng.loc[val_mask, f"{cat}_target_std"] = (
            train_eng.loc[val_mask, cat].map(f_std_dict).fillna(0.0).astype(float)
        )
        train_eng.loc[val_mask, f"{cat}_target_count"] = (
            train_eng.loc[val_mask, cat].map(f_cnt_dict).fillna(0.0).astype(float)
        )
        train_eng.loc[val_mask, f"{cat}_target_min"] = (
            train_eng.loc[val_mask, cat].map(f_min_dict).fillna(trn_global_min).astype(float)
        )
        train_eng.loc[val_mask, f"{cat}_target_max"] = (
            train_eng.loc[val_mask, cat].map(f_max_dict).fillna(trn_global_max).astype(float)
        )
        train_eng.loc[val_mask, f"{cat}_target_sum"] = (
            train_eng.loc[val_mask, cat].map(f_sum_dict).fillna(0.0).astype(float)
        )

# High-Order Group Statistics and Relative Ratio / Difference Features
key_num_cols = [c for c in numerical_cols if c in train_eng.columns]
for dt_col in [c for c in train_eng.columns if "days_since_2000" in c]:
    if dt_col not in key_num_cols:
        key_num_cols.append(dt_col)

for cat in all_cat_cols[:8]:
    for num_col in key_num_cols[:6]:
        grp_stats = train_eng.groupby(cat)[num_col].agg(
            mean="mean", median="median", std="std", min="min", max="max", sum="sum"
        )
        mean_map = grp_stats["mean"].to_dict()
        med_map = grp_stats["median"].to_dict()
        std_map = grp_stats["std"].fillna(0.0).to_dict()
        min_map = grp_stats["min"].to_dict()
        max_map = grp_stats["max"].to_dict()
        sum_map = grp_stats["sum"].to_dict()

        for agg_name, agg_m in [
            ("mean", mean_map),
            ("median", med_map),
            ("std", std_map),
            ("min", min_map),
            ("max", max_map),
            ("sum", sum_map),
        ]:
            feat_name = f"{cat}_{num_col}_{agg_name}"
            train_eng[feat_name] = train_eng[cat].map(agg_m).fillna(0.0).astype(float)
            test_eng[feat_name] = test_eng[cat].map(agg_m).fillna(0.0).astype(float)

        # Relative Ratio and Difference Features
        train_mean = train_eng[cat].map(mean_map).fillna(0.0).astype(float)
        test_mean = test_eng[cat].map(mean_map).fillna(0.0).astype(float)
        train_sum = train_eng[cat].map(sum_map).fillna(0.0).astype(float)
        test_sum = test_eng[cat].map(sum_map).fillna(0.0).astype(float)
        train_med = train_eng[cat].map(med_map).fillna(0.0).astype(float)
        test_med = test_eng[cat].map(med_map).fillna(0.0).astype(float)

        train_eng[f"{cat}_{num_col}_diff_mean"] = (train_eng[num_col] - train_mean).astype(float)
        test_eng[f"{cat}_{num_col}_diff_mean"] = (test_eng[num_col] - test_mean).astype(float)

        train_eng[f"{cat}_{num_col}_diff_med"] = (train_eng[num_col] - train_med).astype(float)
        test_eng[f"{cat}_{num_col}_diff_med"] = (test_eng[num_col] - test_med).astype(float)

        train_eng[f"{cat}_{num_col}_ratio_mean"] = (
            train_eng[num_col] / (train_mean.abs() + 1e-5)
        ).astype(float)
        test_eng[f"{cat}_{num_col}_ratio_mean"] = (
            test_eng[num_col] / (test_mean.abs() + 1e-5)
        ).astype(float)

        train_eng[f"{cat}_{num_col}_sum_share"] = (
            train_eng[num_col] / (train_sum.abs() + 1e-5)
        ).astype(float)
        test_eng[f"{cat}_{num_col}_sum_share"] = (
            test_eng[num_col] / (test_sum.abs() + 1e-5)
        ).astype(float)

# Categorical Label Encoding for GBDT
for cat in all_cat_cols:
    train_eng[cat] = train_eng[cat].astype(str)
    test_eng[cat] = test_eng[cat].astype(str)

    unique_vals = train_eng[cat].unique()
    val_to_idx = {val: idx + 1 for idx, val in enumerate(unique_vals)}
    val_to_idx["__UNSEEN__"] = 0

    train_eng[f"{cat}_cat_code"] = train_eng[cat].map(val_to_idx).fillna(0).astype(int)
    test_eng[f"{cat}_cat_code"] = test_eng[cat].map(val_to_idx).fillna(0).astype(int)

    train_eng.drop(columns=[cat], inplace=True)
    if cat in test_eng.columns:
        test_eng.drop(columns=[cat], inplace=True)

# Align Feature Matrix
meta_cols = [TARGET_COL, "fold"]
if ID_COL and ID_COL in train_eng.columns:
    meta_cols.append(ID_COL)

feature_names = [c for c in train_eng.columns if c not in meta_cols]

for col in feature_names:
    if col not in test_eng.columns:
        test_eng[col] = 0.0

train_eng[feature_names] = (
    train_eng[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
)
test_eng[feature_names] = (
    test_eng[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
)


# =========================================================================
# 3. Model Architectures & Parameter Configurations
# =========================================================================
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        return torch.sqrt(torch.mean((y_pred - y_true) ** 2) + self.eps)


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.ln1 = nn.LayerNorm(out_features)
        self.act1 = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(out_features, out_features)
        self.ln2 = nn.LayerNorm(out_features)
        self.act2 = nn.SiLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.shortcut = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.act1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.ln2(out)
        out = self.dropout2(out)
        return self.act2(out + residual)


class TabularFundingMLP(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_dims: list = [256, 128, 64],
        dropout_rate: float = 0.2,
        use_positive_head: bool = True,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(num_features)
        self.use_positive_head = use_positive_head

        layers = []
        in_dim = num_features
        for h_dim in hidden_dims:
            layers.append(ResidualBlock(in_dim, h_dim, dropout_rate=dropout_rate))
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.regressor = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x_num):
        x = self.input_norm(x_num)
        h = self.backbone(x)
        out = self.regressor(h).squeeze(-1)
        if self.use_positive_head:
            out = F.softplus(out)
        return out


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


# =========================================================================
# 4. K-Fold Training & Cross-Validation
# =========================================================================
oof_preds_lgb = np.zeros(len(train_eng), dtype=np.float64)
oof_preds_lgb_et = np.zeros(len(train_eng), dtype=np.float64)
oof_preds_xgb = np.zeros(len(train_eng), dtype=np.float64)
oof_preds_cb = np.zeros(len(train_eng), dtype=np.float64)
test_preds_lgb = np.zeros(len(test_eng), dtype=np.float64)
test_preds_lgb_et = np.zeros(len(test_eng), dtype=np.float64)
test_preds_xgb = np.zeros(len(test_eng), dtype=np.float64)
test_preds_cb = np.zeros(len(test_eng), dtype=np.float64)

X_test = test_eng[feature_names].values
dtest_xgb = xgb.DMatrix(X_test)

for fold in range(N_SPLITS):
    trn_mask = train_eng["fold"] != fold
    val_mask = train_eng["fold"] == fold

    X_train = train_eng.loc[trn_mask, feature_names].values
    y_train = train_eng.loc[trn_mask, TARGET_COL].values
    X_val = train_eng.loc[val_mask, feature_names].values
    y_val = train_eng.loc[val_mask, TARGET_COL].values

    # LightGBM
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 8,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "random_state": 42 + fold,
        "n_jobs": -1,
        "verbose": -1,
    }

    trn_lgb_data = lgb.Dataset(X_train, label=y_train)
    val_lgb_data = lgb.Dataset(X_val, label=y_val, reference=trn_lgb_data)

    lgb_model = lgb.train(
        lgb_params,
        trn_lgb_data,
        num_boost_round=1500,
        valid_sets=[val_lgb_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_lgb_pred = np.clip(
        lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration),
        a_min=0.0,
        a_max=None,
    )
    oof_preds_lgb[val_mask] = val_lgb_pred
    test_preds_lgb += (
        np.clip(
            lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration),
            a_min=0.0,
            a_max=None,
        )
        / N_SPLITS
    )

    # LightGBM (ExtraTrees mode)
    lgb_et_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "extra_trees": True,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 8,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "random_state": 142 + fold,
        "n_jobs": -1,
        "verbose": -1,
    }

    lgb_et_model = lgb.train(
        lgb_et_params,
        trn_lgb_data,
        num_boost_round=1500,
        valid_sets=[val_lgb_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_lgb_et_pred = np.clip(
        lgb_et_model.predict(X_val, num_iteration=lgb_et_model.best_iteration),
        a_min=0.0,
        a_max=None,
    )
    oof_preds_lgb_et[val_mask] = val_lgb_et_pred
    test_preds_lgb_et += (
        np.clip(
            lgb_et_model.predict(X_test, num_iteration=lgb_et_model.best_iteration),
            a_min=0.0,
            a_max=None,
        )
        / N_SPLITS
    )

    # XGBoost
    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "learning_rate": 0.03,
        "max_depth": 7,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42 + fold,
        "n_jobs": -1,
    }

    dtrain_xgb = xgb.DMatrix(X_train, label=y_train)
    dval_xgb = xgb.DMatrix(X_val, label=y_val)

    xgb_model = xgb.train(
        xgb_params,
        dtrain_xgb,
        num_boost_round=1500,
        evals=[(dval_xgb, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    val_xgb_pred = np.clip(
        xgb_model.predict(dval_xgb, iteration_range=(0, xgb_model.best_iteration + 1)),
        a_min=0.0,
        a_max=None,
    )
    oof_preds_xgb[val_mask] = val_xgb_pred
    test_preds_xgb += (
        np.clip(
            xgb_model.predict(
                dtest_xgb, iteration_range=(0, xgb_model.best_iteration + 1)
            ),
            a_min=0.0,
            a_max=None,
        )
        / N_SPLITS
    )

    # CatBoost
    cb_model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42 + fold,
        verbose=False,
    )
    cb_model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False,
    )

    val_cb_pred = np.clip(
        cb_model.predict(X_val),
        a_min=0.0,
        a_max=None,
    )
    oof_preds_cb[val_mask] = val_cb_pred
    test_preds_cb += (
        np.clip(
            cb_model.predict(X_test),
            a_min=0.0,
            a_max=None,
        )
        / N_SPLITS
    )

    del trn_lgb_data, val_lgb_data, dtrain_xgb, dval_xgb, cb_model, lgb_model, lgb_et_model, xgb_model
    gc.collect()

# =========================================================================
# 5. Out-of-Fold Blending & Validation Score Computation
# =========================================================================
y_true = train_eng[TARGET_COL].values
oof_matrix = np.column_stack([oof_preds_lgb, oof_preds_lgb_et, oof_preds_xgb, oof_preds_cb])
test_matrix = np.column_stack([test_preds_lgb, test_preds_lgb_et, test_preds_xgb, test_preds_cb])
n_models = oof_matrix.shape[1]


def stacking_objective(weights):
    pred = np.dot(oof_matrix, weights)
    rmse = compute_rmse(y_true, np.clip(pred, a_min=0.0, a_max=None))
    ridge_penalty = 1e-5 * np.sum(weights ** 2)
    return rmse + ridge_penalty


opt_res = minimize(
    stacking_objective,
    x0=np.ones(n_models) / n_models,
    method="SLSQP",
    bounds=[(0.0, 1.0) for _ in range(n_models)],
    constraints={"type": "eq", "fun": lambda w: 1.0 - np.sum(w)},
)

if opt_res.success:
    best_weights = opt_res.x / np.sum(opt_res.x)
else:
    best_weights = np.ones(n_models) / n_models

oof_blended = np.clip(
    np.dot(oof_matrix, best_weights),
    a_min=0.0,
    a_max=None,
)
final_val_score = compute_rmse(y_true, oof_blended)

# =========================================================================
# 6. Submission Generation
# =========================================================================
final_test_preds = np.clip(
    np.dot(test_matrix, best_weights),
    a_min=0.0,
    a_max=None,
)
final_test_preds = np.nan_to_num(final_test_preds, nan=0.0)

submission_df = pd.DataFrame(
    {"id": np.arange(len(test_df)), "amountUSD": final_test_preds}
)

submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df.to_csv(submission_path, index=False)

print(f"Final Validation Score: {final_val_score}")
