import os
import re
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import warnings
import itertools
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

seed_everything(42)

# 1. Setup directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 2. Load Raw Datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Amount in USD"
assert (
    target_col in train_df.columns
), f"Target '{target_col}' not found in train.parquet"

y_train = train_df[target_col].copy()
train_features_df = train_df.drop(columns=[target_col]).copy()
test_features_df = test_df.copy()

feature_cols_raw = list(train_features_df.columns)


# 3. Identify Column Types (Dates, Numerics, Text, Categoricals)
def extract_date_features(df, col):
    """Extract multi-scale date, cyclic, and fiscal features from datetime or string date column."""
    dt = pd.to_datetime(df[col], errors="coerce")
    feats = {}
    feats[f"{col}_year"] = dt.dt.year.fillna(-1).astype(np.int32).values
    feats[f"{col}_month"] = dt.dt.month.fillna(-1).astype(np.int32).values
    feats[f"{col}_quarter"] = dt.dt.quarter.fillna(-1).astype(np.int32).values
    feats[f"{col}_day"] = dt.dt.day.fillna(-1).astype(np.int32).values
    feats[f"{col}_dayofweek"] = dt.dt.dayofweek.fillna(-1).astype(np.int32).values
    feats[f"{col}_is_month_end"] = dt.dt.is_month_end.astype(float).fillna(0.0).values
    feats[f"{col}_fiscal_year"] = np.where(
        dt.dt.month >= 7, dt.dt.year + 1, dt.dt.year
    ).astype(np.float32)
    feats[f"{col}_days_since_epoch"] = (
        ((dt - pd.Timestamp("2000-01-01")).dt.days).fillna(-1).astype(np.float32).values
    )
    # Cyclic date encodings
    month_rad = 2.0 * np.pi * dt.dt.month.fillna(1).astype(float) / 12.0
    day_rad = 2.0 * np.pi * dt.dt.day.fillna(1).astype(float) / 31.0
    dow_rad = 2.0 * np.pi * dt.dt.dayofweek.fillna(0).astype(float) / 7.0
    feats[f"{col}_month_sin"] = np.sin(month_rad).astype(np.float32).values
    feats[f"{col}_month_cos"] = np.cos(month_rad).astype(np.float32).values
    feats[f"{col}_day_sin"] = np.sin(day_rad).astype(np.float32).values
    feats[f"{col}_day_cos"] = np.cos(day_rad).astype(np.float32).values
    feats[f"{col}_dow_sin"] = np.sin(dow_rad).astype(np.float32).values
    feats[f"{col}_dow_cos"] = np.cos(dow_rad).astype(np.float32).values
    return pd.DataFrame(feats, index=df.index)


date_cols = []
for c in feature_cols_raw:
    if np.issubdtype(train_features_df[c].dtype, np.datetime64):
        date_cols.append(c)
    elif (
        train_features_df[c].dtype == "object"
        or str(train_features_df[c].dtype) == "category"
    ):
        sample_vals = train_features_df[c].dropna().astype(str).head(50)
        if len(sample_vals) > 0:
            parsed = pd.to_datetime(sample_vals, errors="coerce")
            if parsed.notna().mean() > 0.8 and any(
                re.search(r"date|time|year|period|month|day", c, re.I) for _ in [1]
            ):
                date_cols.append(c)

non_date_cols = [c for c in feature_cols_raw if c not in date_cols]
text_cols = []
cat_cols = []
num_cols = []

for c in non_date_cols:
    if train_features_df[c].dtype in [
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "float32",
        "float64",
    ]:
        num_cols.append(c)
    else:
        avg_len = (
            train_features_df[c].dropna().astype(str).str.len().mean()
            if not train_features_df[c].dropna().empty
            else 0
        )
        nunique = train_features_df[c].nunique(dropna=True)
        if avg_len > 40 or (nunique > 1000 and avg_len > 25):
            text_cols.append(c)
        else:
            cat_cols.append(c)

# 4. CV Scheme Setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_indices = np.zeros(len(train_features_df), dtype=int)
for fold_num, (tr_idx, val_idx) in enumerate(kf.split(train_features_df)):
    fold_indices[val_idx] = fold_num


# 5. Base Feature Engineering
def process_base_features(train_df_raw, test_df_raw, y_tr_target=None, fold_ids=None):
    train_eng = pd.DataFrame(index=train_df_raw.index)
    test_eng = pd.DataFrame(index=test_df_raw.index)

    # 1. Date multi-scale features
    for col in date_cols:
        tr_dt = extract_date_features(train_df_raw, col)
        te_dt = extract_date_features(test_df_raw, col)
        for sub_c in tr_dt.columns:
            train_eng[sub_c] = tr_dt[sub_c]
            test_eng[sub_c] = te_dt[sub_c]

    # 2. Inter-date temporal deltas between all detected date pairs
    for c1, c2 in itertools.combinations(date_cols, 2):
        d1_tr = pd.to_datetime(train_df_raw[c1], errors="coerce")
        d2_tr = pd.to_datetime(train_df_raw[c2], errors="coerce")
        d1_te = pd.to_datetime(test_df_raw[c1], errors="coerce")
        d2_te = pd.to_datetime(test_df_raw[c2], errors="coerce")

        train_eng[f"delta_days_{c1}_{c2}"] = (
            (d1_tr - d2_tr).dt.total_seconds() / 86400.0
        ).fillna(-9999.0).astype(np.float32)
        test_eng[f"delta_days_{c1}_{c2}"] = (
            (d1_te - d2_te).dt.total_seconds() / 86400.0
        ).fillna(-9999.0).astype(np.float32)

    # 3. Numeric features
    for col in num_cols:
        med_val = train_df_raw[col].median()
        train_eng[col] = train_df_raw[col].fillna(med_val).astype(np.float32)
        test_eng[col] = test_df_raw[col].fillna(med_val).astype(np.float32)
        if train_df_raw[col].isna().any():
            train_eng[f"{col}_isna"] = train_df_raw[col].isna().astype(np.float32)
            test_eng[f"{col}_isna"] = test_df_raw[col].isna().astype(np.float32)

    # 4. Categoricals and Categorical Pair Interactions
    for col in cat_cols:
        tr_str = train_df_raw[col].astype(str).fillna("__MISSING__")
        te_str = test_df_raw[col].astype(str).fillna("__MISSING__")

        freq_map = tr_str.value_counts(normalize=True).to_dict()
        train_eng[f"{col}_freq"] = tr_str.map(freq_map).fillna(0.0).astype(np.float32)
        test_eng[f"{col}_freq"] = te_str.map(freq_map).fillna(0.0).astype(np.float32)

        cats = {val: idx for idx, val in enumerate(tr_str.unique())}
        train_eng[f"{col}_encoded"] = tr_str.map(cats).fillna(-1).astype(np.int32)
        test_eng[f"{col}_encoded"] = te_str.map(cats).fillna(-1).astype(np.int32)

    for c1, c2 in itertools.combinations(cat_cols, 2):
        pair_tr = (
            train_df_raw[c1].astype(str).fillna("__MISSING__")
            + "___"
            + train_df_raw[c2].astype(str).fillna("__MISSING__")
        )
        pair_te = (
            test_df_raw[c1].astype(str).fillna("__MISSING__")
            + "___"
            + test_df_raw[c2].astype(str).fillna("__MISSING__")
        )
        pair_freq = pair_tr.value_counts(normalize=True).to_dict()
        train_eng[f"{c1}_{c2}_pair_freq"] = (
            pair_tr.map(pair_freq).fillna(0.0).astype(np.float32)
        )
        test_eng[f"{c1}_{c2}_pair_freq"] = (
            pair_te.map(pair_freq).fillna(0.0).astype(np.float32)
        )

    # 5. Rich Text N-gram TF-IDF, Expanded 25-dim SVD & Leak-Free OOF Ridge Meta-Features
    for col in text_cols:
        tr_text = train_df_raw[col].fillna("").astype(str)
        te_text = test_df_raw[col].fillna("").astype(str)

        train_eng[f"{col}_char_len"] = tr_text.str.len().astype(np.float32)
        test_eng[f"{col}_char_len"] = te_text.str.len().astype(np.float32)
        train_eng[f"{col}_word_count"] = tr_text.apply(lambda x: len(x.split())).astype(
            np.float32
        )
        test_eng[f"{col}_word_count"] = te_text.apply(lambda x: len(x.split())).astype(
            np.float32
        )

        tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            token_pattern=r"(?u)\b\w+\b",
        )
        tfidf_tr = tfidf.fit_transform(tr_text)
        tfidf_te = tfidf.transform(te_text)

        max_comp = min(25, max(2, min(tfidf_tr.shape[0] - 1, tfidf_tr.shape[1] - 1)))
        if max_comp >= 2:
            svd = TruncatedSVD(n_components=max_comp, random_state=42)
            svd_tr = svd.fit_transform(tfidf_tr)
            svd_te = svd.transform(tfidf_te)

            for comp_idx in range(max_comp):
                train_eng[f"{col}_svd_{comp_idx}"] = svd_tr[:, comp_idx].astype(
                    np.float32
                )
                test_eng[f"{col}_svd_{comp_idx}"] = svd_te[:, comp_idx].astype(
                    np.float32
                )

        if y_tr_target is not None and fold_ids is not None:
            oof_ridge = np.zeros(len(train_df_raw), dtype=np.float32)
            test_ridge = np.zeros(len(test_df_raw), dtype=np.float32)
            for f in range(n_splits):
                v_mask = fold_ids == f
                t_mask = ~v_mask
                ridge_model = Ridge(alpha=1.0, random_state=42)
                ridge_model.fit(tfidf_tr[t_mask], y_tr_target.iloc[t_mask])
                oof_ridge[v_mask] = ridge_model.predict(tfidf_tr[v_mask]).astype(np.float32)
                test_ridge += ridge_model.predict(tfidf_te).astype(np.float32) / n_splits
            train_eng[f"{col}_tfidf_ridge_pred"] = oof_ridge
            test_eng[f"{col}_tfidf_ridge_pred"] = test_ridge

    # 6. Group Hierarchical Aggregations (mean, std, min, max, median, relative diffs, ratios)
    agg_numeric_cols = list(num_cols) + [
        f"{c}_days_since_epoch" for c in date_cols if f"{c}_days_since_epoch" in train_eng.columns
    ] + [
        f"{c}_year" for c in date_cols if f"{c}_year" in train_eng.columns
    ]

    for cat_c in cat_cols:
        for num_c in agg_numeric_cols:
            vals_tr = train_eng[num_c] if num_c in train_eng.columns else train_df_raw[num_c].fillna(0)
            vals_te = test_eng[num_c] if num_c in test_eng.columns else test_df_raw[num_c].fillna(0)

            grp_df = pd.DataFrame({
                "cat": train_df_raw[cat_c].astype(str).fillna("__MISSING__"),
                "val": vals_tr
            })
            stats_grp = grp_df.groupby("cat")["val"].agg(
                mean="mean", std="std", min="min", max="max", median="median"
            ).reset_index()

            cat_tr_s = train_df_raw[cat_c].astype(str).fillna("__MISSING__")
            cat_te_s = test_df_raw[cat_c].astype(str).fillna("__MISSING__")

            for stat in ["mean", "std", "min", "max", "median"]:
                stat_map = stats_grp.set_index("cat")[stat].to_dict()
                fallback = float(vals_tr.agg(stat) if not vals_tr.empty else 0.0)
                if np.isnan(fallback):
                    fallback = 0.0

                s_tr_mapped = cat_tr_s.map(stat_map).fillna(fallback).astype(np.float32)
                s_te_mapped = cat_te_s.map(stat_map).fillna(fallback).astype(np.float32)

                train_eng[f"{cat_c}_{num_c}_{stat}"] = s_tr_mapped
                test_eng[f"{cat_c}_{num_c}_{stat}"] = s_te_mapped

            # Relative differences and ratios
            train_eng[f"{cat_c}_{num_c}_diff_mean"] = vals_tr.astype(np.float32) - train_eng[f"{cat_c}_{num_c}_mean"]
            test_eng[f"{cat_c}_{num_c}_diff_mean"] = vals_te.astype(np.float32) - test_eng[f"{cat_c}_{num_c}_mean"]

            train_eng[f"{cat_c}_{num_c}_diff_median"] = vals_tr.astype(np.float32) - train_eng[f"{cat_c}_{num_c}_median"]
            test_eng[f"{cat_c}_{num_c}_diff_median"] = vals_te.astype(np.float32) - test_eng[f"{cat_c}_{num_c}_median"]

            train_eng[f"{cat_c}_{num_c}_ratio_mean"] = vals_tr.astype(np.float32) / (train_eng[f"{cat_c}_{num_c}_mean"].abs() + 1e-5)
            test_eng[f"{cat_c}_{num_c}_ratio_mean"] = vals_te.astype(np.float32) / (test_eng[f"{cat_c}_{num_c}_mean"].abs() + 1e-5)

    return train_eng, test_eng


train_features_processed, test_features_processed = process_base_features(
    train_features_df, test_features_df, y_train, fold_indices
)

# 6. Leak-Free Target Encodings with Cross-Group Interaction Statistics
key_cat_cols = [c for c in cat_cols if train_features_df[c].nunique() >= 2]
pair_cat_items = []
for c1, c2 in itertools.combinations(key_cat_cols, 2):
    pair_name = f"{c1}_x_{c2}"
    s_tr = (
        train_features_df[c1].astype(str).fillna("__MISSING__")
        + "___"
        + train_features_df[c2].astype(str).fillna("__MISSING__")
    )
    s_te = (
        test_features_df[c1].astype(str).fillna("__MISSING__")
        + "___"
        + test_features_df[c2].astype(str).fillna("__MISSING__")
    )
    pair_cat_items.append((pair_name, s_tr, s_te))

all_target_enc_cols = [
    (
        c,
        train_features_df[c].astype(str).fillna("__MISSING__"),
        test_features_df[c].astype(str).fillna("__MISSING__"),
    )
    for c in key_cat_cols
] + pair_cat_items

smoothing_weight = 10.0
global_target_mean = float(y_train.mean())
global_target_std = float(y_train.std())
global_target_median = float(y_train.median())

for col_name, s_tr, s_te in all_target_enc_cols:
    oof_means = np.zeros(len(train_features_df), dtype=np.float32)
    oof_stds = np.zeros(len(train_features_df), dtype=np.float32)
    oof_medians = np.zeros(len(train_features_df), dtype=np.float32)
    oof_counts = np.zeros(len(train_features_df), dtype=np.float32)

    for fold in range(n_splits):
        val_mask = fold_indices == fold
        tr_mask = ~val_mask

        tr_subset = pd.DataFrame(
            {
                "cat": s_tr.loc[tr_mask],
                "target": y_train.loc[tr_mask],
            }
        )

        stats = tr_subset.groupby("cat")["target"].agg(
            count="count", mean="mean", std="std", median="median"
        )
        stats["std"] = stats["std"].fillna(0.0)
        stats["smooth_mean"] = (
            stats["count"] * stats["mean"] + smoothing_weight * global_target_mean
        ) / (stats["count"] + smoothing_weight)
        stats["smooth_std"] = (
            stats["count"] * stats["std"] + smoothing_weight * global_target_std
        ) / (stats["count"] + smoothing_weight)
        stats["smooth_median"] = (
            stats["count"] * stats["median"] + smoothing_weight * global_target_median
        ) / (stats["count"] + smoothing_weight)

        val_cats = s_tr.loc[val_mask]
        oof_means[val_mask] = (
            val_cats.map(stats["smooth_mean"]).fillna(global_target_mean).values
        )
        oof_stds[val_mask] = (
            val_cats.map(stats["smooth_std"]).fillna(global_target_std).values
        )
        oof_medians[val_mask] = (
            val_cats.map(stats["smooth_median"]).fillna(global_target_median).values
        )
        oof_counts[val_mask] = val_cats.map(stats["count"]).fillna(0).values

    train_features_processed[f"{col_name}_target_enc_mean"] = oof_means
    train_features_processed[f"{col_name}_target_enc_std"] = oof_stds
    train_features_processed[f"{col_name}_target_enc_median"] = oof_medians
    train_features_processed[f"{col_name}_target_enc_count"] = oof_counts

    full_subset = pd.DataFrame(
        {
            "cat": s_tr,
            "target": y_train,
        }
    )
    full_stats = full_subset.groupby("cat")["target"].agg(
        count="count", mean="mean", std="std", median="median"
    )
    full_stats["std"] = full_stats["std"].fillna(0.0)
    full_stats["smooth_mean"] = (
        full_stats["count"] * full_stats["mean"] + smoothing_weight * global_target_mean
    ) / (full_stats["count"] + smoothing_weight)
    full_stats["smooth_std"] = (
        full_stats["count"] * full_stats["std"] + smoothing_weight * global_target_std
    ) / (full_stats["count"] + smoothing_weight)
    full_stats["smooth_median"] = (
        full_stats["count"] * full_stats["median"] + smoothing_weight * global_target_median
    ) / (full_stats["count"] + smoothing_weight)

    test_features_processed[f"{col_name}_target_enc_mean"] = (
        s_te.map(full_stats["smooth_mean"])
        .fillna(global_target_mean)
        .astype(np.float32)
        .values
    )
    test_features_processed[f"{col_name}_target_enc_std"] = (
        s_te.map(full_stats["smooth_std"])
        .fillna(global_target_std)
        .astype(np.float32)
        .values
    )
    test_features_processed[f"{col_name}_target_enc_median"] = (
        s_te.map(full_stats["smooth_median"])
        .fillna(global_target_median)
        .astype(np.float32)
        .values
    )
    test_features_processed[f"{col_name}_target_enc_count"] = (
        s_te.map(full_stats["count"]).fillna(0).astype(np.float32).values
    )

# Sanitize feature column names for gradient boosting algorithms
clean_col_names = [
    re.sub(r"[\[\]<>\s,]", "_", str(c)) for c in train_features_processed.columns
]
train_features_processed.columns = clean_col_names
test_features_processed.columns = clean_col_names

train_features_processed["fold"] = fold_indices
train_features_processed[target_col] = y_train.values

train_features_processed.to_parquet("./working/processed_train.parquet", index=False)
test_features_processed.to_parquet("./working/processed_test.parquet", index=False)


# 7. Model Builders
class TabularMLP(nn.Module):
    def __init__(self, in_features, hidden_dims=[256, 128, 64], dropout=0.2):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.down1 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.down2 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.LayerNorm(hidden_dims[2]),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dims[2], 1)

    def forward(self, x):
        x0 = self.in_proj(x)
        x1 = x0 + self.block1(x0)
        x2 = self.down1(x1)
        x3 = x2 + self.block2(x2)
        x4 = self.down2(x3)
        out = self.head(x4)
        return out.squeeze(-1)


def get_lgb_model(params_override=None):
    default_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 2000,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if params_override:
        default_params.update(params_override)
    return lgb.LGBMRegressor(**default_params)


def get_xgb_model(params_override=None):
    default_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 2000,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
    }
    if params_override:
        default_params.update(params_override)
    return xgb.XGBRegressor(**default_params)


def get_cat_model(params_override=None):
    default_params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": False,
        "early_stopping_rounds": 50,
    }
    if params_override:
        default_params.update(params_override)
    return CatBoostRegressor(**default_params)


# 8. Training and Evaluation Pipeline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_cols = [
    c for c in train_features_processed.columns if c not in [target_col, "fold"]
]
X_test = test_features_processed[feature_cols].copy()

oof_lgb_preds = np.zeros(len(train_features_processed), dtype=np.float64)
oof_xgb_preds = np.zeros(len(train_features_processed), dtype=np.float64)
oof_cat_preds = np.zeros(len(train_features_processed), dtype=np.float64)
oof_mlp_preds = np.zeros(len(train_features_processed), dtype=np.float64)

test_lgb_preds = np.zeros(len(test_features_processed), dtype=np.float64)
test_xgb_preds = np.zeros(len(test_features_processed), dtype=np.float64)
test_cat_preds = np.zeros(len(test_features_processed), dtype=np.float64)
test_mlp_preds = np.zeros(len(test_features_processed), dtype=np.float64)

min_target_val = float(train_features_processed[target_col].min())

for fold in range(n_splits):
    val_mask = train_features_processed["fold"] == fold
    tr_mask = ~val_mask

    X_tr = train_features_processed.loc[tr_mask, feature_cols]
    y_tr = train_features_processed.loc[tr_mask, target_col].values

    X_va = train_features_processed.loc[val_mask, feature_cols]
    y_va = train_features_processed.loc[val_mask, target_col].values

    # 1. LightGBM
    lgb_clf = get_lgb_model()
    lgb_clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    val_lgb_pred = lgb_clf.predict(X_va)
    oof_lgb_preds[val_mask] = val_lgb_pred
    test_lgb_preds += lgb_clf.predict(X_test) / n_splits

    # 2. XGBoost
    xgb_clf = get_xgb_model()
    xgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    val_xgb_pred = xgb_clf.predict(X_va)
    oof_xgb_preds[val_mask] = val_xgb_pred
    test_xgb_preds += xgb_clf.predict(X_test) / n_splits

    # 3. CatBoost
    cat_clf = get_cat_model()
    cat_clf.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        use_best_model=True,
    )
    val_cat_pred = cat_clf.predict(X_va)
    oof_cat_preds[val_mask] = val_cat_pred
    test_cat_preds += cat_clf.predict(X_test) / n_splits

    # 4. Tabular MLP
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy="median")

    X_tr_np = imputer.fit_transform(X_tr.replace([np.inf, -np.inf], np.nan))
    X_va_np = imputer.transform(X_va.replace([np.inf, -np.inf], np.nan))
    X_te_np = imputer.transform(X_test.replace([np.inf, -np.inf], np.nan))

    X_tr_scaled = scaler.fit_transform(X_tr_np).astype(np.float32)
    X_va_scaled = scaler.transform(X_va_np).astype(np.float32)
    X_te_scaled = scaler.transform(X_te_np).astype(np.float32)

    tr_dataset = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    va_dataset = TensorDataset(
        torch.tensor(X_va_scaled, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32),
    )
    tr_loader = DataLoader(tr_dataset, batch_size=128, shuffle=True)
    va_loader = DataLoader(va_dataset, batch_size=256, shuffle=False)

    mlp_model = TabularMLP(in_features=X_tr_scaled.shape[1]).to(device)
    optimizer = torch.optim.AdamW(mlp_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(40):
        mlp_model.train()
        for batch_x, batch_y in tr_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = mlp_model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        mlp_model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in va_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = mlp_model(batch_x)
                val_losses.append(criterion(pred, batch_y).item())
        epoch_val_loss = np.mean(val_losses)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = {k: v.cpu().clone() for k, v in mlp_model.state_dict().items()}

    mlp_model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    mlp_model.eval()
    with torch.no_grad():
        val_mlp_pred = (
            mlp_model(torch.tensor(X_va_scaled, dtype=torch.float32).to(device))
            .cpu()
            .numpy()
        )
        test_fold_mlp_pred = (
            mlp_model(torch.tensor(X_te_scaled, dtype=torch.float32).to(device))
            .cpu()
            .numpy()
        )

    oof_mlp_preds[val_mask] = val_mlp_pred
    test_mlp_preds += test_fold_mlp_pred / n_splits

    fold_rmse = np.sqrt(
        mean_squared_error(
            y_va, (val_lgb_pred + val_xgb_pred + val_cat_pred + val_mlp_pred) / 4.0
        )
    )
    print(f"Fold {fold + 1}/{n_splits} Validation RMSE: {fold_rmse:.4f}")

# 9. Optimal Out-of-Fold Blending
y_true = train_features_processed[target_col].values


def ensemble_loss(weights):
    w1, w2, w3, w4 = weights[0], weights[1], weights[2], weights[3]
    blend = w1 * oof_lgb_preds + w2 * oof_xgb_preds + w3 * oof_cat_preds + w4 * oof_mlp_preds
    if min_target_val >= 0:
        blend = np.clip(blend, a_min=0.0, a_max=None)
    return np.sqrt(mean_squared_error(y_true, blend))


res = minimize(
    ensemble_loss,
    [0.25, 0.25, 0.25, 0.25],
    method="SLSQP",
    bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    constraints={"type": "eq", "fun": lambda w: 1.0 - (w[0] + w[1] + w[2] + w[3])},
)
w_lgb, w_xgb, w_cat, w_mlp = res.x[0], res.x[1], res.x[2], res.x[3]
print(
    f"Optimal Ensemble Weights -> LGB: {w_lgb:.4f}, XGB: {w_xgb:.4f}, CAT: {w_cat:.4f}, MLP: {w_mlp:.4f}"
)

final_oof_preds = (
    w_lgb * oof_lgb_preds
    + w_xgb * oof_xgb_preds
    + w_cat * oof_cat_preds
    + w_mlp * oof_mlp_preds
)
if min_target_val >= 0:
    final_oof_preds = np.clip(final_oof_preds, a_min=0.0, a_max=None)

final_score = float(np.sqrt(mean_squared_error(y_true, final_oof_preds)))

# 10. Generate Test Predictions & Save Submission
final_test_preds = (
    w_lgb * test_lgb_preds
    + w_xgb * test_xgb_preds
    + w_cat * test_cat_preds
    + w_mlp * test_mlp_preds
)
if min_target_val >= 0:
    final_test_preds = np.clip(final_test_preds, a_min=0.0, a_max=None)

submission_df = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "Amount in USD": final_test_preds}
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Submission saved: {submission_df.shape} rows.")
print(f"Final Validation Score: {final_score}")
