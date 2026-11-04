import json
import os
from typing import Any, Dict, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

try:
    import catboost as cb

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# ---------------------------------------------------------
# 1. Configuration & Directories
# ---------------------------------------------------------
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(INPUT_DIR, "train.parquet")
TEST_PATH = os.path.join(INPUT_DIR, "test.parquet")
TARGET_COL = "State Rate"
N_SPLITS = 5
RANDOM_SEED = 42

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------
# 2. Data Loading & Feature Engineering
# ---------------------------------------------------------
print("Loading raw datasets...")
train_df = pd.read_parquet(TRAIN_PATH).copy()
test_df = pd.read_parquet(TEST_PATH).copy()

assert (
    TARGET_COL in train_df.columns
), f"Target column '{TARGET_COL}' missing from train dataset!"

y_train = train_df[TARGET_COL].astype(float).values
train_df["target"] = y_train

if "id" not in test_df.columns:
    test_df["id"] = np.arange(len(test_df), dtype=int)
if "id" not in train_df.columns:
    train_df["id"] = np.arange(len(train_df), dtype=int)

# Detect Feature Modalities
all_cols = [c for c in train_df.columns if c not in [TARGET_COL, "target", "id"]]
numeric_cols = []
categorical_cols = []
text_cols = []

for c in all_cols:
    if c not in test_df.columns:
        continue
    if pd.api.types.is_numeric_dtype(train_df[c]):
        numeric_cols.append(c)
    else:
        non_null_strings = train_df[c].dropna().astype(str)
        if len(non_null_strings) > 0:
            avg_word_count = non_null_strings.apply(lambda x: len(x.split())).mean()
            max_word_count = non_null_strings.apply(lambda x: len(x.split())).max()
            if avg_word_count > 4 or max_word_count > 10:
                text_cols.append(c)
            else:
                categorical_cols.append(c)
        else:
            categorical_cols.append(c)

# Stratified K-Fold Setup
num_bins = min(10, len(np.unique(y_train)))
strat_bins = pd.qcut(y_train, q=num_bins, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

train_df["fold"] = -1
for fold_idx, (_, val_idx) in enumerate(skf.split(train_df, strat_bins)):
    train_df.loc[val_idx, "fold"] = fold_idx

# Expanded Multi-Scale Text Features & Metadata
text_feature_dfs_train = []
text_feature_dfs_test = []

for txt_col in text_cols:
    train_txt = train_df[txt_col].fillna("missing").astype(str)
    test_txt = test_df[txt_col].fillna("missing").astype(str)

    # Word n-gram TF-IDF
    tfidf_word = TfidfVectorizer(
        max_features=500,
        stop_words="english",
        ngram_range=(1, 3),
        sublinear_tf=True,
    )
    train_vecs_w = tfidf_word.fit_transform(train_txt)
    test_vecs_w = tfidf_word.transform(test_txt)

    n_comp_w = min(40, train_vecs_w.shape[1] - 1)
    if n_comp_w > 1:
        svd_w = TruncatedSVD(n_components=n_comp_w, random_state=RANDOM_SEED)
        train_svd_w = svd_w.fit_transform(train_vecs_w)
        test_svd_w = svd_w.transform(test_vecs_w)
        w_cols = [f"{txt_col}_word_svd_{i}" for i in range(n_comp_w)]
        text_feature_dfs_train.append(pd.DataFrame(train_svd_w, columns=w_cols, index=train_df.index))
        text_feature_dfs_test.append(pd.DataFrame(test_svd_w, columns=w_cols, index=test_df.index))

    # Char n-gram TF-IDF
    tfidf_char = TfidfVectorizer(
        max_features=500,
        analyzer="char",
        ngram_range=(2, 5),
        sublinear_tf=True,
    )
    train_vecs_c = tfidf_char.fit_transform(train_txt)
    test_vecs_c = tfidf_char.transform(test_txt)

    n_comp_c = min(20, train_vecs_c.shape[1] - 1)
    if n_comp_c > 1:
        svd_c = TruncatedSVD(n_components=n_comp_c, random_state=RANDOM_SEED + 1)
        train_svd_c = svd_c.fit_transform(train_vecs_c)
        test_svd_c = svd_c.transform(test_vecs_c)
        c_cols = [f"{txt_col}_char_svd_{i}" for i in range(n_comp_c)]
        text_feature_dfs_train.append(pd.DataFrame(train_svd_c, columns=c_cols, index=train_df.index))
        text_feature_dfs_test.append(pd.DataFrame(test_svd_c, columns=c_cols, index=test_df.index))

    # Text Metadata Features
    meta_tr = pd.DataFrame(index=train_df.index)
    meta_te = pd.DataFrame(index=test_df.index)
    meta_tr[f"{txt_col}_len"] = train_txt.apply(len).astype(float)
    meta_te[f"{txt_col}_len"] = test_txt.apply(len).astype(float)
    meta_tr[f"{txt_col}_words"] = train_txt.apply(lambda s: len(s.split())).astype(float)
    meta_te[f"{txt_col}_words"] = test_txt.apply(lambda s: len(s.split())).astype(float)
    meta_tr[f"{txt_col}_upper_ratio"] = train_txt.apply(lambda s: sum(1 for c in s if c.isupper()) / (len(s) + 1e-5)).astype(float)
    meta_te[f"{txt_col}_upper_ratio"] = test_txt.apply(lambda s: sum(1 for c in s if c.isupper()) / (len(s) + 1e-5)).astype(float)
    meta_tr[f"{txt_col}_digit_cnt"] = train_txt.apply(lambda s: sum(1 for c in s if c.isdigit())).astype(float)
    meta_te[f"{txt_col}_digit_cnt"] = test_txt.apply(lambda s: sum(1 for c in s if c.isdigit())).astype(float)
    text_feature_dfs_train.append(meta_tr)
    text_feature_dfs_test.append(meta_te)

train_engineered = pd.DataFrame(index=train_df.index)
test_engineered = pd.DataFrame(index=test_df.index)

# Categorical Frequency Encodings
for cat_col in categorical_cols:
    freq_map = train_df[cat_col].value_counts(normalize=True).to_dict()
    train_engineered[f"{cat_col}_freq"] = (
        train_df[cat_col].map(freq_map).fillna(0.0).astype(float)
    )
    test_engineered[f"{cat_col}_freq"] = (
        test_df[cat_col].map(freq_map).fillna(0.0).astype(float)
    )

# Higher-Order Categorical Pairwise Interactions
pair_cols = []
if len(categorical_cols) >= 2:
    primary_cats = categorical_cols[:6]
    for i in range(len(primary_cats)):
        for j in range(i + 1, len(primary_cats)):
            c1, c2 = primary_cats[i], primary_cats[j]
            pair_name = f"{c1}_x_{c2}"
            pair_cols.append(pair_name)
            train_pair = train_df[c1].astype(str) + "__" + train_df[c2].astype(str)
            test_pair = test_df[c1].astype(str) + "__" + test_df[c2].astype(str)
            train_df[pair_name] = train_pair
            test_df[pair_name] = test_pair

            pair_freq = train_pair.value_counts(normalize=True).to_dict()
            train_engineered[f"{pair_name}_freq"] = (
                train_pair.map(pair_freq).fillna(0.0).astype(float)
            )
            test_engineered[f"{pair_name}_freq"] = (
                test_pair.map(pair_freq).fillna(0.0).astype(float)
            )

# Out-Of-Fold Hierarchical Empirical Bayes Target Aggregations
global_mean = float(y_train.mean())
global_std = float(y_train.std())
global_median = float(np.median(y_train))
global_min = float(np.min(y_train))
global_max = float(np.max(y_train))
smoothing_weight = 10.0
target_enc_cols = list(categorical_cols) + pair_cols

for cat_col in target_enc_cols:
    te_mean_train_oof = np.zeros(len(train_df), dtype=float)
    te_std_train_oof = np.zeros(len(train_df), dtype=float)
    te_med_train_oof = np.zeros(len(train_df), dtype=float)
    te_min_train_oof = np.zeros(len(train_df), dtype=float)
    te_max_train_oof = np.zeros(len(train_df), dtype=float)

    for f in range(N_SPLITS):
        trn_mask = train_df["fold"] != f
        val_mask = train_df["fold"] == f

        grp = train_df[trn_mask].groupby(cat_col)["target"]
        stats = grp.agg(["count", "mean", "std", "median", "min", "max"])
        stats["std"] = stats["std"].fillna(0.0)

        smooth_mean = (
            stats["count"] * stats["mean"] + smoothing_weight * global_mean
        ) / (stats["count"] + smoothing_weight)
        smooth_std = (
            stats["count"] * stats["std"] + smoothing_weight * global_std
        ) / (stats["count"] + smoothing_weight)

        te_mean_map = smooth_mean.to_dict()
        te_std_map = smooth_std.to_dict()
        te_med_map = stats["median"].to_dict()
        te_min_map = stats["min"].to_dict()
        te_max_map = stats["max"].to_dict()

        te_mean_train_oof[val_mask] = train_df.loc[val_mask, cat_col].map(te_mean_map).fillna(global_mean).values
        te_std_train_oof[val_mask] = train_df.loc[val_mask, cat_col].map(te_std_map).fillna(global_std).values
        te_med_train_oof[val_mask] = train_df.loc[val_mask, cat_col].map(te_med_map).fillna(global_median).values
        te_min_train_oof[val_mask] = train_df.loc[val_mask, cat_col].map(te_min_map).fillna(global_min).values
        te_max_train_oof[val_mask] = train_df.loc[val_mask, cat_col].map(te_max_map).fillna(global_max).values

    full_grp = train_df.groupby(cat_col)["target"]
    full_stats = full_grp.agg(["count", "mean", "std", "median", "min", "max"])
    full_stats["std"] = full_stats["std"].fillna(0.0)

    full_smooth_mean = (
        full_stats["count"] * full_stats["mean"] + smoothing_weight * global_mean
    ) / (full_stats["count"] + smoothing_weight)
    full_smooth_std = (
        full_stats["count"] * full_stats["std"] + smoothing_weight * global_std
    ) / (full_stats["count"] + smoothing_weight)

    full_mean_map = full_smooth_mean.to_dict()
    full_std_map = full_smooth_std.to_dict()
    full_med_map = full_stats["median"].to_dict()
    full_min_map = full_stats["min"].to_dict()
    full_max_map = full_stats["max"].to_dict()

    train_engineered[f"{cat_col}_te_mean"] = te_mean_train_oof
    train_engineered[f"{cat_col}_te_std"] = te_std_train_oof
    train_engineered[f"{cat_col}_te_med"] = te_med_train_oof
    train_engineered[f"{cat_col}_te_min"] = te_min_train_oof
    train_engineered[f"{cat_col}_te_max"] = te_max_train_oof
    train_engineered[f"{cat_col}_te_diff_global"] = te_mean_train_oof - global_mean

    test_engineered[f"{cat_col}_te_mean"] = test_df[cat_col].map(full_mean_map).fillna(global_mean).values
    test_engineered[f"{cat_col}_te_std"] = test_df[cat_col].map(full_std_map).fillna(global_std).values
    test_engineered[f"{cat_col}_te_med"] = test_df[cat_col].map(full_med_map).fillna(global_median).values
    test_engineered[f"{cat_col}_te_min"] = test_df[cat_col].map(full_min_map).fillna(global_min).values
    test_engineered[f"{cat_col}_te_max"] = test_df[cat_col].map(full_max_map).fillna(global_max).values
    test_engineered[f"{cat_col}_te_diff_global"] = test_engineered[f"{cat_col}_te_mean"] - global_mean

# Comprehensive Multi-Group Numeric Aggregations across Key Categorical Columns
grouping_cols = [c for c in categorical_cols[:5] if c in train_df.columns]
for group_col in grouping_cols:
    for num_col in numeric_cols:
        grp_stats = (
            train_df.groupby(group_col)[num_col]
            .agg(["mean", "std", "median", "min", "max"])
            .reset_index()
        )
        col_prefix = f"{num_col}_by_{group_col}"
        grp_stats.columns = [
            group_col,
            f"{col_prefix}_mean",
            f"{col_prefix}_std",
            f"{col_prefix}_median",
            f"{col_prefix}_min",
            f"{col_prefix}_max",
        ]
        grp_stats[f"{col_prefix}_std"] = grp_stats[f"{col_prefix}_std"].fillna(0.0)

        num_mean = float(train_df[num_col].mean())
        num_med = float(train_df[num_col].median())
        num_min = float(train_df[num_col].min())
        num_max = float(train_df[num_col].max())

        train_m = train_df[[group_col, num_col]].merge(grp_stats, on=group_col, how="left")
        test_m = test_df[[group_col, num_col]].merge(grp_stats, on=group_col, how="left")

        train_engineered[f"{col_prefix}_mean"] = train_m[f"{col_prefix}_mean"].fillna(num_mean)
        test_engineered[f"{col_prefix}_mean"] = test_m[f"{col_prefix}_mean"].fillna(num_mean)

        train_engineered[f"{col_prefix}_std"] = train_m[f"{col_prefix}_std"].fillna(0.0)
        test_engineered[f"{col_prefix}_std"] = test_m[f"{col_prefix}_std"].fillna(0.0)

        train_engineered[f"{col_prefix}_min"] = train_m[f"{col_prefix}_min"].fillna(num_min)
        test_engineered[f"{col_prefix}_min"] = test_m[f"{col_prefix}_min"].fillna(num_min)

        train_engineered[f"{col_prefix}_max"] = train_m[f"{col_prefix}_max"].fillna(num_max)
        test_engineered[f"{col_prefix}_max"] = test_m[f"{col_prefix}_max"].fillna(num_max)

        train_engineered[f"{col_prefix}_diff_mean"] = (train_m[num_col] - train_m[f"{col_prefix}_mean"]).fillna(0.0)
        test_engineered[f"{col_prefix}_diff_mean"] = (test_m[num_col] - test_m[f"{col_prefix}_mean"]).fillna(0.0)

        train_engineered[f"{col_prefix}_ratio_mean"] = (train_m[num_col] / (train_m[f"{col_prefix}_mean"].abs() + 1e-5)).fillna(1.0)
        test_engineered[f"{col_prefix}_ratio_mean"] = (test_m[num_col] / (test_m[f"{col_prefix}_mean"].abs() + 1e-5)).fillna(1.0)

        train_engineered[f"{col_prefix}_zscore"] = (
            (train_m[num_col] - train_m[f"{col_prefix}_mean"]) / (train_m[f"{col_prefix}_std"] + 1e-5)
        ).fillna(0.0)
        test_engineered[f"{col_prefix}_zscore"] = (
            (test_m[num_col] - test_m[f"{col_prefix}_mean"]) / (test_m[f"{col_prefix}_std"] + 1e-5)
        ).fillna(0.0)

# Numeric Features, Missing Indicators & Non-linear Transforms
for num_col in numeric_cols:
    train_engineered[f"{num_col}_isna"] = train_df[num_col].isna().astype(float)
    test_engineered[f"{num_col}_isna"] = test_df[num_col].isna().astype(float)

    med_val = train_df[num_col].median()
    if pd.isna(med_val):
        med_val = 0.0
    train_engineered[num_col] = train_df[num_col].fillna(med_val).astype(float)
    test_engineered[num_col] = test_df[num_col].fillna(med_val).astype(float)

    if (train_engineered[num_col] >= 0).all() and (test_engineered[num_col] >= 0).all():
        train_engineered[f"{num_col}_log1p"] = np.log1p(train_engineered[num_col])
        test_engineered[f"{num_col}_log1p"] = np.log1p(test_engineered[num_col])

# Pairwise Numeric Interaction Features
if len(numeric_cols) >= 2:
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            n1, n2 = numeric_cols[i], numeric_cols[j]
            v1_tr = train_engineered[n1].values
            v2_tr = train_engineered[n2].values
            v1_te = test_engineered[n1].values
            v2_te = test_engineered[n2].values

            train_engineered[f"{n1}_mult_{n2}"] = v1_tr * v2_tr
            test_engineered[f"{n1}_mult_{n2}"] = v1_te * v2_te

            train_engineered[f"{n1}_div_{n2}"] = v1_tr / (np.abs(v2_tr) + 1e-5)
            test_engineered[f"{n1}_div_{n2}"] = v1_te / (np.abs(v2_te) + 1e-5)

            train_engineered[f"{n1}_diff_{n2}"] = v1_tr - v2_tr
            test_engineered[f"{n1}_diff_{n2}"] = v1_te - v2_te

# Categorical Label Encodings
for cat_col in categorical_cols:
    train_cat_series = train_df[cat_col].fillna("__MISSING__").astype(str)
    test_cat_series = test_df[cat_col].fillna("__MISSING__").astype(str)

    unique_categories = {val: idx for idx, val in enumerate(train_cat_series.unique())}
    train_engineered[f"{cat_col}_code"] = (
        train_cat_series.map(unique_categories).fillna(-1).astype(int)
    )
    test_engineered[f"{cat_col}_code"] = (
        test_cat_series.map(unique_categories).fillna(-1).astype(int)
    )

# Combine with Text SVD features
if len(text_feature_dfs_train) > 0:
    for text_df_tr, text_df_te in zip(text_feature_dfs_train, text_feature_dfs_test):
        train_engineered = pd.concat([train_engineered, text_df_tr], axis=1)
        test_engineered = pd.concat([test_engineered, text_df_te], axis=1)

feature_names = list(train_engineered.columns)
train_engineered = (
    train_engineered[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
)
test_engineered = (
    test_engineered[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
)

X = train_engineered.values
y = y_train
folds = train_df["fold"].values
X_test_raw = test_engineered.values
test_ids = test_df["id"].values

n_train = len(train_df)
n_test = len(test_df)
print(f"Features prepared: {len(feature_names)} columns across {n_train} samples.")

# ---------------------------------------------------------
# 3. Model Architectures & Builders
# ---------------------------------------------------------


def get_lgbm_default_params(random_seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 3500,
        "learning_rate": 0.03,
        "num_leaves": 45,
        "max_depth": 8,
        "min_child_samples": 20,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.2,
        "reg_lambda": 1.5,
        "random_state": random_seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_lgbm_deep_params(random_seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 3500,
        "learning_rate": 0.02,
        "num_leaves": 95,
        "max_depth": 10,
        "min_child_samples": 15,
        "subsample": 0.75,
        "subsample_freq": 1,
        "colsample_bytree": 0.65,
        "reg_alpha": 0.5,
        "reg_lambda": 3.0,
        "random_state": random_seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_default_params(random_seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 3500,
        "learning_rate": 0.03,
        "max_depth": 7,
        "min_child_weight": 4,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.3,
        "reg_lambda": 2.0,
        "random_state": random_seed,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def get_catboost_1_params(random_seed: int = 42) -> Dict[str, Any]:
    return {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 3500,
        "learning_rate": 0.04,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": random_seed,
        "verbose": False,
    }


def get_catboost_2_params(random_seed: int = 42) -> Dict[str, Any]:
    return {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 3500,
        "learning_rate": 0.025,
        "depth": 8,
        "l2_leaf_reg": 6.0,
        "random_seed": random_seed,
        "verbose": False,
    }


# ---------------------------------------------------------
# 4. Cross-Validation & Model Training Loop
# ---------------------------------------------------------
oof_lgb = np.zeros(n_train, dtype=np.float64)
oof_lgb_deep = np.zeros(n_train, dtype=np.float64)
oof_xgb = np.zeros(n_train, dtype=np.float64)
oof_cb1 = np.zeros(n_train, dtype=np.float64)
oof_cb2 = np.zeros(n_train, dtype=np.float64)

test_preds_lgb = np.zeros(n_test, dtype=np.float64)
test_preds_lgb_deep = np.zeros(n_test, dtype=np.float64)
test_preds_xgb = np.zeros(n_test, dtype=np.float64)
test_preds_cb1 = np.zeros(n_test, dtype=np.float64)
test_preds_cb2 = np.zeros(n_test, dtype=np.float64)

for fold_idx in range(N_SPLITS):
    trn_mask = folds != fold_idx
    val_mask = folds == fold_idx

    X_train_fold, y_train_fold = X[trn_mask], y[trn_mask]
    X_val_fold, y_val_fold = X[val_mask], y[val_mask]

    # --- LightGBM 1 (Standard) ---
    lgb_params = get_lgbm_default_params(random_seed=RANDOM_SEED + fold_idx)
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(-1)],
    )
    val_pred_lgb = lgb_model.predict(X_val_fold)
    oof_lgb[val_mask] = val_pred_lgb
    test_preds_lgb += lgb_model.predict(X_test_raw) / N_SPLITS

    # --- LightGBM 2 (Deep / Distinct Regularization) ---
    lgb_deep_params = get_lgbm_deep_params(random_seed=RANDOM_SEED + 100 + fold_idx)
    lgb_deep_model = lgb.LGBMRegressor(**lgb_deep_params)
    lgb_deep_model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(-1)],
    )
    val_pred_lgb_deep = lgb_deep_model.predict(X_val_fold)
    oof_lgb_deep[val_mask] = val_pred_lgb_deep
    test_preds_lgb_deep += lgb_deep_model.predict(X_test_raw) / N_SPLITS

    # --- XGBoost ---
    xgb_params = get_xgb_default_params(random_seed=RANDOM_SEED + fold_idx)
    xgb_params["early_stopping_rounds"] = 60
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        verbose=False,
    )
    val_pred_xgb = xgb_model.predict(X_val_fold)
    oof_xgb[val_mask] = val_pred_xgb
    test_preds_xgb += xgb_model.predict(X_test_raw) / N_SPLITS

    # --- CatBoost 1 (Depth 6, Fast Learning Rate) ---
    if HAS_CATBOOST:
        cb1_params = get_catboost_1_params(random_seed=RANDOM_SEED + fold_idx)
        cb1_model = cb.CatBoostRegressor(**cb1_params)
        cb1_model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=(X_val_fold, y_val_fold),
            early_stopping_rounds=60,
            verbose=False,
        )
        val_pred_cb1 = cb1_model.predict(X_val_fold)
        oof_cb1[val_mask] = val_pred_cb1
        test_preds_cb1 += cb1_model.predict(X_test_raw) / N_SPLITS

        # --- CatBoost 2 (Depth 8, L2 Leaf Regularization) ---
        cb2_params = get_catboost_2_params(random_seed=RANDOM_SEED + 200 + fold_idx)
        cb2_model = cb.CatBoostRegressor(**cb2_params)
        cb2_model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=(X_val_fold, y_val_fold),
            early_stopping_rounds=60,
            verbose=False,
        )
        val_pred_cb2 = cb2_model.predict(X_val_fold)
        oof_cb2[val_mask] = val_pred_cb2
        test_preds_cb2 += cb2_model.predict(X_test_raw) / N_SPLITS
    else:
        val_pred_cb1 = np.zeros_like(val_pred_lgb)
        val_pred_cb2 = np.zeros_like(val_pred_lgb)

    fold_rmse_lgb = root_mean_squared_error(y_val_fold, val_pred_lgb)
    fold_rmse_lgb_deep = root_mean_squared_error(y_val_fold, val_pred_lgb_deep)
    fold_rmse_xgb = root_mean_squared_error(y_val_fold, val_pred_xgb)
    fold_rmse_cb1 = (
        root_mean_squared_error(y_val_fold, val_pred_cb1) if HAS_CATBOOST else 0.0
    )
    fold_rmse_cb2 = (
        root_mean_squared_error(y_val_fold, val_pred_cb2) if HAS_CATBOOST else 0.0
    )
    print(
        f"Fold {fold_idx + 1}/{N_SPLITS} | LGB1: {fold_rmse_lgb:.4f} | LGB2: {fold_rmse_lgb_deep:.4f} | XGB: {fold_rmse_xgb:.4f} | CB1: {fold_rmse_cb1:.4f} | CB2: {fold_rmse_cb2:.4f}"
    )

# ---------------------------------------------------------
# 5. Out-Of-Fold Evaluation & Ensemble Weight Optimization
# ---------------------------------------------------------
min_bound = min(0.0, float(np.min(y_train)))
max_bound = max(100.0, float(np.max(y_train)))

if HAS_CATBOOST:
    oof_matrix = np.column_stack([oof_lgb, oof_lgb_deep, oof_xgb, oof_cb1, oof_cb2])
    test_matrix = np.column_stack(
        [test_preds_lgb, test_preds_lgb_deep, test_preds_xgb, test_preds_cb1, test_preds_cb2]
    )
else:
    oof_matrix = np.column_stack([oof_lgb, oof_lgb_deep, oof_xgb])
    test_matrix = np.column_stack(
        [test_preds_lgb, test_preds_lgb_deep, test_preds_xgb]
    )

num_models = oof_matrix.shape[1]


def objective_func(weights):
    weights = np.array(weights)
    pred = np.clip(np.dot(oof_matrix, weights), min_bound, max_bound)
    return root_mean_squared_error(y, pred) + 1e-4 * np.sum(weights ** 2)


initial_weights = np.ones(num_models) / num_models
bounds = [(0.0, 1.0) for _ in range(num_models)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    objective_func,
    initial_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
optimal_weights = opt_res.x if opt_res.success else (np.ones(num_models) / num_models)
optimal_weights = np.maximum(0.0, optimal_weights)
if np.sum(optimal_weights) > 0:
    optimal_weights /= np.sum(optimal_weights)
else:
    optimal_weights = np.ones(num_models) / num_models

final_oof_preds = np.clip(np.dot(oof_matrix, optimal_weights), min_bound, max_bound)
final_score = root_mean_squared_error(y, final_oof_preds)

# ---------------------------------------------------------
# 6. Generate Submission File & Verification
# ---------------------------------------------------------
final_test_preds = np.dot(test_matrix, optimal_weights)

# Apply physical target domain clipping ([0.0, 100.0] or train min/max bounds)
min_bound = min(0.0, float(np.min(y_train)))
max_bound = max(100.0, float(np.max(y_train)))
final_test_preds = np.clip(final_test_preds, min_bound, max_bound)

submission_df = pd.DataFrame(
    {"id": test_ids.astype(int), "State Rate": final_test_preds.astype(float)}
)

submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df.to_csv(submission_path, index=False)

assert os.path.exists(submission_path), "Submission file was not created!"
assert len(submission_df) == len(
    test_df
), f"Submission row count mismatch: {len(submission_df)} vs {len(test_df)}"
assert list(submission_df.columns) == [
    "id",
    "State Rate",
], f"Incorrect columns in submission: {submission_df.columns}"
assert (
    not submission_df["State Rate"].isna().any()
), "Submission contains NaN predictions!"

print(f"Final Validation Score: {final_score}")
