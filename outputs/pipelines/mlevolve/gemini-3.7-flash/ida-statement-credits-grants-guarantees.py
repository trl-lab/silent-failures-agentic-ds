import json
import os
import re
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb

try:
    import catboost as cb

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. Directory and Data Loading
# ==============================================================================
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Original Principal Amount (US$)"
assert target_col in train_df.columns, f"Target column '{target_col}' not found."

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# ==============================================================================
# 2. Dynamic Feature Type Detection
# ==============================================================================
all_cols = [c for c in train_df.columns if c != target_col]
date_cols, text_cols, cat_cols, num_cols = [], [], [], []

for col in all_cols:
    sample_series = train_df[col].dropna()
    if len(sample_series) == 0:
        cat_cols.append(col)
        continue

    col_lower = col.lower()
    is_date_name = any(k in col_lower for k in ["date", "year", "month", "day", "time"])
    if is_date_name:
        try:
            pd.to_datetime(sample_series.iloc[:50], errors="raise")
            date_cols.append(col)
            continue
        except Exception:
            pass

    if pd.api.types.is_numeric_dtype(train_df[col]):
        if train_df[col].nunique() < 15 and "id" not in col_lower:
            cat_cols.append(col)
        else:
            num_cols.append(col)
    elif pd.api.types.is_string_dtype(train_df[col]) or pd.api.types.is_object_dtype(
        train_df[col]
    ):
        avg_len = sample_series.astype(str).str.len().mean()
        if (
            avg_len > 30
            or "name" in col_lower
            or "desc" in col_lower
            or "project" in col_lower
        ):
            text_cols.append(col)
        else:
            cat_cols.append(col)
    else:
        cat_cols.append(col)


# ==============================================================================
# 3. Stateless Feature Transformations
# ==============================================================================
def extract_date_features(df, date_columns):
    features = pd.DataFrame(index=df.index)
    parsed_dates = {}
    for col in date_columns:
        dt_s = pd.to_datetime(df[col], errors="coerce")
        parsed_dates[col] = dt_s
        clean_name = re.sub(r"[^\w\s]", "", col).replace(" ", "_")

        features[f"{clean_name}_year"] = dt_s.dt.year.fillna(-1).astype(np.int32)
        features[f"{clean_name}_month"] = dt_s.dt.month.fillna(-1).astype(np.int8)
        features[f"{clean_name}_quarter"] = dt_s.dt.quarter.fillna(-1).astype(np.int8)
        features[f"{clean_name}_day"] = dt_s.dt.day.fillna(-1).astype(np.int8)
        features[f"{clean_name}_dayofweek"] = dt_s.dt.dayofweek.fillna(-1).astype(
            np.int8
        )
        features[f"{clean_name}_is_month_end"] = dt_s.dt.is_month_end.fillna(
            False
        ).astype(np.int8)
        features[f"{clean_name}_timestamp"] = (dt_s.astype("int64") // 10**9).where(
            dt_s.notnull(), -1
        )

    date_keys = list(parsed_dates.keys())
    for i in range(len(date_keys)):
        for j in range(i + 1, len(date_keys)):
            c1, c2 = date_keys[i], date_keys[j]
            name1 = re.sub(r"[^\w\s]", "", c1).replace(" ", "_")
            name2 = re.sub(r"[^\w\s]", "", c2).replace(" ", "_")
            diff_days = (parsed_dates[c2] - parsed_dates[c1]).dt.days
            features[f"duration_{name1}_to_{name2}"] = diff_days.fillna(-999).astype(
                np.float32
            )
    return features


train_date_feats = extract_date_features(train_df, date_cols)
test_date_feats = extract_date_features(test_df, date_cols)

# Frequency Encoding
freq_features_train = pd.DataFrame(index=train_df.index)
freq_features_test = pd.DataFrame(index=test_df.index)
for col in cat_cols:
    clean_name = re.sub(r"[^\w\s]", "", col).replace(" ", "_")
    val_counts = train_df[col].astype(str).value_counts(dropna=False, normalize=True)
    freq_features_train[f"{clean_name}_freq"] = (
        train_df[col].astype(str).map(val_counts).fillna(0).astype(np.float32)
    )
    freq_features_test[f"{clean_name}_freq"] = (
        test_df[col].astype(str).map(val_counts).fillna(0).astype(np.float32)
    )

# Label Encoding
cat_train_encoded = pd.DataFrame(index=train_df.index)
cat_test_encoded = pd.DataFrame(index=test_df.index)
for col in cat_cols:
    clean_name = re.sub(r"[^\w\s]", "", col).replace(" ", "_")
    cat_series_train = train_df[col].astype(str).fillna("__MISSING__")
    cat_series_test = test_df[col].astype(str).fillna("__MISSING__")
    unique_vals = {val: idx for idx, val in enumerate(cat_series_train.unique(), 1)}
    cat_train_encoded[f"{clean_name}_cat"] = (
        cat_series_train.map(unique_vals).fillna(0).astype(np.int32)
    )
    cat_test_encoded[f"{clean_name}_cat"] = (
        cat_series_test.map(unique_vals).fillna(0).astype(np.int32)
    )

# Numerical Clean Features
num_train = pd.DataFrame(index=train_df.index)
num_test = pd.DataFrame(index=test_df.index)
for col in num_cols:
    clean_name = re.sub(r"[^\w\s]", "", col).replace(" ", "_")
    med = train_df[col].median()
    num_train[f"{clean_name}_num"] = train_df[col].fillna(med).astype(np.float32)
    num_test[f"{clean_name}_num"] = test_df[col].fillna(med).astype(np.float32)
    num_train[f"{clean_name}_isna"] = train_df[col].isna().astype(np.int8)
    num_test[f"{clean_name}_isna"] = test_df[col].isna().astype(np.int8)

# Hierarchical Group Aggregations on Numerical & Duration features
group_features_train = pd.DataFrame(index=train_df.index)
group_features_test = pd.DataFrame(index=test_df.index)

candidate_group_cols = [c for c in cat_cols if 2 <= train_df[c].nunique() <= 500]
top_group_cols = sorted(candidate_group_cols, key=lambda c: train_df[c].nunique(), reverse=True)[:6]

num_and_dur_train = pd.concat([num_train, train_date_feats], axis=1)
num_and_dur_test = pd.concat([num_test, test_date_feats], axis=1)
numeric_target_cols = [c for c in num_and_dur_train.columns if not c.endswith("_isna") and not c.endswith("_cat")][:8]

combined_df_features = pd.concat([
    pd.concat([train_df[top_group_cols], num_and_dur_train[numeric_target_cols]], axis=1),
    pd.concat([test_df[top_group_cols], num_and_dur_test[numeric_target_cols]], axis=1)
], axis=0, ignore_index=True)

for g_col in top_group_cols:
    g_clean = re.sub(r"[^\w\s]", "", g_col).replace(" ", "_")
    for n_col in numeric_target_cols:
        n_clean = re.sub(r"[^\w\s]", "", n_col).replace(" ", "_")
        agg_df = combined_df_features.groupby(g_col)[n_col].agg(["mean", "std", "min", "max"]).reset_index()
        mean_map = dict(zip(agg_df[g_col], agg_df["mean"]))
        std_map = dict(zip(agg_df[g_col], agg_df["std"].fillna(0.0)))
        min_map = dict(zip(agg_df[g_col], agg_df["min"]))
        max_map = dict(zip(agg_df[g_col], agg_df["max"]))

        g_mean_tr = train_df[g_col].map(mean_map).fillna(0.0).astype(np.float32)
        g_mean_te = test_df[g_col].map(mean_map).fillna(0.0).astype(np.float32)

        group_features_train[f"grp_{g_clean}_{n_clean}_mean"] = g_mean_tr
        group_features_test[f"grp_{g_clean}_{n_clean}_mean"] = g_mean_te
        group_features_train[f"grp_{g_clean}_{n_clean}_std"] = train_df[g_col].map(std_map).fillna(0.0).astype(np.float32)
        group_features_test[f"grp_{g_clean}_{n_clean}_std"] = test_df[g_col].map(std_map).fillna(0.0).astype(np.float32)
        group_features_train[f"grp_{g_clean}_{n_clean}_min"] = train_df[g_col].map(min_map).fillna(0.0).astype(np.float32)
        group_features_test[f"grp_{g_clean}_{n_clean}_min"] = test_df[g_col].map(min_map).fillna(0.0).astype(np.float32)
        group_features_train[f"grp_{g_clean}_{n_clean}_max"] = train_df[g_col].map(max_map).fillna(0.0).astype(np.float32)
        group_features_test[f"grp_{g_clean}_{n_clean}_max"] = test_df[g_col].map(max_map).fillna(0.0).astype(np.float32)

        # Relative difference and ratio
        val_tr = num_and_dur_train[n_col].values
        val_te = num_and_dur_test[n_col].values
        group_features_train[f"grp_{g_clean}_{n_clean}_diff"] = (val_tr - g_mean_tr).astype(np.float32)
        group_features_test[f"grp_{g_clean}_{n_clean}_diff"] = (val_te - g_mean_te).astype(np.float32)
        group_features_train[f"grp_{g_clean}_{n_clean}_ratio"] = (val_tr / (np.abs(g_mean_tr) + 1e-4)).astype(np.float32)
        group_features_test[f"grp_{g_clean}_{n_clean}_ratio"] = (val_te / (np.abs(g_mean_te) + 1e-4)).astype(np.float32)

# ==============================================================================
# 4. Leak-Free Cross-Validation Stateful Encodings (TE, 2-Way Interactions, Ridge Text)
# ==============================================================================
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
train_df["fold"] = -1
for fold, (trn_idx, val_idx) in enumerate(kf.split(train_df)):
    train_df.loc[val_idx, "fold"] = fold

target_vals = train_df[target_col].values
global_mean = float(np.mean(target_vals))

te_cols = [c for c in cat_cols if train_df[c].nunique() > 1]

# 2-way categorical interaction pairs
interaction_cols_pairs = []
top_interact_cats = [c for c in cat_cols if 2 <= train_df[c].nunique() <= 200][:5]
for i in range(len(top_interact_cats)):
    for j in range(i + 1, len(top_interact_cats)):
        interaction_cols_pairs.append((top_interact_cats[i], top_interact_cats[j]))

interaction_train = pd.DataFrame(index=train_df.index)
interaction_test = pd.DataFrame(index=test_df.index)
interact_names = []

for c1, c2 in interaction_cols_pairs:
    pair_name = f"inter_{re.sub(r'[^a-zA-Z0-9]', '', c1)}_{re.sub(r'[^a-zA-Z0-9]', '', c2)}"
    interact_names.append(pair_name)
    train_df[pair_name] = train_df[c1].astype(str) + "__" + train_df[c2].astype(str)
    test_df[pair_name] = test_df[c1].astype(str) + "__" + test_df[c2].astype(str)
    interaction_train[f"{pair_name}_te_mean"] = 0.0
    interaction_train[f"{pair_name}_te_std"] = 0.0
    interaction_test[f"{pair_name}_te_mean"] = 0.0
    interaction_test[f"{pair_name}_te_std"] = 0.0

te_train = pd.DataFrame(index=train_df.index)
te_test = pd.DataFrame(index=test_df.index)

for col in te_cols:
    clean_name = re.sub(r"[^\w\s]", "", col).replace(" ", "_")
    te_train[f"{clean_name}_te_mean"] = 0.0
    te_train[f"{clean_name}_te_std"] = 0.0
    te_test[f"{clean_name}_te_mean"] = 0.0
    te_test[f"{clean_name}_te_std"] = 0.0

smooth_weight = 10.0
for fold in range(N_SPLITS):
    trn_idx = train_df["fold"] != fold
    val_idx = train_df["fold"] == fold
    trn_subset = train_df[trn_idx]
    val_subset = train_df[val_idx]

    # 1D Target Encodings
    for col in te_cols:
        clean_name = re.sub(r"[^\w\s]", "", col).replace(" ", "_")
        stats = (
            trn_subset.groupby(col)[target_col]
            .agg(["count", "mean", "std"])
            .reset_index()
        )
        stats["smooth_mean"] = (
            stats["count"] * stats["mean"] + smooth_weight * global_mean
        ) / (stats["count"] + smooth_weight)
        stats["std"] = stats["std"].fillna(0.0)

        mapping_mean = dict(zip(stats[col], stats["smooth_mean"]))
        mapping_std = dict(zip(stats[col], stats["std"]))

        te_train.loc[val_idx, f"{clean_name}_te_mean"] = (
            val_subset[col].map(mapping_mean).fillna(global_mean)
        )
        te_train.loc[val_idx, f"{clean_name}_te_std"] = (
            val_subset[col].map(mapping_std).fillna(0.0)
        )

    # 2D Interaction Target Encodings
    for pair_name in interact_names:
        stats = (
            trn_subset.groupby(pair_name)[target_col]
            .agg(["count", "mean", "std"])
            .reset_index()
        )
        stats["smooth_mean"] = (
            stats["count"] * stats["mean"] + smooth_weight * global_mean
        ) / (stats["count"] + smooth_weight)
        stats["std"] = stats["std"].fillna(0.0)

        mapping_mean = dict(zip(stats[pair_name], stats["smooth_mean"]))
        mapping_std = dict(zip(stats[pair_name], stats["std"]))

        interaction_train.loc[val_idx, f"{pair_name}_te_mean"] = (
            val_subset[pair_name].map(mapping_mean).fillna(global_mean)
        )
        interaction_train.loc[val_idx, f"{pair_name}_te_std"] = (
            val_subset[pair_name].map(mapping_std).fillna(0.0)
        )

for col in te_cols:
    clean_name = re.sub(r"[^\w\s]", "", col).replace(" ", "_")
    stats = (
        train_df.groupby(col)[target_col].agg(["count", "mean", "std"]).reset_index()
    )
    stats["smooth_mean"] = (
        stats["count"] * stats["mean"] + smooth_weight * global_mean
    ) / (stats["count"] + smooth_weight)
    stats["std"] = stats["std"].fillna(0.0)

    mapping_mean = dict(zip(stats[col], stats["smooth_mean"]))
    mapping_std = dict(zip(stats[col], stats["std"]))

    te_test[f"{clean_name}_te_mean"] = (
        test_df[col].map(mapping_mean).fillna(global_mean)
    )
    te_test[f"{clean_name}_te_std"] = test_df[col].map(mapping_std).fillna(0.0)

for pair_name in interact_names:
    stats = (
        train_df.groupby(pair_name)[target_col].agg(["count", "mean", "std"]).reset_index()
    )
    stats["smooth_mean"] = (
        stats["count"] * stats["mean"] + smooth_weight * global_mean
    ) / (stats["count"] + smooth_weight)
    stats["std"] = stats["std"].fillna(0.0)

    mapping_mean = dict(zip(stats[pair_name], stats["smooth_mean"]))
    mapping_std = dict(zip(stats[pair_name], stats["std"]))

    interaction_test[f"{pair_name}_te_mean"] = (
        test_df[pair_name].map(mapping_mean).fillna(global_mean)
    )
    interaction_test[f"{pair_name}_te_std"] = test_df[pair_name].map(mapping_std).fillna(0.0)

# Text Feature Extraction with TF-IDF, SVD & Out-of-Fold Ridge Text Meta-Features
text_features_train = pd.DataFrame(index=train_df.index)
text_features_test = pd.DataFrame(index=test_df.index)

if len(text_cols) > 0:
    combined_train_text = train_df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    combined_test_text = test_df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)

    tfidf_ridge = TfidfVectorizer(max_features=12000, stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    tfidf_trn_all = tfidf_ridge.fit_transform(combined_train_text)
    tfidf_tst_all = tfidf_ridge.transform(combined_test_text)

    oof_ridge_meta = np.zeros(len(train_df), dtype=np.float32)
    test_ridge_meta = np.zeros(len(test_df), dtype=np.float32)

    for fold in range(N_SPLITS):
        trn_idx = train_df["fold"] != fold
        val_idx = train_df["fold"] == fold
        ridge_model = Ridge(alpha=15.0, random_state=42)
        ridge_model.fit(tfidf_trn_all[trn_idx], target_vals[trn_idx])
        oof_ridge_meta[val_idx] = ridge_model.predict(tfidf_trn_all[val_idx])
        test_ridge_meta += ridge_model.predict(tfidf_tst_all) / N_SPLITS

    text_features_train["oof_ridge_text_meta"] = oof_ridge_meta
    text_features_test["oof_ridge_text_meta"] = test_ridge_meta

    for col in text_cols:
        clean_name = re.sub(r"[^\w\s]", "", col).replace(" ", "_")
        train_text = train_df[col].fillna("").astype(str)
        test_text = test_df[col].fillna("").astype(str)

        text_features_train[f"{clean_name}_char_len"] = train_text.str.len().astype(
            np.float32
        )
        text_features_test[f"{clean_name}_char_len"] = test_text.str.len().astype(
            np.float32
        )
        text_features_train[f"{clean_name}_word_count"] = (
            train_text.str.split().str.len().astype(np.float32)
        )
        text_features_test[f"{clean_name}_word_count"] = (
            test_text.str.split().str.len().astype(np.float32)
        )

        tfidf = TfidfVectorizer(
            max_features=2500, stop_words="english", ngram_range=(1, 2)
        )
        tfidf_trn = tfidf.fit_transform(train_text)
        tfidf_tst = tfidf.transform(test_text)

        n_comp = min(8, tfidf_trn.shape[1] - 1) if tfidf_trn.shape[1] > 1 else 1
        if n_comp > 0:
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            svd_trn = svd.fit_transform(tfidf_trn)
            svd_tst = svd.transform(tfidf_tst)

            for c_idx in range(n_comp):
                text_features_train[f"{clean_name}_svd_{c_idx}"] = svd_trn[
                    :, c_idx
                ].astype(np.float32)
                text_features_test[f"{clean_name}_svd_{c_idx}"] = svd_tst[
                    :, c_idx
                ].astype(np.float32)

# Assemble Features
X_train_df = pd.concat(
    [
        num_train,
        train_date_feats,
        freq_features_train,
        cat_train_encoded,
        group_features_train,
        interaction_train,
        te_train,
        text_features_train,
    ],
    axis=1,
)

X_test_df = pd.concat(
    [
        num_test,
        test_date_feats,
        freq_features_test,
        cat_test_encoded,
        group_features_test,
        interaction_test,
        te_test,
        text_features_test,
    ],
    axis=1,
)

clean_col_names = [re.sub(r"[^\w]", "_", c) for c in X_train_df.columns]
X_train_df.columns = clean_col_names
X_test_df.columns = clean_col_names

X_train_data = X_train_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
X_test_data = X_test_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
y_train = train_df[target_col].values.astype(np.float64)
folds = train_df["fold"].values


# ==============================================================================
# 5. Model Architecture & Hyperparameter Definitions
# ==============================================================================
def get_lgbm_params(seed: int = 42) -> dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 45,
        "max_depth": 7,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_estimators": 2500,
        "verbosity": -1,
        "n_jobs": -1,
    }


def get_lgbm_huber_params(seed: int = 42) -> dict:
    return {
        "objective": "huber",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 25,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "random_state": seed,
        "n_estimators": 2500,
        "verbosity": -1,
        "n_jobs": -1,
    }


def get_xgb_params(seed: int = 42) -> dict:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_estimators": 2500,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def get_catboost_params(seed: int = 42) -> dict:
    if not HAS_CATBOOST:
        return {}
    return {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "learning_rate": 0.035,
        "depth": 6,
        "l2_leaf_reg": 4.0,
        "random_seed": seed,
        "iterations": 2500,
        "verbose": False,
        "thread_count": -1,
    }


# ==============================================================================
# 6. Model Training & Cross-Validation
# ==============================================================================
oof_lgb = np.zeros(len(X_train_data), dtype=np.float64)
test_lgb = np.zeros(len(X_test_data), dtype=np.float64)

oof_lgb_huber = np.zeros(len(X_train_data), dtype=np.float64)
test_lgb_huber = np.zeros(len(X_test_data), dtype=np.float64)

oof_xgb = np.zeros(len(X_train_data), dtype=np.float64)
test_xgb = np.zeros(len(X_test_data), dtype=np.float64)

if HAS_CATBOOST:
    oof_cb = np.zeros(len(X_train_data), dtype=np.float64)
    test_cb = np.zeros(len(X_test_data), dtype=np.float64)

for fold in range(N_SPLITS):
    trn_mask = folds != fold
    val_mask = folds == fold

    X_tr, y_tr = X_train_data.iloc[trn_mask], y_train[trn_mask]
    X_va, y_val = X_train_data.iloc[val_mask], y_train[val_mask]

    # --- LightGBM (RMSE) ---
    lgb_p = get_lgbm_params(seed=42 + fold)
    n_est = lgb_p.pop("n_estimators", 2500)
    lgb_train_ds = lgb.Dataset(X_tr, label=y_tr)
    lgb_val_ds = lgb.Dataset(X_va, label=y_val, reference=lgb_train_ds)

    lgb_model = lgb.train(
        lgb_p,
        lgb_train_ds,
        num_boost_round=n_est,
        valid_sets=[lgb_val_ds],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    oof_lgb[val_mask] = lgb_model.predict(X_va, num_iteration=lgb_model.best_iteration)
    test_lgb += (
        lgb_model.predict(X_test_data, num_iteration=lgb_model.best_iteration)
        / N_SPLITS
    )

    # --- LightGBM (Huber Outlier-Robust) ---
    lgb_h_p = get_lgbm_huber_params(seed=142 + fold)
    n_est_h = lgb_h_p.pop("n_estimators", 2500)
    lgb_h_train_ds = lgb.Dataset(X_tr, label=y_tr)
    lgb_h_val_ds = lgb.Dataset(X_va, label=y_val, reference=lgb_h_train_ds)

    lgb_huber_model = lgb.train(
        lgb_h_p,
        lgb_h_train_ds,
        num_boost_round=n_est_h,
        valid_sets=[lgb_h_val_ds],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    oof_lgb_huber[val_mask] = lgb_huber_model.predict(X_va, num_iteration=lgb_huber_model.best_iteration)
    test_lgb_huber += (
        lgb_huber_model.predict(X_test_data, num_iteration=lgb_huber_model.best_iteration)
        / N_SPLITS
    )

    # --- XGBoost ---
    xgb_p = get_xgb_params(seed=42 + fold)
    xgb_model = xgb.XGBRegressor(**xgb_p, early_stopping_rounds=50)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_val)], verbose=False)

    oof_xgb[val_mask] = xgb_model.predict(X_va)
    test_xgb += xgb_model.predict(X_test_data) / N_SPLITS

    # --- CatBoost ---
    if HAS_CATBOOST:
        cb_p = get_catboost_params(seed=42 + fold)
        cb_model = cb.CatBoostRegressor(**cb_p, early_stopping_rounds=50)
        cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_val), verbose=False)

        oof_cb[val_mask] = cb_model.predict(X_va)
        test_cb += cb_model.predict(X_test_data) / N_SPLITS

    fold_lgb_rmse = np.sqrt(mean_squared_error(y_val, oof_lgb[val_mask]))
    fold_lgb_huber_rmse = np.sqrt(mean_squared_error(y_val, oof_lgb_huber[val_mask]))
    fold_xgb_rmse = np.sqrt(mean_squared_error(y_val, oof_xgb[val_mask]))
    log_line = f"Fold {fold + 1}/{N_SPLITS} | LGBM: {fold_lgb_rmse:,.2f} | LGBM-Huber: {fold_lgb_huber_rmse:,.2f} | XGB: {fold_xgb_rmse:,.2f}"
    if HAS_CATBOOST:
        fold_cb_rmse = np.sqrt(mean_squared_error(y_val, oof_cb[val_mask]))
        log_line += f" | CatBoost: {fold_cb_rmse:,.2f}"
    print(log_line)

# ==============================================================================
# 7. Constrained Optimization Blending & Evaluation
# ==============================================================================
models_oof = [oof_lgb, oof_lgb_huber, oof_xgb]
models_test = [test_lgb, test_lgb_huber, test_xgb]

if HAS_CATBOOST:
    models_oof.append(oof_cb)
    models_test.append(test_cb)

oof_matrix = np.column_stack(models_oof)
test_matrix = np.column_stack(models_test)
n_models = oof_matrix.shape[1]


def rmse_objective(weights):
    w = weights / np.sum(weights)
    pred = np.dot(oof_matrix, w)
    pred = np.clip(pred, a_min=0.0, a_max=None)
    return np.sqrt(mean_squared_error(y_train, pred))


init_weights = np.ones(n_models) / n_models
bounds = [(0.0, 1.0) for _ in range(n_models)]
res = minimize(
    rmse_objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints={"type": "eq", "fun": lambda w: 1.0 - np.sum(w)},
)

optimal_weights = res.x / np.sum(res.x) if res.success else np.ones(n_models) / n_models
oof_final = np.clip(np.dot(oof_matrix, optimal_weights), a_min=0.0, a_max=None)
final_val_score = float(np.sqrt(mean_squared_error(y_train, oof_final)))

# ==============================================================================
# 8. Generate Submission File & Report Metric
# ==============================================================================
test_final = np.clip(np.dot(test_matrix, optimal_weights), a_min=0.0, a_max=None)

submission_df = pd.DataFrame({"id": np.arange(len(test_final)), target_col: test_final})
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_val_score}")
