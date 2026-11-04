import os
from typing import Any, Dict, Optional, Tuple
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from catboost import CatBoostClassifier

# Ensure output directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# =========================================================================
# 1. Data Loading and Preprocessing
# =========================================================================

print("Loading raw train and test datasets...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "APPETENCY"
if target_col not in train_df.columns:
    possible_targets = [
        c for c in train_df.columns if "target" in c.lower() or "appetency" in c.lower()
    ]
    if possible_targets:
        target_col = possible_targets[0]

print(f"Target column identified: {target_col}")

# Extract and map target variable {-1, 1} -> {0, 1}
y_raw = train_df[target_col].copy()
if set(y_raw.dropna().unique()).issubset({-1, 1, -1.0, 1.0}):
    y = (y_raw == 1).astype(int).values
elif set(y_raw.dropna().unique()).issubset({0, 1, 0.0, 1.0}):
    y = y_raw.astype(int).values
else:
    unique_vals = sorted(y_raw.dropna().unique())
    y = (y_raw == unique_vals[-1]).astype(int).values

print(
    f"Target distribution: Positive = {np.sum(y == 1)} ({np.mean(y == 1):.4%}), Negative = {np.sum(y == 0)}"
)

# Drop target column from training features
train_features = train_df.drop(columns=[target_col]).copy()
test_features = test_df.copy()

# Ensure matching columns between train and test
feature_cols = [col for col in train_features.columns if col in test_features.columns]
train_features = train_features[feature_cols]
test_features = test_features[feature_cols]

# Identify feature types: Numerical vs Categorical
num_cols = []
cat_cols = []

for col in feature_cols:
    if pd.api.types.is_numeric_dtype(train_features[col]):
        num_cols.append(col)
    else:
        cat_cols.append(col)

print(
    f"Initial feature count: {len(feature_cols)} (Numerical: {len(num_cols)}, Categorical: {len(cat_cols)})"
)

# 1. Pruning constant and extremely sparse columns (> 98% missing)
cols_to_drop = []
for col in feature_cols:
    train_null_ratio = train_features[col].isnull().mean()
    if train_null_ratio > 0.98:
        cols_to_drop.append(col)
    elif train_features[col].nunique(dropna=True) <= 1:
        cols_to_drop.append(col)

cols_to_drop = list(set(cols_to_drop))
print(f"Pruning {len(cols_to_drop)} uninformative/empty columns...")

train_features.drop(columns=cols_to_drop, inplace=True, errors="ignore")
test_features.drop(columns=cols_to_drop, inplace=True, errors="ignore")

num_cols = [c for c in num_cols if c not in cols_to_drop]
cat_cols = [c for c in cat_cols if c not in cols_to_drop]

# Dictionary buffers to collect new features cleanly and prevent DataFrame fragmentation
train_new_cols = {}
test_new_cols = {}

# 2. Missingness Indicators & Summary Features
print("Constructing missingness indicators and summary features...")
for col in feature_cols:
    if col in train_features.columns:
        null_ratio = train_features[col].isnull().mean()
        if 0.05 <= null_ratio <= 0.95:
            train_new_cols[f"{col}_isna"] = train_features[col].isnull().astype(np.int8)
            test_new_cols[f"{col}_isna"] = test_features[col].isnull().astype(np.int8)

train_new_cols["missing_count"] = train_features.isnull().sum(axis=1).astype(np.float32)
test_new_cols["missing_count"] = test_features.isnull().sum(axis=1).astype(np.float32)
train_new_cols["missing_ratio"] = (train_new_cols["missing_count"] / len(feature_cols)).astype(np.float32)
test_new_cols["missing_ratio"] = (test_new_cols["missing_count"] / len(feature_cols)).astype(np.float32)

# Latent missingness structure via TruncatedSVD (5 components)
miss_mat_tr = train_features.isnull().astype(np.float32).values
miss_mat_te = test_features.isnull().astype(np.float32).values
if miss_mat_tr.shape[1] >= 5:
    svd_miss = TruncatedSVD(n_components=5, random_state=42)
    tr_svd_miss = svd_miss.fit_transform(miss_mat_tr)
    te_svd_miss = svd_miss.transform(miss_mat_te)
    for k in range(5):
        train_new_cols[f"miss_svd_{k}"] = tr_svd_miss[:, k].astype(np.float32)
        test_new_cols[f"miss_svd_{k}"] = te_svd_miss[:, k].astype(np.float32)

# 3. Row-level Numerical Aggregations
if len(num_cols) > 0:
    print("Constructing row-level numerical aggregations...")
    train_num_mat = train_features[num_cols].values
    test_num_mat = test_features[num_cols].values

    train_new_cols["num_row_mean"] = np.nanmean(train_num_mat, axis=1).astype(np.float32)
    test_new_cols["num_row_mean"] = np.nanmean(test_num_mat, axis=1).astype(np.float32)

    train_new_cols["num_row_std"] = np.nanstd(train_num_mat, axis=1).astype(np.float32)
    test_new_cols["num_row_std"] = np.nanstd(test_num_mat, axis=1).astype(np.float32)

    train_new_cols["num_row_sum"] = np.nansum(train_num_mat, axis=1).astype(np.float32)
    test_new_cols["num_row_sum"] = np.nansum(test_num_mat, axis=1).astype(np.float32)

# 4. Frequency Encoding for Categorical Columns
print("Generating frequency encodings for categorical features...")
for col in cat_cols:
    train_series_str = train_features[col].astype(str).fillna("__MISSING__")
    test_series_str = test_features[col].astype(str).fillna("__MISSING__")

    combined = pd.concat([train_series_str, test_series_str], axis=0)
    freq_map = combined.value_counts(normalize=True).to_dict()

    train_new_cols[f"{col}_freq"] = (
        train_series_str.map(freq_map).fillna(0.0).astype(np.float32)
    )
    test_new_cols[f"{col}_freq"] = (
        test_series_str.map(freq_map).fillna(0.0).astype(np.float32)
    )

# 5. Multi-Scale Out-Of-Fold (OOF) Target Encoding with 10-Fold Stratified Split
print("Performing Multi-Scale Out-Of-Fold Target Encoding (10-Fold)...")
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_indices = list(skf.split(train_features, y))

global_mean = float(np.mean(y))
smoothing_values = [5.0, 20.0]

for col in cat_cols:
    train_col_str = train_features[col].astype(str).fillna("__MISSING__")
    test_col_str = test_features[col].astype(str).fillna("__MISSING__")

    for smoothing in smoothing_values:
        suffix = f"_te_s{int(smoothing)}"
        oof_target_enc = np.full(len(train_features), global_mean, dtype=np.float32)

        for train_idx, val_idx in fold_indices:
            tr_cat = train_col_str.iloc[train_idx]
            tr_y = y[train_idx]

            cat_stats = (
                pd.DataFrame({"cat": tr_cat, "target": tr_y})
                .groupby("cat")["target"]
                .agg(["count", "mean"])
            )
            smoothed_val = (
                cat_stats["count"] * cat_stats["mean"] + smoothing * global_mean
            ) / (cat_stats["count"] + smoothing)
            cat_map = smoothed_val.to_dict()

            oof_target_enc[val_idx] = (
                train_col_str.iloc[val_idx].map(cat_map).fillna(global_mean).values
            )

        overall_stats = (
            pd.DataFrame({"cat": train_col_str, "target": y})
            .groupby("cat")["target"]
            .agg(["count", "mean"])
        )
        overall_smoothed = (
            overall_stats["count"] * overall_stats["mean"] + smoothing * global_mean
        ) / (overall_stats["count"] + smoothing)
        overall_cat_map = overall_smoothed.to_dict()
        test_target_enc = test_col_str.map(overall_cat_map).fillna(global_mean).astype(np.float32).values

        train_new_cols[f"{col}{suffix}"] = oof_target_enc
        test_new_cols[f"{col}{suffix}"] = test_target_enc

# 5b. Categorical 2-Way Interactions with OOF Target Encoding and Frequency Encoding
print("Constructing 2-way categorical interactions with OOF target and frequency encoding...")
top_cat_for_pairs = [c for c in cat_cols if 2 <= train_features[c].nunique(dropna=True) <= 50][:6]
cat_pairs = []
for i in range(len(top_cat_for_pairs)):
    for j in range(i + 1, len(top_cat_for_pairs)):
        cat_pairs.append((top_cat_for_pairs[i], top_cat_for_pairs[j]))

for c1, c2 in cat_pairs:
    pair_tr = train_features[c1].astype(str).fillna("__NA__") + "_" + train_features[c2].astype(str).fillna("__NA__")
    pair_te = test_features[c1].astype(str).fillna("__NA__") + "_" + test_features[c2].astype(str).fillna("__NA__")

    comb_pair = pd.concat([pair_tr, pair_te], axis=0)
    pair_freq_map = comb_pair.value_counts(normalize=True).to_dict()
    train_new_cols[f"{c1}_{c2}_freq"] = pair_tr.map(pair_freq_map).fillna(0.0).astype(np.float32)
    test_new_cols[f"{c1}_{c2}_freq"] = pair_te.map(pair_freq_map).fillna(0.0).astype(np.float32)

    pair_oof_te = np.full(len(train_features), global_mean, dtype=np.float32)
    pair_smoothing = 15.0
    for train_idx, val_idx in fold_indices:
        tr_c = pair_tr.iloc[train_idx]
        tr_y = y[train_idx]
        cat_stats = pd.DataFrame({"cat": tr_c, "target": tr_y}).groupby("cat")["target"].agg(["count", "mean"])
        smoothed_val = (cat_stats["count"] * cat_stats["mean"] + pair_smoothing * global_mean) / (cat_stats["count"] + pair_smoothing)
        cat_map = smoothed_val.to_dict()
        pair_oof_te[val_idx] = pair_tr.iloc[val_idx].map(cat_map).fillna(global_mean).values

    overall_stats = pd.DataFrame({"cat": pair_tr, "target": y}).groupby("cat")["target"].agg(["count", "mean"])
    overall_smoothed = (overall_stats["count"] * overall_stats["mean"] + pair_smoothing * global_mean) / (overall_stats["count"] + pair_smoothing)
    overall_cat_map = overall_smoothed.to_dict()
    pair_te_enc = pair_te.map(overall_cat_map).fillna(global_mean).astype(np.float32).values

    train_new_cols[f"{c1}_{c2}_te"] = pair_oof_te
    test_new_cols[f"{c1}_{c2}_te"] = pair_te_enc

# 6. Category-Numerical Group Aggregations
print("Constructing category-numerical group aggregations...")
valid_cats = [c for c in cat_cols if 3 <= train_features[c].nunique(dropna=True) <= 100][:4]
valid_nums = [c for c in num_cols if train_features[c].notnull().sum() > 20000][:4]

for cat_col in valid_cats:
    tr_cat_s = train_features[cat_col].astype(str).fillna("__MISSING__")
    te_cat_s = test_features[cat_col].astype(str).fillna("__MISSING__")
    for num_col in valid_nums:
        grp_stats = (
            pd.DataFrame({"cat": tr_cat_s, "val": train_features[num_col]})
            .groupby("cat")["val"]
            .agg(["mean", "std", "min", "max"])
        )
        for stat_type in ["mean", "std", "min", "max"]:
            stat_map = grp_stats[stat_type].to_dict()
            feat_name = f"{cat_col}_{num_col}_{stat_type}"
            train_new_cols[feat_name] = tr_cat_s.map(stat_map).fillna(0.0).astype(np.float32)
            test_new_cols[feat_name] = te_cat_s.map(stat_map).fillna(0.0).astype(np.float32)

# 7. Ordinal / Category Codes for Tree Models
for col in cat_cols:
    train_col_str = train_features[col].astype(str).fillna("__MISSING__")
    test_col_str = test_features[col].astype(str).fillna("__MISSING__")

    unique_categories = list(train_col_str.unique())
    cat_to_code = {cat: idx for idx, cat in enumerate(unique_categories)}

    train_new_cols[f"{col}_cat"] = train_col_str.map(cat_to_code).fillna(-1).astype(int)
    test_new_cols[f"{col}_cat"] = test_col_str.map(cat_to_code).fillna(-1).astype(int)

# 8. Numerical Interaction Features
if len(num_cols) >= 2:
    print("Computing top interaction features...")
    valid_num_cols = [c for c in num_cols if train_features[c].notnull().sum() > 1000]
    if len(valid_num_cols) >= 2:
        top_num = valid_num_cols[:4]
        for i in range(len(top_num)):
            for j in range(i + 1, len(top_num)):
                col1, col2 = top_num[i], top_num[j]
                train_new_cols[f"{col1}_mult_{col2}"] = (
                    (train_features[col1] * train_features[col2]).fillna(0.0).astype(np.float32)
                )
                test_new_cols[f"{col1}_mult_{col2}"] = (
                    (test_features[col1] * test_features[col2]).fillna(0.0).astype(np.float32)
                )
                train_new_cols[f"{col1}_diff_{col2}"] = (
                    (train_features[col1] - train_features[col2]).fillna(0.0).astype(np.float32)
                )
                test_new_cols[f"{col1}_diff_{col2}"] = (
                    (test_features[col1] - test_features[col2]).fillna(0.0).astype(np.float32)
                )

# 8b. Robust Quantile Binning for Predictive Numerical Features
print("Generating quantile binning features for top numericals...")
top_nums_bin = [c for c in num_cols if train_features[c].notnull().sum() > 5000][:6]
for c in top_nums_bin:
    try:
        tr_series = train_features[c].dropna()
        if len(tr_series) > 100:
            _, bin_edges = pd.qcut(tr_series, q=10, labels=False, retbins=True, duplicates="drop")
            train_new_cols[f"{c}_qbin"] = pd.cut(train_features[c], bins=bin_edges, labels=False, include_lowest=True).fillna(-1).astype(np.float32)
            test_new_cols[f"{c}_qbin"] = pd.cut(test_features[c], bins=bin_edges, labels=False, include_lowest=True).fillna(-1).astype(np.float32)
    except Exception:
        pass

# Drop original categorical raw strings and concatenate engineered features in single operation
train_features.drop(columns=cat_cols, inplace=True, errors="ignore")
test_features.drop(columns=cat_cols, inplace=True, errors="ignore")

train_features = pd.concat([train_features, pd.DataFrame(train_new_cols, index=train_features.index)], axis=1)
test_features = pd.concat([test_features, pd.DataFrame(test_new_cols, index=test_features.index)], axis=1)

# Ensure clean string column names
train_features.columns = [str(c) for c in train_features.columns]
test_features.columns = [str(c) for c in test_features.columns]

print(f"Final feature count for modeling: {train_features.shape[1]}")

# =========================================================================
# 2. Model Configurations
# =========================================================================


def get_lightgbm_model(
    random_state: int = 42, custom_params: Optional[Dict[str, Any]] = None
) -> lgb.LGBMClassifier:
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
        "importance_type": "gain",
    }
    if custom_params:
        params.update(custom_params)
    return lgb.LGBMClassifier(**params)


def get_dart_lgb_model(
    random_state: int = 42, custom_params: Optional[Dict[str, Any]] = None
) -> lgb.LGBMClassifier:
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "dart",
        "n_estimators": 600,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "drop_rate": 0.1,
        "max_drop": 50,
        "skip_drop": 0.5,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
        "importance_type": "gain",
    }
    if custom_params:
        params.update(custom_params)
    return lgb.LGBMClassifier(**params)


def get_xgboost_model(
    random_state: int = 42, custom_params: Optional[Dict[str, Any]] = None
) -> xgb.XGBClassifier:
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 15,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "colsample_bylevel": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
    }
    if custom_params:
        params.update(custom_params)
    return xgb.XGBClassifier(**params)


def get_catboost_model(
    random_state: int = 42, custom_params: Optional[Dict[str, Any]] = None
) -> CatBoostClassifier:
    params = {
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 5,
        "l2_leaf_reg": 6.0,
        "random_strength": 0.8,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": random_state,
        "early_stopping_rounds": 50,
        "verbose": False,
        "thread_count": -1,
    }
    if custom_params:
        params.update(custom_params)
    return CatBoostClassifier(**params)


# =========================================================================
# 3. Stratified Cross-Validation, Ensembling & Evaluation
# =========================================================================

n_folds = len(fold_indices)
n_train = len(train_features)
n_test = len(test_features)

print(
    f"Starting {n_folds}-Fold cross-validation training on {n_train} samples with {train_features.shape[1]} features..."
)

oof_lgb = np.zeros(n_train, dtype=np.float64)
oof_dart = np.zeros(n_train, dtype=np.float64)
oof_xgb = np.zeros(n_train, dtype=np.float64)
oof_cb = np.zeros(n_train, dtype=np.float64)
oof_ensemble = np.zeros(n_train, dtype=np.float64)

test_preds_lgb = np.zeros(n_test, dtype=np.float64)
test_preds_dart = np.zeros(n_test, dtype=np.float64)
test_preds_xgb = np.zeros(n_test, dtype=np.float64)
test_preds_cb = np.zeros(n_test, dtype=np.float64)

for fold, (train_idx, val_idx) in enumerate(fold_indices):
    X_tr, y_tr = train_features.iloc[train_idx], y[train_idx]
    X_va, y_va = train_features.iloc[val_idx], y[val_idx]

    # 1. Train Deep LightGBM GBDT
    lgb_clf = get_lightgbm_model(random_state=42 + fold)
    lgb_clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    val_pred_lgb = lgb_clf.predict_proba(X_va)[:, 1]
    test_pred_lgb = lgb_clf.predict_proba(test_features)[:, 1]
    joblib.dump(lgb_clf, f"./working/lgb_fold_{fold}.joblib")

    oof_lgb[val_idx] = val_pred_lgb
    test_preds_lgb += test_pred_lgb / n_folds

    # 2. Train LightGBM DART
    dart_clf = get_dart_lgb_model(random_state=42 + fold)
    dart_clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[lgb.log_evaluation(period=0)],
    )
    val_pred_dart = dart_clf.predict_proba(X_va)[:, 1]
    test_pred_dart = dart_clf.predict_proba(test_features)[:, 1]
    joblib.dump(dart_clf, f"./working/dart_fold_{fold}.joblib")

    oof_dart[val_idx] = val_pred_dart
    test_preds_dart += test_pred_dart / n_folds

    # 3. Train Histogram XGBoost
    xgb_clf = get_xgboost_model(random_state=42 + fold)
    xgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    val_pred_xgb = xgb_clf.predict_proba(X_va)[:, 1]
    test_pred_xgb = xgb_clf.predict_proba(test_features)[:, 1]
    joblib.dump(xgb_clf, f"./working/xgb_fold_{fold}.joblib")

    oof_xgb[val_idx] = val_pred_xgb
    test_preds_xgb += test_pred_xgb / n_folds

    # 4. Train Tuned CatBoost
    cb_clf = get_catboost_model(random_state=42 + fold)
    cb_clf.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    val_pred_cb = cb_clf.predict_proba(X_va)[:, 1]
    test_pred_cb = cb_clf.predict_proba(test_features)[:, 1]
    cb_clf.save_model(f"./working/cb_fold_{fold}.cbm")

    oof_cb[val_idx] = val_pred_cb
    test_preds_cb += test_pred_cb / n_folds

    fold_lgb_auc = roc_auc_score(y_va, val_pred_lgb)
    fold_dart_auc = roc_auc_score(y_va, val_pred_dart)
    fold_xgb_auc = roc_auc_score(y_va, val_pred_xgb)
    fold_cb_auc = roc_auc_score(y_va, val_pred_cb)

    print(
        f"Fold {fold+1:02d}/{n_folds} - LGB: {fold_lgb_auc:.5f} | DART: {fold_dart_auc:.5f} | XGB: {fold_xgb_auc:.5f} | CB: {fold_cb_auc:.5f}"
    )

# Compute Model Out-Of-Fold Scores
overall_lgb_auc = roc_auc_score(y, oof_lgb)
overall_dart_auc = roc_auc_score(y, oof_dart)
overall_xgb_auc = roc_auc_score(y, oof_xgb)
overall_cb_auc = roc_auc_score(y, oof_cb)

print(
    f"OOF GBDT AUC: {overall_lgb_auc:.5f} | OOF DART AUC: {overall_dart_auc:.5f} | OOF XGBoost AUC: {overall_xgb_auc:.5f} | OOF CatBoost AUC: {overall_cb_auc:.5f}"
)

# Convert out-of-fold and test predictions to percentile ranks
rank_oof_lgb = rankdata(oof_lgb) / len(oof_lgb)
rank_oof_dart = rankdata(oof_dart) / len(oof_dart)
rank_oof_xgb = rankdata(oof_xgb) / len(oof_xgb)
rank_oof_cb = rankdata(oof_cb) / len(oof_cb)

rank_test_lgb = rankdata(test_preds_lgb) / len(test_preds_lgb)
rank_test_dart = rankdata(test_preds_dart) / len(test_preds_dart)
rank_test_xgb = rankdata(test_preds_xgb) / len(test_preds_xgb)
rank_test_cb = rankdata(test_preds_cb) / len(test_preds_cb)

# Optimize Ensemble Weights via Nelder-Mead Rank Out-Of-Fold ROC-AUC Optimization
def rank_ensemble_objective(weights):
    w = np.maximum(0, weights)
    s = np.sum(w)
    if s > 0:
        w = w / s
    else:
        w = np.ones_like(w) / len(w)
    blend_oof = (
        w[0] * rank_oof_lgb
        + w[1] * rank_oof_dart
        + w[2] * rank_oof_xgb
        + w[3] * rank_oof_cb
    )
    return -roc_auc_score(y, blend_oof)

opt_result = minimize(
    rank_ensemble_objective,
    x0=[0.30, 0.25, 0.25, 0.20],
    method="Nelder-Mead",
    options={"maxiter": 500, "disp": False},
)

opt_weights = np.maximum(0, opt_result.x)
opt_weights = opt_weights / np.sum(opt_weights)
print(
    f"Optimized Rank Ensemble Weights: LGB={opt_weights[0]:.4f}, DART={opt_weights[1]:.4f}, XGB={opt_weights[2]:.4f}, CB={opt_weights[3]:.4f}"
)

oof_ensemble = (
    opt_weights[0] * rank_oof_lgb
    + opt_weights[1] * rank_oof_dart
    + opt_weights[2] * rank_oof_xgb
    + opt_weights[3] * rank_oof_cb
)
overall_ensemble_auc = roc_auc_score(y, oof_ensemble)
print(f"Overall OOF Rank Ensemble AUC: {overall_ensemble_auc:.5f}")

# Final Test Predictions with Rank-Weighted Blending
final_test_preds = (
    opt_weights[0] * rank_test_lgb
    + opt_weights[1] * rank_test_dart
    + opt_weights[2] * rank_test_xgb
    + opt_weights[3] * rank_test_cb
)

# Generate and verify submission file
submission_df = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "APPETENCY": final_test_preds}
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(
    f"Submission saved successfully to {submission_path} with shape {submission_df.shape}."
)

# Output final evaluation score aligned with the competition metric
print(f"Final Validation Score: {overall_ensemble_auc}")
