import json
import os
from typing import Any, Dict, Optional, Tuple
import torch

from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb

# -----------------------------------------------------------------------------
# 0. Setup and Directory Configuration
# -----------------------------------------------------------------------------
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# 1. Data Ingestion and Target Standardization
# -----------------------------------------------------------------------------
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "antibody_truth"
id_col = (
    "id" if "id" in train_df.columns else ("ID" if "ID" in train_df.columns else None)
)

if target_col in train_df.columns:
    if train_df[target_col].dtype == object or train_df[target_col].dtype == bool:
        mapping = {
            "positive": 1,
            "pos": 1,
            "true": 1,
            "1": 1,
            1: 1,
            True: 1,
            "negative": 0,
            "neg": 0,
            "false": 0,
            "0": 0,
            0: 0,
            False: 0,
        }
        train_df[target_col] = (
            train_df[target_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(mapping)
            .fillna(0)
            .astype(int)
        )
    else:
        train_df[target_col] = train_df[target_col].astype(int)

# -----------------------------------------------------------------------------
# 2. Feature Modality Detection & K-Fold Partitioning
# -----------------------------------------------------------------------------
candidate_cols = [c for c in train_df.columns if c not in [target_col, id_col]]
numeric_cols = []
categorical_cols = []

for c in candidate_cols:
    if pd.api.types.is_numeric_dtype(train_df[c]):
        if train_df[c].nunique() <= 5 and train_df[c].dtype == int:
            categorical_cols.append(c)
        else:
            numeric_cols.append(c)
    else:
        categorical_cols.append(c)

n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
train_df["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df[target_col])):
    train_df.loc[val_idx, "fold"] = fold

# -----------------------------------------------------------------------------
# 3. Feature Engineering: Missingness, Ratios, Interactions, and Encodings
# -----------------------------------------------------------------------------
# 3A. Missingness Indicators
for c in numeric_cols + categorical_cols:
    train_na = train_df[c].isna().astype(np.float32)
    test_na = test_df[c].isna().astype(np.float32)
    if train_na.sum() > 0 or test_na.sum() > 0:
        feat_name = f"{c}_isna"
        train_df[feat_name] = train_na
        test_df[feat_name] = test_na

# 3B. Serological Profile Aggregations, Log1p, and Numeric Interactions
if len(numeric_cols) > 0:
    train_num_mat = train_df[numeric_cols].values
    test_num_mat = test_df[numeric_cols].values

    train_df["row_mean"] = np.nanmean(train_num_mat, axis=1).astype(np.float32)
    test_df["row_mean"] = np.nanmean(test_num_mat, axis=1).astype(np.float32)

    train_df["row_std"] = np.nanstd(train_num_mat, axis=1).astype(np.float32)
    test_df["row_std"] = np.nanstd(test_num_mat, axis=1).astype(np.float32)

    train_df["row_min"] = np.nanmin(train_num_mat, axis=1).astype(np.float32)
    test_df["row_min"] = np.nanmin(test_num_mat, axis=1).astype(np.float32)

    train_df["row_max"] = np.nanmax(train_num_mat, axis=1).astype(np.float32)
    test_df["row_max"] = np.nanmax(test_num_mat, axis=1).astype(np.float32)

    train_df["row_sum"] = np.nansum(train_num_mat, axis=1).astype(np.float32)
    test_df["row_sum"] = np.nansum(test_num_mat, axis=1).astype(np.float32)

    train_df["row_median"] = np.nanmedian(train_num_mat, axis=1).astype(np.float32)
    test_df["row_median"] = np.nanmedian(test_num_mat, axis=1).astype(np.float32)

    train_df["row_ptp"] = (train_df["row_max"] - train_df["row_min"]).astype(np.float32)
    test_df["row_ptp"] = (test_df["row_max"] - test_df["row_min"]).astype(np.float32)

    eps = 1e-6
    for c in numeric_cols:
        train_df[f"{c}_rel_mean_ratio"] = (
            (train_df[c] + eps) / (train_df["row_mean"].abs() + eps)
        ).astype(np.float32)
        test_df[f"{c}_rel_mean_ratio"] = (
            (test_df[c] + eps) / (test_df["row_mean"].abs() + eps)
        ).astype(np.float32)

        train_df[f"{c}_rel_mean_diff"] = (
            train_df[c] - train_df["row_mean"]
        ).astype(np.float32)
        test_df[f"{c}_rel_mean_diff"] = (
            test_df[c] - test_df["row_mean"]
        ).astype(np.float32)

        train_df[f"{c}_rel_max_ratio"] = (
            (train_df[c] + eps) / (train_df["row_max"].abs() + eps)
        ).astype(np.float32)
        test_df[f"{c}_rel_max_ratio"] = (
            (test_df[c] + eps) / (test_df["row_max"].abs() + eps)
        ).astype(np.float32)

        min_val = train_df[c].min()
        if pd.notna(min_val) and min_val >= 0:
            skew_val = train_df[c].skew()
            if skew_val > 1.5:
                train_df[f"{c}_log1p"] = np.log1p(train_df[c].clip(lower=0)).astype(np.float32)
                test_df[f"{c}_log1p"] = np.log1p(test_df[c].clip(lower=0)).astype(np.float32)

    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            col1 = numeric_cols[i]
            col2 = numeric_cols[j]

            diff_name = f"{col1}_minus_{col2}"
            train_df[diff_name] = (train_df[col1] - train_df[col2]).astype(np.float32)
            test_df[diff_name] = (test_df[col1] - test_df[col2]).astype(np.float32)

            ratio_name = f"{col1}_div_{col2}"
            train_df[ratio_name] = (
                (train_df[col1] + eps) / (train_df[col2].abs() + eps)
            ).astype(np.float32)
            test_df[ratio_name] = (
                (test_df[col1] + eps) / (test_df[col2].abs() + eps)
            ).astype(np.float32)

# 3C. Categorical Encoding (Frequency, Target Encoding, Label Encoding)
for cat in categorical_cols:
    train_df[cat] = train_df[cat].fillna("__MISSING__").astype(str)
    test_df[cat] = test_df[cat].fillna("__MISSING__").astype(str)

    freq_map = train_df[cat].value_counts(normalize=True).to_dict()
    train_df[f"{cat}_freq"] = train_df[cat].map(freq_map).fillna(0).astype(np.float32)
    test_df[f"{cat}_freq"] = test_df[cat].map(freq_map).fillna(0).astype(np.float32)

    smooth_weight = 10.0
    global_target_mean = train_df[target_col].mean()
    te_col = f"{cat}_target_enc"
    train_df[te_col] = np.nan

    for fold in range(n_splits):
        tr_fold = train_df[train_df["fold"] != fold]
        val_fold = train_df[train_df["fold"] == fold]
        tr_target_mean = tr_fold[target_col].mean()

        counts = tr_fold.groupby(cat)[target_col].count()
        sums = tr_fold.groupby(cat)[target_col].sum()
        smoothed = (sums + smooth_weight * tr_target_mean) / (
            counts + smooth_weight
        )

        train_df.loc[train_df["fold"] == fold, te_col] = (
            val_fold[cat].map(smoothed).fillna(tr_target_mean)
        )

    full_counts = train_df.groupby(cat)[target_col].count()
    full_sums = train_df.groupby(cat)[target_col].sum()
    full_smoothed = (full_sums + smooth_weight * global_target_mean) / (
        full_counts + smooth_weight
    )
    test_df[te_col] = (
        test_df[cat].map(full_smoothed).fillna(global_target_mean).astype(np.float32)
    )

    le = LabelEncoder()
    all_cats = list(set(train_df[cat].unique()).union(set(test_df[cat].unique())))
    le.fit(all_cats)
    train_df[f"{cat}_le"] = le.transform(train_df[cat])
    test_df[f"{cat}_le"] = le.transform(test_df[cat])

# 3D. Group-based Aggregations
if len(categorical_cols) > 0 and len(numeric_cols) > 0:
    primary_cat = categorical_cols[0]
    primary_num = numeric_cols[0]

    train_df[f"{primary_num}_by_{primary_cat}_mean"] = np.nan
    train_df[f"{primary_num}_by_{primary_cat}_std"] = np.nan

    for fold in range(n_splits):
        tr_fold = train_df[train_df["fold"] != fold]
        val_fold = train_df[train_df["fold"] == fold]

        grp_mean = tr_fold.groupby(primary_cat)[primary_num].mean()
        grp_std = tr_fold.groupby(primary_cat)[primary_num].std()

        global_mean = tr_fold[primary_num].mean()
        global_std = (
            tr_fold[primary_num].std() if pd.notna(tr_fold[primary_num].std()) else 0.0
        )

        train_df.loc[
            train_df["fold"] == fold, f"{primary_num}_by_{primary_cat}_mean"
        ] = (val_fold[primary_cat].map(grp_mean).fillna(global_mean))
        train_df.loc[
            train_df["fold"] == fold, f"{primary_num}_by_{primary_cat}_std"
        ] = (val_fold[primary_cat].map(grp_std).fillna(global_std))

    full_grp_mean = train_df.groupby(primary_cat)[primary_num].mean()
    full_grp_std = train_df.groupby(primary_cat)[primary_num].std()
    test_df[f"{primary_num}_by_{primary_cat}_mean"] = (
        test_df[primary_cat]
        .map(full_grp_mean)
        .fillna(train_df[primary_num].mean())
        .astype(np.float32)
    )
    test_df[f"{primary_num}_by_{primary_cat}_std"] = (
        test_df[primary_cat]
        .map(full_grp_std)
        .fillna(train_df[primary_num].std())
        .astype(np.float32)
    )

engineered_feature_cols = [
    c
    for c in train_df.columns
    if c not in [target_col, "fold", id_col] + categorical_cols
]

for col in engineered_feature_cols:
    train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan)
    test_df[col] = test_df[col].replace([np.inf, -np.inf], np.nan)
    median_val = train_df[col].median()
    if pd.isna(median_val):
        median_val = 0.0
    train_df[col] = train_df[col].fillna(median_val)
    test_df[col] = test_df[col].fillna(median_val)

train_df.to_parquet("./working/train_features.parquet", index=False)
test_df.to_parquet("./working/test_features.parquet", index=False)

metadata = {
    "target_col": target_col,
    "id_col": id_col,
    "n_splits": n_splits,
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "feature_cols": engineered_feature_cols,
    "n_train_samples": len(train_df),
    "n_test_samples": len(test_df),
    "n_features": len(engineered_feature_cols),
}
with open("./working/feature_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)


# -----------------------------------------------------------------------------
# 4. Multi-Paradigm Cross-Validation Training & Out-of-Fold Optimization
# -----------------------------------------------------------------------------
n_train = len(train_df)
n_test = len(test_df)
feature_cols = engineered_feature_cols

oof_preds_lgb = np.zeros(n_train, dtype=np.float32)
test_preds_lgb = np.zeros(n_test, dtype=np.float32)

oof_preds_xgb = np.zeros(n_train, dtype=np.float32)
test_preds_xgb = np.zeros(n_test, dtype=np.float32)

oof_preds_cb = np.zeros(n_train, dtype=np.float32)
test_preds_cb = np.zeros(n_test, dtype=np.float32)

oof_preds_et = np.zeros(n_train, dtype=np.float32)
test_preds_et = np.zeros(n_test, dtype=np.float32)

y_all = train_df[target_col].values.astype(np.float32)
X_all = train_df[feature_cols].values.astype(np.float32)
X_test_arr = test_df[feature_cols].values.astype(np.float32)

for fold in range(n_splits):
    train_mask = train_df["fold"] != fold
    val_mask = train_df["fold"] == fold

    X_tr, y_tr = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    val_indices = np.where(val_mask)[0]

    # --- 4A. LightGBM ---
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 24,
        "max_depth": 5,
        "min_child_samples": 15,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 1500,
        "random_state": 42 + fold,
        "n_jobs": -1,
        "verbose": -1,
    }

    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_lgb_raw = lgb_model.predict_proba(X_val)[:, 1]
    test_lgb_raw = lgb_model.predict_proba(X_test_arr)[:, 1]
    oof_preds_lgb[val_indices] = rankdata(val_lgb_raw) / len(val_lgb_raw)
    test_preds_lgb += (rankdata(test_lgb_raw) / len(test_lgb_raw)) / n_splits

    # --- 4B. XGBoost ---
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.2,
        "reg_lambda": 2.0,
        "n_estimators": 1500,
        "random_state": 42 + fold,
        "n_jobs": -1,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
    }

    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    val_xgb_raw = xgb_model.predict_proba(X_val)[:, 1]
    test_xgb_raw = xgb_model.predict_proba(X_test_arr)[:, 1]
    oof_preds_xgb[val_indices] = rankdata(val_xgb_raw) / len(val_xgb_raw)
    test_preds_xgb += (rankdata(test_xgb_raw) / len(test_xgb_raw)) / n_splits

    # --- 4C. CatBoost ---
    cb_params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "learning_rate": 0.03,
        "depth": 5,
        "l2_leaf_reg": 3.0,
        "iterations": 1500,
        "random_seed": 42 + fold,
        "verbose": False,
        "early_stopping_rounds": 50,
    }

    cb_model = CatBoostClassifier(**cb_params)
    cb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    val_cb_raw = cb_model.predict_proba(X_val)[:, 1]
    test_cb_raw = cb_model.predict_proba(X_test_arr)[:, 1]
    oof_preds_cb[val_indices] = rankdata(val_cb_raw) / len(val_cb_raw)
    test_preds_cb += (rankdata(test_cb_raw) / len(test_cb_raw)) / n_splits

    # --- 4D. ExtraTrees ---
    et_params = {
        "n_estimators": 400,
        "max_depth": 10,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "random_state": 42 + fold,
        "n_jobs": -1,
    }

    et_model = ExtraTreesClassifier(**et_params)
    et_model.fit(X_tr, y_tr)

    val_et_raw = et_model.predict_proba(X_val)[:, 1]
    test_et_raw = et_model.predict_proba(X_test_arr)[:, 1]
    oof_preds_et[val_indices] = rankdata(val_et_raw) / len(val_et_raw)
    test_preds_et += (rankdata(test_et_raw) / len(test_et_raw)) / n_splits

    fold_auc_lgb = roc_auc_score(y_val, val_lgb_raw)
    fold_auc_xgb = roc_auc_score(y_val, val_xgb_raw)
    fold_auc_cb = roc_auc_score(y_val, val_cb_raw)
    fold_auc_et = roc_auc_score(y_val, val_et_raw)
    print(
        f"Fold {fold} AUC -> LGB: {fold_auc_lgb:.5f} | XGB: {fold_auc_xgb:.5f} | CB: {fold_auc_cb:.5f} | ET: {fold_auc_et:.5f}"
    )

# -----------------------------------------------------------------------------
# 5. Rank Normalization & Ensemble Optimization
# -----------------------------------------------------------------------------
oof_lgb_rank = rankdata(oof_preds_lgb) / len(oof_preds_lgb)
oof_xgb_rank = rankdata(oof_preds_xgb) / len(oof_preds_xgb)
oof_cb_rank = rankdata(oof_preds_cb) / len(oof_preds_cb)
oof_et_rank = rankdata(oof_preds_et) / len(oof_preds_et)

test_lgb_rank = rankdata(test_preds_lgb) / len(test_preds_lgb)
test_xgb_rank = rankdata(test_preds_xgb) / len(test_preds_xgb)
test_cb_rank = rankdata(test_preds_cb) / len(test_preds_cb)
test_et_rank = rankdata(test_preds_et) / len(test_preds_et)

auc_lgb_global = roc_auc_score(y_all, oof_lgb_rank)
auc_xgb_global = roc_auc_score(y_all, oof_xgb_rank)
auc_cb_global = roc_auc_score(y_all, oof_cb_rank)
auc_et_global = roc_auc_score(y_all, oof_et_rank)
print(
    f"Global OOF AUC -> LGB: {auc_lgb_global:.5f} | XGB: {auc_xgb_global:.5f} | CB: {auc_cb_global:.5f} | ET: {auc_et_global:.5f}"
)


def loss_func(weights):
    weights = np.maximum(0, np.array(weights))
    w_sum = np.sum(weights)
    if w_sum <= 0:
        return 0.0
    w = weights / w_sum
    blend = (
        w[0] * oof_lgb_rank
        + w[1] * oof_xgb_rank
        + w[2] * oof_cb_rank
        + w[3] * oof_et_rank
    )
    return -roc_auc_score(y_all, blend)


init_weights = [0.4, 0.2, 0.3, 0.1]
bounds = [(0, 1), (0, 1), (0, 1), (0, 1)]
res = minimize(
    loss_func,
    init_weights,
    bounds=bounds,
    method="Nelder-Mead",
    options={"maxiter": 1000, "adaptive": True},
)

best_w = np.maximum(0, res.x)
best_w = best_w / np.sum(best_w)
w_lgb, w_xgb, w_cb, w_et = best_w
print(
    f"Optimal Ensemble Weights -> LGB: {w_lgb:.3f}, XGB: {w_xgb:.3f}, CB: {w_cb:.3f}, ET: {w_et:.3f}"
)

final_oof_preds = (
    w_lgb * oof_lgb_rank
    + w_xgb * oof_xgb_rank
    + w_cb * oof_cb_rank
    + w_et * oof_et_rank
)
final_validation_score = float(roc_auc_score(y_all, final_oof_preds))

final_test_preds = (
    w_lgb * test_lgb_rank
    + w_xgb * test_xgb_rank
    + w_cb * test_cb_rank
    + w_et * test_et_rank
)

# -----------------------------------------------------------------------------
# 7. Output Submission File
# -----------------------------------------------------------------------------
submission_df = pd.DataFrame(
    {"id": np.arange(len(test_df), dtype=int), "antibody_truth": final_test_preds}
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Final Validation Score: {final_validation_score}")