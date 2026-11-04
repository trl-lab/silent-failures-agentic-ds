import json
import math
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# =========================================================================
# 0. Global Configuration & Reproducibility
# =========================================================================
SEED = 42
N_SPLITS = 10
TARGET_COL = "Average or Greater than 30"
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================================
# 1. Data Loading & Feature Engineering
# =========================================================================
print("Loading train and test datasets...")
train_df = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
test_df = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

print(f"Train raw shape: {train_df.shape}, Test raw shape: {test_df.shape}")

# Identify Target column
if TARGET_COL not in train_df.columns:
    for col in train_df.columns:
        if "average" in col.lower() or "greater" in col.lower() or "30" in col:
            TARGET_COL = col
            break

train_df[TARGET_COL] = train_df[TARGET_COL].astype(int)

# Detect identifier columns
id_cols = [
    col
    for col in train_df.columns
    if col.lower() in ["id", "index", "opeid", "unitid", "school_id"]
]
if not id_cols and "id" in test_df.columns:
    id_cols = ["id"]

feature_cols = [c for c in train_df.columns if c != TARGET_COL and c not in id_cols]

# Stratified K-Fold partition strictly before feature transformations
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
train_df["fold"] = -1
for fold_idx, (train_idx, val_idx) in enumerate(
    skf.split(train_df, train_df[TARGET_COL])
):
    train_df.loc[val_idx, "fold"] = fold_idx

# Identify numeric and categorical modalities
numeric_cols = []
categorical_cols = []

for col in feature_cols:
    if pd.api.types.is_numeric_dtype(train_df[col]):
        if train_df[col].nunique() <= 10 and not pd.api.types.is_float_dtype(
            train_df[col]
        ):
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)
    else:
        categorical_cols.append(col)

train_feat = train_df.copy()
test_feat = test_df.copy()

# Missing value indicators
for col in numeric_cols:
    if train_df[col].isna().sum() > 0 or test_df[col].isna().sum() > 0:
        train_feat[f"{col}_isna"] = train_feat[col].isna().astype(np.float32)
        test_feat[f"{col}_isna"] = test_feat[col].isna().astype(np.float32)

# Heavy-tailed numeric transformations
for col in numeric_cols:
    min_val = train_df[col].min()
    if min_val >= 0:
        skew_val = train_df[col].skew()
        if skew_val > 1.5:
            train_feat[f"{col}_log1p"] = np.log1p(
                np.clip(train_feat[col].fillna(0), 0, None)
            )
            test_feat[f"{col}_log1p"] = np.log1p(
                np.clip(test_feat[col].fillna(0), 0, None)
            )

# Continuous feature interactions and ratios
if len(numeric_cols) >= 2:
    top_num = numeric_cols[: min(6, len(numeric_cols))]
    for i in range(len(top_num)):
        for j in range(i + 1, len(top_num)):
            c1, c2 = top_num[i], top_num[j]
            train_feat[f"{c1}_div_{c2}"] = train_feat[c1] / (
                train_feat[c2].abs() + 1e-5
            )
            test_feat[f"{c1}_div_{c2}"] = test_feat[c1] / (test_feat[c2].abs() + 1e-5)

            train_feat[f"{c1}_diff_{c2}"] = train_feat[c1] - train_feat[c2]
            test_feat[f"{c1}_diff_{c2}"] = test_feat[c1] - test_feat[c2]

            train_feat[f"{c1}_mult_{c2}"] = train_feat[c1] * train_feat[c2]
            test_feat[f"{c1}_mult_{c2}"] = test_feat[c1] * test_feat[c2]

# Categorical Feature Engineering: Frequency, OOF Target, and Label Encoding
for cat_col in categorical_cols:
    train_feat[cat_col] = train_feat[cat_col].astype(str).fillna("__MISSING__")
    test_feat[cat_col] = test_feat[cat_col].astype(str).fillna("__MISSING__")

    # Frequency Encoding
    freq_map = train_feat[cat_col].value_counts(normalize=True).to_dict()
    train_feat[f"{cat_col}_freq"] = (
        train_feat[cat_col].map(freq_map).fillna(0).astype(np.float32)
    )
    test_feat[f"{cat_col}_freq"] = (
        test_feat[cat_col].map(freq_map).fillna(0).astype(np.float32)
    )

    # Out-of-Fold Target Encoding
    global_target_mean = train_df[TARGET_COL].mean()
    smoothing = 10.0
    train_feat[f"{cat_col}_te"] = np.nan
    for fold_idx in range(N_SPLITS):
        tr_mask = train_feat["fold"] != fold_idx
        va_mask = train_feat["fold"] == fold_idx

        fold_stats = (
            train_feat[tr_mask].groupby(cat_col)[TARGET_COL].agg(["count", "mean"])
        )
        smoothed_te = (
            fold_stats["count"] * fold_stats["mean"] + smoothing * global_target_mean
        ) / (fold_stats["count"] + smoothing)

        train_feat.loc[va_mask, f"{cat_col}_te"] = (
            train_feat.loc[va_mask, cat_col].map(smoothed_te).fillna(global_target_mean)
        )

    full_stats = train_feat.groupby(cat_col)[TARGET_COL].agg(["count", "mean"])
    full_smoothed_te = (
        full_stats["count"] * full_stats["mean"] + smoothing * global_target_mean
    ) / (full_stats["count"] + smoothing)
    test_feat[f"{cat_col}_te"] = (
        test_feat[cat_col].map(full_smoothed_te).fillna(global_target_mean)
    )

    # Label Encoding
    le = LabelEncoder()
    all_categories = list(
        set(
            train_feat[cat_col].unique().tolist() + test_feat[cat_col].unique().tolist()
        )
    )
    le.fit(all_categories)
    train_feat[f"{cat_col}_le"] = le.transform(train_feat[cat_col]).astype(np.int32)
    test_feat[f"{cat_col}_le"] = le.transform(test_feat[cat_col]).astype(np.int32)

# Group Aggregations & Cohort Normalization across all Categorical Columns (Out-of-Fold)
for cat_col in categorical_cols:
    for num_col in numeric_cols:
        col_mean_name = f"{num_col}_grp_{cat_col}_mean"
        col_std_name = f"{num_col}_grp_{cat_col}_std"
        col_min_name = f"{num_col}_grp_{cat_col}_min"
        col_max_name = f"{num_col}_grp_{cat_col}_max"
        col_zscore_name = f"{num_col}_grp_{cat_col}_zscore"

        train_feat[col_mean_name] = np.nan
        train_feat[col_std_name] = np.nan
        train_feat[col_min_name] = np.nan
        train_feat[col_max_name] = np.nan

        global_mean = train_feat[num_col].mean()
        global_std = train_feat[num_col].std() if train_feat[num_col].std() > 0 else 1.0
        global_min = train_feat[num_col].min()
        global_max = train_feat[num_col].max()

        for fold_idx in range(N_SPLITS):
            tr_mask = train_feat["fold"] != fold_idx
            va_mask = train_feat["fold"] == fold_idx

            grp_stats = (
                train_feat[tr_mask]
                .groupby(cat_col)[num_col]
                .agg(["mean", "std", "min", "max"])
            )

            train_feat.loc[va_mask, col_mean_name] = (
                train_feat.loc[va_mask, cat_col]
                .map(grp_stats["mean"])
                .fillna(global_mean)
            )
            train_feat.loc[va_mask, col_std_name] = (
                train_feat.loc[va_mask, cat_col]
                .map(grp_stats["std"])
                .fillna(global_std)
            )
            train_feat.loc[va_mask, col_min_name] = (
                train_feat.loc[va_mask, cat_col]
                .map(grp_stats["min"])
                .fillna(global_min)
            )
            train_feat.loc[va_mask, col_max_name] = (
                train_feat.loc[va_mask, cat_col]
                .map(grp_stats["max"])
                .fillna(global_max)
            )

        full_grp_stats = (
            train_feat.groupby(cat_col)[num_col].agg(["mean", "std", "min", "max"])
        )
        test_feat[col_mean_name] = (
            test_feat[cat_col].map(full_grp_stats["mean"]).fillna(global_mean)
        )
        test_feat[col_std_name] = (
            test_feat[cat_col].map(full_grp_stats["std"]).fillna(global_std)
        )
        test_feat[col_min_name] = (
            test_feat[cat_col].map(full_grp_stats["min"]).fillna(global_min)
        )
        test_feat[col_max_name] = (
            test_feat[cat_col].map(full_grp_stats["max"]).fillna(global_max)
        )

        train_feat[col_zscore_name] = (
            train_feat[num_col] - train_feat[col_mean_name]
        ) / (train_feat[col_std_name].abs() + 1e-5)
        test_feat[col_zscore_name] = (
            test_feat[num_col] - test_feat[col_mean_name]
        ) / (test_feat[col_std_name].abs() + 1e-5)

# Clean infinite and NaN values in created features
processed_feature_cols = [
    c for c in train_feat.columns if c not in [TARGET_COL, "fold"] + id_cols
]
for c in processed_feature_cols:
    if pd.api.types.is_numeric_dtype(train_feat[c]):
        train_feat[c] = train_feat[c].replace([np.inf, -np.inf], np.nan)
        test_feat[c] = test_feat[c].replace([np.inf, -np.inf], np.nan)

# Save Processed Datasets
train_feat.to_parquet(os.path.join(WORKING_DIR, "train_processed.parquet"), index=False)
test_feat.to_parquet(os.path.join(WORKING_DIR, "test_processed.parquet"), index=False)

model_feature_cols = [
    c for c in processed_feature_cols if pd.api.types.is_numeric_dtype(train_feat[c])
]
print(
    f"Feature engineering complete. Total numeric features for modeling: {len(model_feature_cols)}"
)


# =========================================================================
# 2. Model Hyperparameters & Definitions
# =========================================================================
def get_lgb_params(seed: int = 42) -> dict:
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "num_leaves": 24,
        "max_depth": -1,
        "min_child_samples": 25,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.2,
        "reg_lambda": 2.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_params(seed: int = 42) -> dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "learning_rate": 0.02,
        "max_depth": 5,
        "min_child_weight": 4,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.2,
        "reg_lambda": 2.0,
        "random_state": seed,
        "n_estimators": 2000,
        "n_jobs": -1,
    }


def get_cb_params(seed: int = 42) -> dict:
    return {
        "iterations": 2000,
        "learning_rate": 0.02,
        "depth": 5,
        "l2_leaf_reg": 5.0,
        "eval_metric": "AUC",
        "random_seed": seed,
        "verbose": False,
        "thread_count": -1,
    }


# =========================================================================
# 3. Stratified Cross-Validation Training & Inference
# =========================================================================
if "id" in test_feat.columns:
    test_ids = test_feat["id"].values
else:
    test_ids = np.arange(len(test_feat))

X_test_all = test_feat[model_feature_cols].copy()
n_train = len(train_feat)
n_test = len(test_feat)

oof_lgb = np.zeros(n_train, dtype=np.float32)
oof_xgb = np.zeros(n_train, dtype=np.float32)
oof_cb = np.zeros(n_train, dtype=np.float32)

test_preds_lgb = np.zeros(n_test, dtype=np.float32)
test_preds_xgb = np.zeros(n_test, dtype=np.float32)
test_preds_cb = np.zeros(n_test, dtype=np.float32)

y_all = train_feat[TARGET_COL].values

for fold in range(N_SPLITS):
    tr_mask = (train_feat["fold"] != fold).values
    va_mask = (train_feat["fold"] == fold).values

    X_train_f = train_feat.loc[tr_mask, model_feature_cols].copy()
    y_train_f = y_all[tr_mask]
    X_val_f = train_feat.loc[va_mask, model_feature_cols].copy()
    y_val_f = y_all[va_mask]

    # --- A. LightGBM ---
    lgb_train_data = lgb.Dataset(X_train_f, label=y_train_f)
    lgb_val_data = lgb.Dataset(X_val_f, label=y_val_f, reference=lgb_train_data)
    lgb_params = get_lgb_params(seed=SEED + fold)

    lgb_model = lgb.train(
        lgb_params,
        lgb_train_data,
        num_boost_round=2000,
        valid_sets=[lgb_val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_pred_lgb = lgb_model.predict(X_val_f, num_iteration=lgb_model.best_iteration)
    test_pred_lgb_fold = lgb_model.predict(
        X_test_all, num_iteration=lgb_model.best_iteration
    )

    oof_lgb[va_mask] = val_pred_lgb
    test_preds_lgb += test_pred_lgb_fold / N_SPLITS

    # --- B. XGBoost ---
    dtrain = xgb.DMatrix(X_train_f, label=y_train_f)
    dval = xgb.DMatrix(X_val_f, label=y_val_f)
    dtest = xgb.DMatrix(X_test_all)

    xgb_params = get_xgb_params(seed=SEED + fold)
    num_rounds = xgb_params.pop("n_estimators", 1500)
    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=num_rounds,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    val_pred_xgb = xgb_model.predict(dval)
    test_pred_xgb_fold = xgb_model.predict(dtest)

    oof_xgb[va_mask] = val_pred_xgb
    test_preds_xgb += test_pred_xgb_fold / N_SPLITS

    # --- C. CatBoost ---
    cb_params = get_cb_params(seed=SEED + fold)
    cb_model = CatBoostClassifier(**cb_params)
    cb_model.fit(
        X_train_f,
        y_train_f,
        eval_set=(X_val_f, y_val_f),
        early_stopping_rounds=50,
        verbose=False,
    )

    val_pred_cb = cb_model.predict_proba(X_val_f)[:, 1]
    test_pred_cb_fold = cb_model.predict_proba(X_test_all)[:, 1]

    oof_cb[va_mask] = val_pred_cb
    test_preds_cb += test_pred_cb_fold / N_SPLITS

    fold_lgb_auc = roc_auc_score(y_val_f, val_pred_lgb)
    fold_xgb_auc = roc_auc_score(y_val_f, val_pred_xgb)
    fold_cb_auc = roc_auc_score(y_val_f, val_pred_cb)
    print(
        f"Fold {fold+1}/{N_SPLITS} | LightGBM AUC: {fold_lgb_auc:.5f} | XGBoost AUC: {fold_xgb_auc:.5f} | CatBoost AUC: {fold_cb_auc:.5f}"
    )

# =========================================================================
# 4. Out-of-Fold Optimization & Final Scoring
# =========================================================================
overall_lgb_auc = roc_auc_score(y_all, oof_lgb)
overall_xgb_auc = roc_auc_score(y_all, oof_xgb)
overall_cb_auc = roc_auc_score(y_all, oof_cb)

print(
    f"\nOverall OOF Scores -> LightGBM: {overall_lgb_auc:.5f} | XGBoost: {overall_xgb_auc:.5f} | CatBoost: {overall_cb_auc:.5f}"
)


def to_rank(x: np.ndarray) -> np.ndarray:
    return (rankdata(x) - 1.0) / (len(x) - 1.0 + 1e-8)


oof_lgb_rank = to_rank(oof_lgb)
oof_xgb_rank = to_rank(oof_xgb)
oof_cb_rank = to_rank(oof_cb)

test_lgb_rank = to_rank(test_preds_lgb)
test_xgb_rank = to_rank(test_preds_xgb)
test_cb_rank = to_rank(test_preds_cb)


def ensemble_loss(weights):
    w1, w2, w3 = weights
    w_sum = w1 + w2 + w3 + 1e-8
    blend = (w1 * oof_lgb_rank + w2 * oof_xgb_rank + w3 * oof_cb_rank) / w_sum
    return -roc_auc_score(y_all, blend)


init_weights = [0.34, 0.33, 0.33]
bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
opt_res = minimize(
    ensemble_loss,
    init_weights,
    bounds=bounds,
    method="Nelder-Mead",
    options={"maxiter": 300},
)

opt_w = np.maximum(opt_res.x, 0)
if np.sum(opt_w) > 0:
    opt_w = opt_w / np.sum(opt_w)
else:
    opt_w = np.array([1 / 3, 1 / 3, 1 / 3])

print(
    f"Optimal Weights -> LightGBM: {opt_w[0]:.3f}, XGBoost: {opt_w[1]:.3f}, CatBoost: {opt_w[2]:.3f}"
)

final_oof_preds = (
    opt_w[0] * oof_lgb_rank + opt_w[1] * oof_xgb_rank + opt_w[2] * oof_cb_rank
)
final_val_auc = roc_auc_score(y_all, final_oof_preds)

# =========================================================================
# 5. Output Submission Generation & Final Validation Score
# =========================================================================
final_test_preds = (
    opt_w[0] * test_lgb_rank + opt_w[1] * test_xgb_rank + opt_w[2] * test_cb_rank
)

sub_df = pd.DataFrame({"id": test_ids, TARGET_COL: final_test_preds})
sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
sub_df.to_csv(sub_path, index=False)

assert os.path.exists(sub_path), "Submission file was not generated!"
assert len(sub_df) == len(
    test_feat
), f"Submission row count mismatch: {len(sub_df)} vs {len(test_feat)}"
assert list(sub_df.columns) == [
    "id",
    TARGET_COL,
], f"Invalid submission columns: {sub_df.columns.tolist()}"
assert not sub_df[TARGET_COL].isna().any(), "Submission contains NaNs!"

print(f"Final Validation Score: {final_val_auc}")
