import gc
import json
import os
import re
import warnings
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import xgboost as xgb

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------
# 1. Directories & Data Loading
# -------------------------------------------------------------------------
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

train_path = os.path.join(INPUT_DIR, "train.parquet")
test_path = os.path.join(INPUT_DIR, "test.parquet")

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Amount"
assert (
    target_col in train_df.columns
), f"Target column {target_col} missing from train data"

# Drop rows where target is NaN in train
train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)
y_train = train_df[target_col].astype(np.float64)

id_cols = [
    col
    for col in train_df.columns
    if col.lower() in ["id", "foreign gift and contract id", "opeid", "ope id"]
    and col != target_col
]

n_train = len(train_df)
n_test = len(test_df)

# Combine for structural consistency in schema detection
combined_df = pd.concat(
    [train_df.drop(columns=[target_col]), test_df], axis=0, ignore_index=True
)

# -------------------------------------------------------------------------
# 2. Date Feature Engineering & Temporal Differentials
# -------------------------------------------------------------------------
datetime_cols = []
for col in combined_df.columns:
    if col in id_cols:
        continue
    if pd.api.types.is_datetime64_any_dtype(combined_df[col]):
        datetime_cols.append(col)
    elif combined_df[col].dtype == "object":
        sample = combined_df[col].dropna().head(20).astype(str)
        if len(sample) > 0:
            try:
                parsed = pd.to_datetime(sample, errors="raise")
                datetime_cols.append(col)
            except (ValueError, TypeError):
                pass

date_features = pd.DataFrame(index=combined_df.index)
parsed_dates = {}

for col in datetime_cols:
    parsed_date = pd.to_datetime(combined_df[col], errors="coerce")
    invalid_mask = (parsed_date.dt.year < 1900) | (parsed_date.dt.year > 2100)
    parsed_date = parsed_date.mask(invalid_mask)
    parsed_date = pd.to_datetime(parsed_date, errors="coerce")
    parsed_dates[col] = parsed_date
    prefix = f"date_{col}_"
    date_features[prefix + "year"] = parsed_date.dt.year.fillna(-1).astype(np.int32)
    date_features[prefix + "month"] = parsed_date.dt.month.fillna(-1).astype(np.int32)
    date_features[prefix + "day"] = parsed_date.dt.day.fillna(-1).astype(np.int32)
    date_features[prefix + "dayofweek"] = parsed_date.dt.dayofweek.fillna(-1).astype(
        np.int32
    )
    date_features[prefix + "dayofyear"] = parsed_date.dt.dayofyear.fillna(-1).astype(
        np.int32
    )
    date_features[prefix + "quarter"] = parsed_date.dt.quarter.fillna(-1).astype(
        np.int32
    )
    date_features[prefix + "is_month_start"] = parsed_date.dt.is_month_start.fillna(
        False
    ).astype(np.int8)
    date_features[prefix + "is_month_end"] = parsed_date.dt.is_month_end.fillna(
        False
    ).astype(np.int8)
    ts = (parsed_date - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
    date_features[prefix + "timestamp"] = ts.fillna(-1).astype(np.float64)

# Date differences / spans between pairs of date columns
date_col_list = list(parsed_dates.keys())
for i in range(len(date_col_list)):
    for j in range(i + 1, len(date_col_list)):
        c1, c2 = date_col_list[i], date_col_list[j]
        diff_days = (parsed_dates[c2] - parsed_dates[c1]).dt.total_seconds() / 86400.0
        date_features[f"diff_days_{c1}_vs_{c2}"] = diff_days.fillna(-9999).astype(
            np.float32
        )

# -------------------------------------------------------------------------
# 3. Categorical & Text Feature Engineering
# -------------------------------------------------------------------------
non_date_cols = [
    c for c in combined_df.columns if c not in datetime_cols and c not in id_cols
]
cat_cols = [
    c
    for c in non_date_cols
    if combined_df[c].dtype == "object" or combined_df[c].dtype.name == "category"
]
num_cols = [c for c in non_date_cols if c not in cat_cols]

text_features = pd.DataFrame(index=combined_df.index)
for col in cat_cols:
    str_series = combined_df[col].fillna("").astype(str)
    text_features[f"{col}_char_len"] = str_series.apply(len).astype(np.int32)
    text_features[f"{col}_word_count"] = str_series.apply(
        lambda x: len(x.split()) if x else 0
    ).astype(np.int32)
    text_features[f"{col}_digit_count"] = str_series.apply(
        lambda x: sum(c.isdigit() for c in x)
    ).astype(np.int32)
    text_features[f"{col}_upper_count"] = str_series.apply(
        lambda x: sum(c.isupper() for c in x)
    ).astype(np.int32)
    text_features[f"{col}_is_missing"] = combined_df[col].isna().astype(np.int8)

# Frequency Encoding for categorical features
freq_features = pd.DataFrame(index=combined_df.index)
for col in cat_cols:
    val_counts = train_df[col].value_counts(dropna=True)
    freq_features[f"{col}_freq_count"] = (
        combined_df[col].map(val_counts).fillna(0).astype(np.int32)
    )
    freq_features[f"{col}_freq_ratio"] = (
        freq_features[f"{col}_freq_count"] / max(n_train, 1)
    ).astype(np.float32)

# Pairwise Categorical Interaction Frequencies
if len(cat_cols) >= 2:
    for i in range(len(cat_cols)):
        for j in range(i + 1, min(len(cat_cols), i + 4)):
            col1, col2 = cat_cols[i], cat_cols[j]
            pair_train = train_df[col1].astype(str) + "___" + train_df[col2].astype(str)
            pair_comb = (
                combined_df[col1].astype(str) + "___" + combined_df[col2].astype(str)
            )
            pair_counts = pair_train.value_counts(dropna=True)
            freq_features[f"pair_freq_{col1}_{col2}"] = (
                pair_comb.map(pair_counts).fillna(0).astype(np.int32)
            )

# Label Encoding for tree models
label_features = pd.DataFrame(index=combined_df.index)
for col in cat_cols:
    le = LabelEncoder()
    str_col = combined_df[col].fillna("__MISSING__").astype(str)
    label_features[f"{col}_le"] = le.fit_transform(str_col).astype(np.int32)

# Per-column TF-IDF and Truncated SVD for text and categorical fields
svd_feature_list = []
for col in cat_cols:
    str_series_comb = combined_df[col].fillna("").astype(str)
    str_series_train = train_df[col].fillna("").astype(str)
    if str_series_train.nunique() > 2:
        try:
            n_comp = 5 if str_series_train.nunique() > 20 else 2
            tfidf = TfidfVectorizer(
                max_features=1000,
                stop_words="english",
                token_pattern=r"(?u)\b\w+\b",
                min_df=2,
            )
            tfidf_train = tfidf.fit_transform(str_series_train)
            actual_comp = min(n_comp, tfidf_train.shape[1] - 1)
            if actual_comp >= 1:
                svd = TruncatedSVD(n_components=actual_comp, random_state=42)
                svd.fit(tfidf_train)
                tfidf_comb = tfidf.transform(str_series_comb)
                svd_comb = svd.transform(tfidf_comb)
                clean_c = re.sub(r"[\[\]<>{}:,\"\']", "_", str(col)).strip()
                col_svd_df = pd.DataFrame(
                    svd_comb,
                    columns=[f"svd_{clean_c}_{k}" for k in range(actual_comp)],
                    index=combined_df.index,
                ).astype(np.float32)
                svd_feature_list.append(col_svd_df)
        except Exception:
            pass

if len(svd_feature_list) > 0:
    svd_features = pd.concat(svd_feature_list, axis=1)
else:
    svd_features = pd.DataFrame(index=combined_df.index)

# Hierarchical group aggregations (frequency, duration statistics, contract count)
group_features = pd.DataFrame(index=combined_df.index)
group_cat_cols = [c for c in cat_cols if 2 <= combined_df[c].nunique() <= 5000]
duration_cols = [c for c in date_features.columns if "diff_days" in c or "timestamp" in c]
agg_num_cols = list(num_cols) + duration_cols

for g_col in group_cat_cols:
    clean_g = re.sub(r"[\[\]<>{}:,\"\']", "_", str(g_col)).strip()
    group_counts = combined_df.groupby(g_col)[g_col].transform("count").astype(np.float32)
    group_features[f"grp_count_{clean_g}"] = group_counts

    for n_col in agg_num_cols[:6]:
        if n_col in combined_df.columns:
            series = combined_df[n_col]
        elif n_col in date_features.columns:
            series = date_features[n_col]
        else:
            continue
        grp = series.groupby(combined_df[g_col])
        clean_n = re.sub(r"[\[\]<>{}:,\"\']", "_", str(n_col)).strip()
        group_features[f"grp_{clean_g}_{clean_n}_mean"] = grp.transform("mean").astype(np.float32)
        group_features[f"grp_{clean_g}_{clean_n}_std"] = grp.transform("std").fillna(0).astype(np.float32)

if len(group_cat_cols) >= 2:
    for i in range(min(len(group_cat_cols), 4)):
        for j in range(i + 1, min(len(group_cat_cols), 5)):
            c1, c2 = group_cat_cols[i], group_cat_cols[j]
            clean_c1 = re.sub(r"[\[\]<>{}:,\"\']", "_", str(c1)).strip()
            clean_c2 = re.sub(r"[\[\]<>{}:,\"\']", "_", str(c2)).strip()
            clean_pair = f"{clean_c1}_x_{clean_c2}"
            pair_series = combined_df[c1].astype(str) + "___" + combined_df[c2].astype(str)
            group_features[f"hier_count_{clean_pair}"] = pair_series.map(pair_series.value_counts()).astype(np.float32)

# -------------------------------------------------------------------------
# 4. Numerical Features Handling
# -------------------------------------------------------------------------
numeric_features = pd.DataFrame(index=combined_df.index)
for col in num_cols:
    numeric_features[f"{col}_is_missing"] = combined_df[col].isna().astype(np.int8)
    median_val = train_df[col].median()
    if pd.isna(median_val):
        median_val = 0.0
    numeric_features[col] = combined_df[col].fillna(median_val).astype(np.float32)
    numeric_features[f"{col}_log1p"] = np.log1p(
        np.maximum(0, numeric_features[col])
    ).astype(np.float32)

# -------------------------------------------------------------------------
# 5. Assemble Full Feature Matrix & Sanitize Names
# -------------------------------------------------------------------------
full_features = pd.concat(
    [
        date_features,
        text_features,
        freq_features,
        group_features,
        label_features,
        svd_features,
        numeric_features,
    ],
    axis=1,
)

# Sanitize feature column names for LightGBM / XGBoost compatibility
clean_columns = [
    re.sub(r"[\[\]<>{}:,\"\']", "_", str(c)).strip() for c in full_features.columns
]
full_features.columns = clean_columns

X_train_df = full_features.iloc[:n_train].reset_index(drop=True)
X_test_df = full_features.iloc[n_train:].reset_index(drop=True)

# -------------------------------------------------------------------------
# 6. Stratified K-Fold Split & Out-of-Fold Target Encoding
# -------------------------------------------------------------------------
N_SPLITS = 5
RANDOM_STATE = 42

num_bins = min(10, len(np.unique(y_train)))
folds_array = np.zeros(n_train, dtype=np.int32)

try:
    strat_bins = pd.qcut(y_train, q=num_bins, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_df, strat_bins)):
        folds_array[val_idx] = fold
except Exception:
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_df)):
        folds_array[val_idx] = fold

X_train_df["fold"] = folds_array

# Out-of-Fold (OOF) Target Encoding strictly computed on training folds
top_target_cat_cols = [
    c for c in cat_cols if train_df[c].nunique() > 1 and train_df[c].nunique() < 5000
]

global_mean = float(y_train.mean())
global_std = float(y_train.std()) if len(y_train) > 1 else 1.0
smoothing = 10.0

for col in top_target_cat_cols:
    clean_col_name = re.sub(r"[\[\]<>{}:,\"\']", "_", str(col)).strip()
    target_enc_mean_name = f"{clean_col_name}_target_mean"
    target_enc_std_name = f"{clean_col_name}_target_std"
    X_train_df[target_enc_mean_name] = np.nan
    X_train_df[target_enc_std_name] = np.nan
    X_test_df[target_enc_mean_name] = np.nan
    X_test_df[target_enc_std_name] = np.nan

    train_col_series = train_df[col].astype(str)
    stats = y_train.groupby(train_col_series).agg(["count", "mean", "std"])
    stats["std"] = stats["std"].fillna(0.0)
    smoothed_mean_full = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
        stats["count"] + smoothing
    )
    smoothed_std_full = (stats["count"] * stats["std"] + smoothing * global_std) / (
        stats["count"] + smoothing
    )

    X_test_df[target_enc_mean_name] = (
        test_df[col]
        .astype(str)
        .map(smoothed_mean_full)
        .fillna(global_mean)
        .astype(np.float32)
    )
    X_test_df[target_enc_std_name] = (
        test_df[col]
        .astype(str)
        .map(smoothed_std_full)
        .fillna(global_std)
        .astype(np.float32)
    )

    for fold in range(N_SPLITS):
        tr_idx = np.where(folds_array != fold)[0]
        va_idx = np.where(folds_array == fold)[0]

        tr_y = y_train.iloc[tr_idx]
        tr_cats = train_col_series.iloc[tr_idx]
        fold_stats = tr_y.groupby(tr_cats).agg(["count", "mean", "std"])
        fold_stats["std"] = fold_stats["std"].fillna(0.0)
        fold_mean = float(tr_y.mean())
        fold_std = float(tr_y.std()) if len(tr_y) > 1 else 1.0

        smoothed_fold_mean = (
            fold_stats["count"] * fold_stats["mean"] + smoothing * fold_mean
        ) / (fold_stats["count"] + smoothing)
        smoothed_fold_std = (
            fold_stats["count"] * fold_stats["std"] + smoothing * fold_std
        ) / (fold_stats["count"] + smoothing)

        va_cats = train_col_series.iloc[va_idx]
        X_train_df.loc[va_idx, target_enc_mean_name] = (
            va_cats.map(smoothed_fold_mean).fillna(fold_mean).astype(np.float32)
        )
        X_train_df.loc[va_idx, target_enc_std_name] = (
            va_cats.map(smoothed_fold_std).fillna(fold_std).astype(np.float32)
        )


# -------------------------------------------------------------------------
# 7. Model Configurations
# -------------------------------------------------------------------------
def get_lgb_model_params(random_state: int = 42) -> dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "num_leaves": 45,
        "max_depth": 7,
        "min_child_samples": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_model_params(random_state: int = 42) -> dict:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.80,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def get_cb_model_params(random_state: int = 42) -> dict:
    return {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 2500,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": random_state,
        "verbose": False,
    }


# -------------------------------------------------------------------------
# 8. Stratified 5-Fold Training Loop (LightGBM, XGBoost, & CatBoost)
# -------------------------------------------------------------------------
feature_cols = [c for c in X_train_df.columns if c != "fold"]
X_test = X_test_df[feature_cols].copy()
y_true = y_train.values

oof_lgb = np.zeros(len(X_train_df), dtype=np.float64)
oof_xgb = np.zeros(len(X_train_df), dtype=np.float64)
oof_cb = np.zeros(len(X_train_df), dtype=np.float64)

test_lgb_preds = np.zeros(len(X_test_df), dtype=np.float64)
test_xgb_preds = np.zeros(len(X_test_df), dtype=np.float64)
test_cb_preds = np.zeros(len(X_test_df), dtype=np.float64)

for fold in range(N_SPLITS):
    tr_idx = np.where(X_train_df["fold"] != fold)[0]
    va_idx = np.where(X_train_df["fold"] == fold)[0]

    X_tr = X_train_df.iloc[tr_idx][feature_cols]
    y_tr = y_train.iloc[tr_idx]

    X_va = X_train_df.iloc[va_idx][feature_cols]
    y_va = y_train.iloc[va_idx]

    # --- LightGBM Regressor ---
    lgb_params = get_lgb_model_params(random_state=42 + fold)
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    pred_va_lgb = model_lgb.predict(X_va)
    oof_lgb[va_idx] = pred_va_lgb
    test_lgb_preds += model_lgb.predict(X_test) / N_SPLITS

    # --- XGBoost Regressor ---
    xgb_params = get_xgb_model_params(random_state=42 + fold)
    xgb_params["early_stopping_rounds"] = 50
    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    pred_va_xgb = model_xgb.predict(X_va)
    oof_xgb[va_idx] = pred_va_xgb
    test_xgb_preds += model_xgb.predict(X_test) / N_SPLITS

    # --- CatBoost Regressor ---
    cb_params = get_cb_model_params(random_state=42 + fold)
    cb_params["early_stopping_rounds"] = 50
    model_cb = CatBoostRegressor(**cb_params)
    model_cb.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        verbose=False,
    )

    pred_va_cb = model_cb.predict(X_va)
    oof_cb[va_idx] = pred_va_cb
    test_cb_preds += model_cb.predict(X_test) / N_SPLITS


# -------------------------------------------------------------------------
# 9. Model Blending & Out-of-Fold Evaluation
# -------------------------------------------------------------------------
def loss_objective(weights):
    w1, w2, w3 = weights
    pred = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cb
    pred = np.clip(pred, 0, None)
    return np.sqrt(mean_squared_error(y_true, pred))


opt_res = minimize(
    loss_objective,
    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0},
    method="SLSQP",
)

w_lgb, w_xgb, w_cb = (
    opt_res.x if opt_res.success else (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
)

oof_ensemble = np.clip(w_lgb * oof_lgb + w_xgb * oof_xgb + w_cb * oof_cb, 0, None)
final_val_score = float(np.sqrt(mean_squared_error(y_true, oof_ensemble)))

# -------------------------------------------------------------------------
# 10. Generate Final Test Predictions & Save Submission
# -------------------------------------------------------------------------
final_test_predictions = np.clip(
    w_lgb * test_lgb_preds + w_xgb * test_xgb_preds + w_cb * test_cb_preds, 0, None
)

submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df = pd.DataFrame(
    {
        "id": np.arange(len(final_test_predictions), dtype=np.int64),
        "Amount": final_test_predictions,
    }
)

assert len(submission_df) == len(
    test_df
), f"Submission row count {len(submission_df)} does not match test set {len(test_df)}"
assert not submission_df["Amount"].isna().any(), "Submission contains NaN predictions"
assert list(submission_df.columns) == ["id", "Amount"], "Incorrect submission columns"

submission_df.to_csv(submission_path, index=False)
gc.collect()

print(f"Final Validation Score: {final_val_score}")