import json
import os
import re
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. Setup & Data Ingestion
# ---------------------------------------------------------
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

train_path = os.path.join(INPUT_DIR, "train.parquet")
test_path = os.path.join(INPUT_DIR, "test.parquet")

print("Loading raw parquet data...")
df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

target_col = "Disbursements USD (Trust funds)"
if target_col not in df_train.columns:
    for col in df_train.columns:
        if "disbursement" in col.lower():
            target_col = col
            break

print(f"Target column: '{target_col}'")
print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")

y_train = df_train[target_col].copy()
df_train_features = df_train.drop(columns=[target_col]).copy()
df_test_features = df_test.copy()

df_train_features["__is_train__"] = 1
df_test_features["__is_train__"] = 0

combined_df = pd.concat(
    [df_train_features, df_test_features], axis=0, ignore_index=True
)

# ---------------------------------------------------------
# 2. Date Parsing & Lifecycle Feature Engineering
# ---------------------------------------------------------
date_cols = []
for col in combined_df.columns:
    if col == "__is_train__":
        continue
    if any(k in col.lower() for k in ["date", "year", "time", "day", "month"]):
        date_cols.append(col)
    elif combined_df[col].dtype == "object":
        sample = combined_df[col].dropna().head(10).astype(str)
        if (
            len(sample) > 0
            and sample.str.contains(
                r"\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}"
            ).all()
        ):
            if col not in date_cols:
                date_cols.append(col)

parsed_date_cols = []
for col in date_cols:
    try:
        parsed_series = pd.to_datetime(combined_df[col], errors="coerce")
        if parsed_series.notnull().sum() > 0.1 * len(combined_df):
            date_name = f"dt_{col}"
            combined_df[date_name] = parsed_series
            parsed_date_cols.append(date_name)

            combined_df[f"{col}_year"] = parsed_series.dt.year.astype("float32")
            combined_df[f"{col}_month"] = parsed_series.dt.month.astype("float32")
            combined_df[f"{col}_quarter"] = parsed_series.dt.quarter.astype("float32")
            combined_df[f"{col}_day"] = parsed_series.dt.day.astype("float32")
            combined_df[f"{col}_dayofweek"] = parsed_series.dt.dayofweek.astype(
                "float32"
            )
            combined_df[f"{col}_dayofyear"] = parsed_series.dt.dayofyear.astype(
                "float32"
            )
            combined_df[f"{col}_is_month_end"] = parsed_series.dt.is_month_end.astype(
                "float32"
            )
    except Exception:
        continue

for i in range(len(parsed_date_cols)):
    for j in range(i + 1, len(parsed_date_cols)):
        d1 = parsed_date_cols[i]
        d2 = parsed_date_cols[j]
        diff_days = (combined_df[d2] - combined_df[d1]).dt.total_seconds() / (24 * 3600)
        col_name = f"diff_days_{d2}_minus_{d1}"
        combined_df[col_name] = diff_days.astype("float32")

closing_cols = [
    c
    for c in parsed_date_cols
    if any(k in c.lower() for k in ["closing", "end", "close"])
]
effective_cols = [
    c
    for c in parsed_date_cols
    if any(k in c.lower() for k in ["effective", "start", "approval", "agreement"])
]
orig_closing_cols = [
    c for c in closing_cols if any(k in c.lower() for k in ["orig", "first", "initial"])
]
revised_closing_cols = [
    c for c in closing_cols if any(k in c.lower() for k in ["rev", "curr", "actual"])
]

if closing_cols and effective_cols:
    eff_col = effective_cols[0]
    close_col = revised_closing_cols[0] if revised_closing_cols else closing_cols[0]
    total_proj_days = (
        combined_df[close_col] - combined_df[eff_col]
    ).dt.total_seconds() / (24 * 3600)
    now_ts = pd.Timestamp.now()
    elapsed_days = (now_ts - combined_df[eff_col]).dt.total_seconds() / (24 * 3600)
    remaining_days = (combined_df[close_col] - now_ts).dt.total_seconds() / (24 * 3600)

    combined_df["proj_total_duration_days"] = total_proj_days.clip(lower=1.0).astype("float32")
    combined_df["proj_elapsed_days"] = elapsed_days.astype("float32")
    combined_df["proj_remaining_days"] = remaining_days.astype("float32")
    combined_df["proj_timeline_completion_ratio"] = (
        elapsed_days / total_proj_days.clip(lower=1.0)
    ).clip(lower=0.0, upper=2.0).astype("float32")
    combined_df["proj_remaining_timeline_ratio"] = (
        remaining_days / total_proj_days.clip(lower=1.0)
    ).clip(lower=-1.0, upper=2.0).astype("float32")
    combined_df["proj_is_expired"] = (
        combined_df[close_col] < now_ts
    ).astype("float32")

if orig_closing_cols and revised_closing_cols:
    orig_c = orig_closing_cols[0]
    rev_c = revised_closing_cols[0]
    extension_days = (combined_df[rev_c] - combined_df[orig_c]).dt.total_seconds() / (24 * 3600)
    combined_df["grant_extension_days"] = extension_days.fillna(0.0).astype("float32")
    combined_df["grant_is_extended"] = (extension_days > 0).astype("float32")
    if "proj_total_duration_days" in combined_df.columns:
        combined_df["grant_extension_ratio"] = (
            extension_days / combined_df["proj_total_duration_days"].clip(lower=1.0)
        ).fillna(0.0).astype("float32")

combined_df.drop(columns=parsed_date_cols, inplace=True)

# ---------------------------------------------------------
# 3. Numeric and Commitment Amount Feature Engineering
# ---------------------------------------------------------
numeric_cols = []
for col in combined_df.columns:
    if col == "__is_train__":
        continue
    if pd.api.types.is_numeric_dtype(combined_df[col]):
        numeric_cols.append(col)

commitment_cols = [
    c
    for c in numeric_cols
    if any(
        k in c.lower()
        for k in ["commitment", "amount", "usd", "grant", "fund", "budget"]
    )
]

for col in commitment_cols:
    combined_df[f"{col}_log1p"] = np.log1p(
        np.maximum(0, combined_df[col].fillna(0))
    ).astype("float32")
    combined_df[f"{col}_is_zero"] = (combined_df[col] == 0).astype("float32")
    combined_df[f"{col}_is_negative"] = (combined_df[col] < 0).astype("float32")

for i in range(len(commitment_cols)):
    for j in range(i + 1, len(commitment_cols)):
        c1 = commitment_cols[i]
        c2 = commitment_cols[j]
        combined_df[f"num_diff_{c1}_minus_{c2}"] = (
            combined_df[c1] - combined_df[c2]
        ).astype("float32")
        combined_df[f"num_ratio_{c1}_to_{c2}"] = (
            combined_df[c1] / (combined_df[c2].abs() + 1.0)
        ).astype("float32")

if "proj_total_duration_days" in combined_df.columns:
    for col in commitment_cols:
        combined_df[f"{col}_burn_rate_total_days"] = (
            combined_df[col] / combined_df["proj_total_duration_days"].clip(lower=1.0)
        ).astype("float32")
        combined_df[f"{col}_burn_rate_monthly"] = (
            combined_df[col] / (combined_df["proj_total_duration_days"].clip(lower=1.0) / 30.4375)
        ).astype("float32")
        if "proj_elapsed_days" in combined_df.columns:
            combined_df[f"{col}_burn_rate_elapsed_days"] = (
                combined_df[col] / combined_df["proj_elapsed_days"].clip(lower=1.0)
            ).astype("float32")
            combined_df[f"{col}_expected_to_date"] = (
                combined_df[f"{col}_burn_rate_total_days"] * combined_df["proj_elapsed_days"].clip(lower=0.0)
            ).astype("float32")
            if "proj_remaining_days" in combined_df.columns:
                combined_df[f"{col}_remaining_velocity"] = (
                    combined_df[col] / (combined_df["proj_remaining_days"].clip(lower=1.0) + 1.0)
                ).astype("float32")

# ---------------------------------------------------------
# 4. Categorical, Text SVD, and Hierarchical Group Aggregates
# ---------------------------------------------------------
raw_cat_cols = []
for col in combined_df.columns:
    if col == "__is_train__":
        continue
    if not pd.api.types.is_numeric_dtype(combined_df[col]):
        raw_cat_cols.append(col)

is_train_mask = (combined_df["__is_train__"] == 1).values

# Pairwise categorical interactions
cat_interaction_cols = []
if len(raw_cat_cols) >= 2:
    for i in range(len(raw_cat_cols)):
        for j in range(i + 1, min(len(raw_cat_cols), i + 4)):
            col_a = raw_cat_cols[i]
            col_b = raw_cat_cols[j]
            inter_col = f"inter_{col_a}_{col_b}"
            combined_df[inter_col] = (
                combined_df[col_a].astype(str).fillna("MISSING")
                + "__"
                + combined_df[col_b].astype(str).fillna("MISSING")
            )
            cat_interaction_cols.append(inter_col)

cat_cols = raw_cat_cols + cat_interaction_cols

# Text SVD per individual column (fit on train only)
for col in raw_cat_cols:
    str_series = combined_df[col].fillna("MISSING").astype(str)
    train_str = str_series[is_train_mask]
    n_unique = train_str.nunique()
    if n_unique > 5:
        n_comp = min(4, n_unique - 1)
        try:
            tfidf_col = TfidfVectorizer(max_features=1000, stop_words="english", ngram_range=(1, 2))
            mat_train = tfidf_col.fit_transform(train_str)
            if mat_train.shape[1] >= n_comp:
                svd_col = TruncatedSVD(n_components=n_comp, random_state=42)
                svd_col.fit(mat_train)
                mat_all = tfidf_col.transform(str_series)
                svd_feats_col = svd_col.transform(mat_all)
                for idx in range(n_comp):
                    combined_df[f"svd_{col}_{idx}"] = svd_feats_col[:, idx].astype("float32")
        except Exception:
            pass

if raw_cat_cols:
    text_corpus = combined_df[raw_cat_cols].fillna("MISSING").astype(str).agg(" ".join, axis=1)
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    tfidf.fit(text_corpus[is_train_mask])
    svd = TruncatedSVD(n_components=8, random_state=42)
    svd.fit(tfidf.transform(text_corpus[is_train_mask]))
    svd_feats = svd.transform(tfidf.transform(text_corpus))
    for svd_idx in range(8):
        combined_df[f"tfidf_global_svd_{svd_idx}"] = svd_feats[:, svd_idx].astype("float32")

for col in cat_cols:
    str_series = combined_df[col].astype(str).fillna("MISSING")
    combined_df[f"{col}_char_len"] = str_series.str.len().astype("float32")
    combined_df[f"{col}_word_count"] = (
        str_series.str.split().str.len().astype("float32")
    )
    freq_map = str_series[is_train_mask].value_counts(normalize=True).to_dict()
    combined_df[f"{col}_freq"] = str_series.map(freq_map).fillna(0.0).astype("float32")
    combined_df[col] = combined_df[col].astype("category").cat.codes.astype("int32")

group_candidate_cols = [c for c in cat_cols if 1 < combined_df.loc[is_train_mask, c].nunique() < 500]
target_agg_metrics = (
    commitment_cols[:5] if len(commitment_cols) >= 5 else commitment_cols
)
if "proj_total_duration_days" in combined_df.columns:
    target_agg_metrics.append("proj_total_duration_days")

for group_col in group_candidate_cols[:8]:
    for val_col in target_agg_metrics:
        train_subset = combined_df[is_train_mask]
        grouped = train_subset.groupby(group_col)[val_col]
        mean_map = grouped.mean().to_dict()
        std_map = grouped.std().fillna(0.0).to_dict()
        min_map = grouped.min().to_dict()
        max_map = grouped.max().to_dict()
        median_map = grouped.median().to_dict()
        skew_map = grouped.apply(lambda s: s.skew() if len(s) > 2 else 0.0).fillna(0.0).to_dict()

        agg_mean = combined_df[group_col].map(mean_map).fillna(train_subset[val_col].mean()).astype("float32")
        agg_std = combined_df[group_col].map(std_map).fillna(0.0).astype("float32")
        agg_min = combined_df[group_col].map(min_map).fillna(train_subset[val_col].min()).astype("float32")
        agg_max = combined_df[group_col].map(max_map).fillna(train_subset[val_col].max()).astype("float32")
        agg_median = combined_df[group_col].map(median_map).fillna(train_subset[val_col].median()).astype("float32")
        agg_skew = combined_df[group_col].map(skew_map).fillna(0.0).astype("float32")

        combined_df[f"agg_mean_{val_col}_by_{group_col}"] = agg_mean
        combined_df[f"agg_std_{val_col}_by_{group_col}"] = agg_std
        combined_df[f"agg_min_{val_col}_by_{group_col}"] = agg_min
        combined_df[f"agg_max_{val_col}_by_{group_col}"] = agg_max
        combined_df[f"agg_median_{val_col}_by_{group_col}"] = agg_median
        combined_df[f"agg_skew_{val_col}_by_{group_col}"] = agg_skew

        # Relative Group Z-Score and Difference metrics
        combined_df[f"zscore_{val_col}_by_{group_col}"] = (
            (combined_df[val_col] - agg_mean) / (agg_std + 1e-5)
        ).astype("float32")
        combined_df[f"diff_from_group_mean_{val_col}_by_{group_col}"] = (
            combined_df[val_col] - agg_mean
        ).astype("float32")
        combined_df[f"ratio_to_group_mean_{val_col}_by_{group_col}"] = (
            combined_df[val_col] / (agg_mean.abs() + 1e-5)
        ).astype("float32")

# ---------------------------------------------------------
# 5. Out-of-fold Target Encoding & Clean Train/Test Splits
# ---------------------------------------------------------
is_train_mask = (combined_df["__is_train__"] == 1).values
train_df = (
    combined_df[is_train_mask].drop(columns=["__is_train__"]).reset_index(drop=True)
)
test_df = (
    combined_df[~is_train_mask].drop(columns=["__is_train__"]).reset_index(drop=True)
)

train_df[target_col] = y_train.values

n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
train_df["fold"] = -1

for fold, (_, val_idx) in enumerate(kf.split(train_df)):
    train_df.loc[val_idx, "fold"] = fold

high_impact_cats = group_candidate_cols[:6]
global_target_mean = float(train_df[target_col].mean())
global_target_std = float(train_df[target_col].std())

for cat_col in high_impact_cats:
    te_mean_col = f"oof_target_enc_mean_{cat_col}"
    te_std_col = f"oof_target_enc_std_{cat_col}"
    train_df[te_mean_col] = np.nan
    train_df[te_std_col] = np.nan
    test_df[te_mean_col] = 0.0
    test_df[te_std_col] = 0.0

    smoothing_weight = 10.0
    for fold in range(n_splits):
        f_tr_idx = train_df["fold"] != fold
        f_val_idx = train_df["fold"] == fold

        stats = (
            train_df.loc[f_tr_idx]
            .groupby(cat_col)[target_col]
            .agg(["mean", "std", "count"])
        )
        stats["std"] = stats["std"].fillna(0.0)

        smooth_mean = (
            stats["count"] * stats["mean"] + smoothing_weight * global_target_mean
        ) / (stats["count"] + smoothing_weight)
        smooth_std = (
            stats["count"] * stats["std"] + smoothing_weight * global_target_std
        ) / (stats["count"] + smoothing_weight)

        train_df.loc[f_val_idx, te_mean_col] = (
            train_df.loc[f_val_idx, cat_col]
            .map(smooth_mean)
            .fillna(global_target_mean)
            .astype("float32")
        )
        train_df.loc[f_val_idx, te_std_col] = (
            train_df.loc[f_val_idx, cat_col]
            .map(smooth_std)
            .fillna(global_target_std)
            .astype("float32")
        )

    full_stats = train_df.groupby(cat_col)[target_col].agg(["mean", "std", "count"])
    full_stats["std"] = full_stats["std"].fillna(0.0)
    full_smooth_mean = (
        full_stats["count"] * full_stats["mean"] + smoothing_weight * global_target_mean
    ) / (full_stats["count"] + smoothing_weight)
    full_smooth_std = (
        full_stats["count"] * full_stats["std"] + smoothing_weight * global_target_std
    ) / (full_stats["count"] + smoothing_weight)

    test_df[te_mean_col] = (
        test_df[cat_col]
        .map(full_smooth_mean)
        .fillna(global_target_mean)
        .astype("float32")
    )
    test_df[te_std_col] = (
        test_df[cat_col]
        .map(full_smooth_std)
        .fillna(global_target_std)
        .astype("float32")
    )

all_feature_cols = [c for c in train_df.columns if c not in [target_col, "fold"]]

for col in all_feature_cols:
    train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan)
    test_df[col] = test_df[col].replace([np.inf, -np.inf], np.nan)

    if train_df[col].isnull().any() or test_df[col].isnull().any():
        med_val = train_df[col].median()
        if pd.isna(med_val):
            med_val = 0.0
        train_df[col] = train_df[col].fillna(med_val)
        test_df[col] = test_df[col].fillna(med_val)

train_df.to_parquet(os.path.join(WORKING_DIR, "processed_train.parquet"), index=False)
test_df.to_parquet(os.path.join(WORKING_DIR, "processed_test.parquet"), index=False)


# ---------------------------------------------------------
# 6. Model Training & Validation (10-Fold CV Multi-Architecture Ensemble)
# ---------------------------------------------------------
X = train_df[all_feature_cols]
y = train_df[target_col].values
X_test = test_df[all_feature_cols]
folds = train_df["fold"].values

oof_preds_lgb = np.zeros(len(train_df), dtype=np.float64)
oof_preds_dart = np.zeros(len(train_df), dtype=np.float64)
oof_preds_xgb = np.zeros(len(train_df), dtype=np.float64)
oof_preds_cat = np.zeros(len(train_df), dtype=np.float64)
oof_preds_et = np.zeros(len(train_df), dtype=np.float64)

test_preds_lgb = np.zeros(len(test_df), dtype=np.float64)
test_preds_dart = np.zeros(len(test_df), dtype=np.float64)
test_preds_xgb = np.zeros(len(test_df), dtype=np.float64)
test_preds_cat = np.zeros(len(test_df), dtype=np.float64)
test_preds_et = np.zeros(len(test_df), dtype=np.float64)

min_target_val = float(np.min(y))

print(f"Starting {n_splits}-Fold Cross-Validation training...")

for fold in range(n_splits):
    idx_tr = np.where(folds != fold)[0]
    idx_val = np.where(folds == fold)[0]

    X_tr, y_tr = X.iloc[idx_tr], y[idx_tr]
    X_val, y_val = X.iloc[idx_val], y[idx_val]

    # --- 1. LightGBM GBDT ---
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 47,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "n_estimators": 3000,
        "random_state": 42 + fold,
        "n_jobs": -1,
        "verbose": -1,
    }
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=80, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_pred_lgb = model_lgb.predict(X_val)
    oof_preds_lgb[idx_val] = val_pred_lgb
    test_preds_lgb += model_lgb.predict(X_test) / n_splits

    # --- 2. LightGBM DART ---
    dart_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "dart",
        "learning_rate": 0.04,
        "num_leaves": 31,
        "drop_rate": 0.05,
        "skip_drop": 0.5,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "n_estimators": 750,
        "random_state": 142 + fold,
        "n_jobs": -1,
        "verbose": -1,
    }
    model_dart = lgb.LGBMRegressor(**dart_params)
    model_dart.fit(X_tr, y_tr)

    val_pred_dart = model_dart.predict(X_val)
    oof_preds_dart[idx_val] = val_pred_dart
    test_preds_dart += model_dart.predict(X_test) / n_splits

    # --- 3. XGBoost Hist ---
    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 3000,
        "random_state": 42 + fold,
        "n_jobs": -1,
        "tree_method": "hist",
        "early_stopping_rounds": 80,
    }
    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    val_pred_xgb = model_xgb.predict(X_val)
    oof_preds_xgb[idx_val] = val_pred_xgb
    test_preds_xgb += model_xgb.predict(X_test) / n_splits

    # --- 4. CatBoost Regressor ---
    cb_params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "iterations": 3000,
        "random_seed": 42 + fold,
        "verbose": False,
        "early_stopping_rounds": 80,
    }
    model_cat = cb.CatBoostRegressor(**cb_params)
    model_cat.fit(
        X_tr,
        y_tr,
        eval_set=(X_val, y_val),
        verbose=False,
    )

    val_pred_cat = model_cat.predict(X_val)
    oof_preds_cat[idx_val] = val_pred_cat
    test_preds_cat += model_cat.predict(X_test) / n_splits

    # --- 5. ExtraTrees Regressor ---
    model_et = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=22,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features=0.75,
        n_jobs=-1,
        random_state=42 + fold,
    )
    model_et.fit(X_tr, y_tr)
    val_pred_et = model_et.predict(X_val)
    oof_preds_et[idx_val] = val_pred_et
    test_preds_et += model_et.predict(X_test) / n_splits

    fold_rmse_lgb = root_mean_squared_error(y_val, val_pred_lgb)
    fold_rmse_dart = root_mean_squared_error(y_val, val_pred_dart)
    fold_rmse_xgb = root_mean_squared_error(y_val, val_pred_xgb)
    fold_rmse_cat = root_mean_squared_error(y_val, val_pred_cat)
    fold_rmse_et = root_mean_squared_error(y_val, val_pred_et)
    print(
        f"Fold {fold + 1}/{n_splits} - LGB: {fold_rmse_lgb:.4f} | DART: {fold_rmse_dart:.4f} | XGB: {fold_rmse_xgb:.4f} | CAT: {fold_rmse_cat:.4f} | ET: {fold_rmse_et:.4f}"
    )

# ---------------------------------------------------------
# 7. Multi-Model SLSQP Optimal Blending & Submission Generation
# ---------------------------------------------------------
rmse_lgb_total = root_mean_squared_error(y, oof_preds_lgb)
rmse_dart_total = root_mean_squared_error(y, oof_preds_dart)
rmse_xgb_total = root_mean_squared_error(y, oof_preds_xgb)
rmse_cat_total = root_mean_squared_error(y, oof_preds_cat)
rmse_et_total = root_mean_squared_error(y, oof_preds_et)
print(
    f"Overall OOF - LightGBM: {rmse_lgb_total:.4f} | LGBM-DART: {rmse_dart_total:.4f} | XGBoost: {rmse_xgb_total:.4f} | CatBoost: {rmse_cat_total:.4f} | ExtraTrees: {rmse_et_total:.4f}"
)


def blend_objective(weights):
    w1, w2, w3, w4, w5 = weights
    pred = (
        w1 * oof_preds_lgb
        + w2 * oof_preds_dart
        + w3 * oof_preds_xgb
        + w4 * oof_preds_cat
        + w5 * oof_preds_et
    )
    if min_target_val >= 0:
        pred = np.clip(pred, a_min=min_target_val, a_max=None)
    return root_mean_squared_error(y, pred)


init_weights = [0.25, 0.20, 0.20, 0.25, 0.10]
opt_res = minimize(
    blend_objective,
    init_weights,
    method="SLSQP",
    bounds=[(0, 1)] * 5,
    constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0},
)
w_lgb, w_dart, w_xgb, w_cat, w_et = (
    opt_res.x if opt_res.success else init_weights
)
print(
    f"Optimal Ensemble Weights -> LGB: {w_lgb:.3f}, DART: {w_dart:.3f}, XGB: {w_xgb:.3f}, CAT: {w_cat:.3f}, ET: {w_et:.3f}"
)

oof_final = (
    w_lgb * oof_preds_lgb
    + w_dart * oof_preds_dart
    + w_xgb * oof_preds_xgb
    + w_cat * oof_preds_cat
    + w_et * oof_preds_et
)
test_final = (
    w_lgb * test_preds_lgb
    + w_dart * test_preds_dart
    + w_xgb * test_preds_xgb
    + w_cat * test_preds_cat
    + w_et * test_preds_et
)

if min_target_val >= 0:
    oof_final = np.clip(oof_final, a_min=min_target_val, a_max=None)
    test_final = np.clip(test_final, a_min=min_target_val, a_max=None)

final_score = root_mean_squared_error(y, oof_final)

if "id" in df_test.columns:
    submission_id = df_test["id"].values
elif "ID" in df_test.columns:
    submission_id = df_test["ID"].values
else:
    submission_id = np.arange(len(test_df))

submission_df = pd.DataFrame({"id": submission_id, target_col: test_final})
submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df.to_csv(submission_path, index=False)

print(f"Submission saved to '{submission_path}' with shape {submission_df.shape}.")
print(f"Final Validation Score: {final_score}")