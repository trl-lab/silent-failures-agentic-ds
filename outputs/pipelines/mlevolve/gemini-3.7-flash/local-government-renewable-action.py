import json
import os
import re
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import catboost as cb
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. Setup & Directory Configuration
# ==============================================================================
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==============================================================================
# 2. Data Loading & Stratified Split Setup
# ==============================================================================
train_path = os.path.join(INPUT_DIR, "train.parquet")
test_path = os.path.join(INPUT_DIR, "test.parquet")

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Size of Purchase (MW)"
if target_col not in train_df.columns:
    matched = [
        c
        for c in train_df.columns
        if "size" in c.lower() or "purchase" in c.lower() or "mw" in c.lower()
    ]
    target_col = matched[0] if matched else train_df.columns[-1]

y_train = train_df[target_col].copy()
X_train_raw = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.copy()

id_col = None
for col in ["id", "ID", "Id", "index"]:
    if col in X_test_raw.columns:
        id_col = col
        break

if id_col is not None and id_col in X_train_raw.columns:
    X_train_raw = X_train_raw.drop(columns=[id_col])
if id_col is not None:
    test_ids = X_test_raw[id_col].values
    X_test_raw = X_test_raw.drop(columns=[id_col])
else:
    test_ids = np.arange(len(X_test_raw))

# Stratified 5-Fold Split for Continuous Target
n_splits = 5
n_bins = 10
target_binned = pd.qcut(y_train.rank(method="first"), q=n_bins, labels=False)
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_indices = np.zeros(len(train_df), dtype=int)
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_raw, target_binned)):
    fold_indices[val_idx] = fold


# ==============================================================================
# 3. Feature Engineering Pipeline
# ==============================================================================
def extract_date_and_numeric_features(df):
    """Extract numeric values, years, and dates from object/numeric columns."""
    df_out = pd.DataFrame(index=df.index)

    for col in df.columns:
        if df[col].dtype == "object":
            year_series = (
                df[col].astype(str).str.extract(r"(\b(?:19|20)\d{2}\b)", expand=False)
            )
            if year_series.notna().sum() > 0.1 * len(df):
                df_out[f"{col}_extracted_year"] = pd.to_numeric(
                    year_series, errors="coerce"
                )

            num_series = (
                df[col].astype(str).str.extract(r"(\d+(?:\.\d+)?)", expand=False)
            )
            if num_series.notna().sum() > 0.1 * len(df):
                df_out[f"{col}_extracted_num"] = pd.to_numeric(
                    num_series, errors="coerce"
                )

            df_out[f"{col}_char_len"] = df[col].astype(str).str.len()
            df_out[f"{col}_word_count"] = (
                df[col].astype(str).apply(lambda x: len(x.split()))
            )

            keywords = [
                "solar",
                "wind",
                "storage",
                "battery",
                "hydro",
                "ppa",
                "vppa",
                "rooftop",
                "offsite",
                "onsite",
                "community",
                "green",
                "rec",
                "utility",
            ]
            for kw in keywords:
                df_out[f"{col}_has_{kw}"] = (
                    df[col].astype(str).str.lower().str.contains(kw).astype(int)
                )
        elif np.issubdtype(df[col].dtype, np.number):
            df_out[col] = df[col]
            if (df[col] >= 0).all() and (df[col].skew() > 1.5):
                df_out[f"{col}_log1p"] = np.log1p(df[col])

    return df_out


X_train_fe = extract_date_and_numeric_features(X_train_raw)
X_test_fe = extract_date_and_numeric_features(X_test_raw)

common_cols = [c for c in X_train_fe.columns if c in X_test_fe.columns]
X_train_fe = X_train_fe[common_cols].copy()
X_test_fe = X_test_fe[common_cols].copy()

categorical_cols = [
    c
    for c in X_train_raw.columns
    if X_train_raw[c].dtype == "object" or X_train_raw[c].nunique() < 50
]

pair_count = 0
for i in range(len(categorical_cols)):
    for j in range(i + 1, len(categorical_cols)):
        if pair_count >= 15:
            break
        c1, c2 = categorical_cols[i], categorical_cols[j]
        inter_name = f"{c1}__x__{c2}"
        X_train_fe[inter_name] = (
            X_train_raw[c1].astype(str) + "_" + X_train_raw[c2].astype(str)
        )
        X_test_fe[inter_name] = (
            X_test_raw[c1].astype(str) + "_" + X_test_raw[c2].astype(str)
        )
        categorical_cols.append(inter_name)
        pair_count += 1

# Frequency and Label Encodings
for col in categorical_cols:
    combined_counts = pd.concat(
        [
            X_train_raw[col] if col in X_train_raw else X_train_fe[col],
            X_test_raw[col] if col in X_test_raw else X_test_fe[col],
        ]
    ).value_counts(dropna=False)

    col_src_tr = X_train_raw[col] if col in X_train_raw else X_train_fe[col]
    col_src_te = X_test_raw[col] if col in X_test_raw else X_test_fe[col]

    X_train_fe[f"{col}_freq"] = col_src_tr.map(combined_counts).fillna(0).astype(float)
    X_test_fe[f"{col}_freq"] = col_src_te.map(combined_counts).fillna(0).astype(float)

    le = LabelEncoder()
    str_tr = col_src_tr.fillna("__NA__").astype(str)
    str_te = col_src_te.fillna("__NA__").astype(str)
    all_vals = list(set(str_tr.unique()).union(set(str_te.unique())))
    le.fit(all_vals)
    X_train_fe[f"{col}_le"] = le.transform(str_tr)
    X_test_fe[f"{col}_le"] = le.transform(str_te)

# Leak-Free Out-Of-Fold (OOF) Target Encoding
global_mean = y_train.mean()
smoothing = 10.0

for col in categorical_cols:
    target_enc_col = f"{col}_oof_target_mean"
    X_train_fe[target_enc_col] = np.nan

    col_src_tr = X_train_raw[col] if col in X_train_raw else X_train_fe[col]
    col_src_te = X_test_raw[col] if col in X_test_raw else X_test_fe[col]

    for fold in range(n_splits):
        val_mask = fold_indices == fold
        trn_mask = ~val_mask

        trn_cat = col_src_tr[trn_mask].astype(str)
        trn_y = y_train[trn_mask]

        stats = trn_y.groupby(trn_cat).agg(["count", "mean"])
        smoothed_means = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )

        val_cat = col_src_tr[val_mask].astype(str)
        X_train_fe.loc[val_mask, target_enc_col] = val_cat.map(smoothed_means).fillna(
            global_mean
        )

    full_stats = y_train.groupby(col_src_tr.astype(str)).agg(["count", "mean"])
    full_smoothed = (
        full_stats["count"] * full_stats["mean"] + smoothing * global_mean
    ) / (full_stats["count"] + smoothing)
    X_test_fe[target_enc_col] = (
        col_src_te.astype(str).map(full_smoothed).fillna(global_mean)
    )

# Group-by Aggregate Statistics
num_cols_for_agg = [
    c
    for c in X_train_fe.columns
    if np.issubdtype(X_train_fe[c].dtype, np.number)
    and "le" not in c
    and "target" not in c
    and "has_" not in c
]
key_cats = [c for c in categorical_cols[:6] if c in X_train_raw]

for cat_col in key_cats:
    for num_col in num_cols_for_agg[:4]:
        group_means = X_train_fe.groupby(X_train_raw[cat_col].astype(str))[
            num_col
        ].mean()
        group_stds = (
            X_train_fe.groupby(X_train_raw[cat_col].astype(str))[num_col]
            .std()
            .fillna(0)
        )

        X_train_fe[f"{num_col}_mean_by_{cat_col}"] = (
            X_train_raw[cat_col]
            .astype(str)
            .map(group_means)
            .fillna(X_train_fe[num_col].mean())
        )
        X_test_fe[f"{num_col}_mean_by_{cat_col}"] = (
            X_test_raw[cat_col]
            .astype(str)
            .map(group_means)
            .fillna(X_train_fe[num_col].mean())
        )

        X_train_fe[f"{num_col}_diff_from_mean_{cat_col}"] = (
            X_train_fe[num_col] - X_train_fe[f"{num_col}_mean_by_{cat_col}"]
        )
        X_test_fe[f"{num_col}_diff_from_mean_{cat_col}"] = (
            X_test_fe[num_col] - X_test_fe[f"{num_col}_mean_by_{cat_col}"]
        )

# Final Clean-up and Imputation
X_train_final = X_train_fe.select_dtypes(include=[np.number]).copy()
X_test_final = X_test_fe.select_dtypes(include=[np.number]).copy()

for col in X_train_final.columns:
    median_val = X_train_final[col].median()
    if pd.isna(median_val):
        median_val = 0.0
    X_train_final[col] = X_train_final[col].fillna(median_val)
    if col in X_test_final.columns:
        X_test_final[col] = X_test_final[col].fillna(median_val)

X_train_final[target_col] = y_train.values
X_train_final["fold"] = fold_indices

train_save_path = os.path.join(WORKING_DIR, "train_processed.parquet")
test_save_path = os.path.join(WORKING_DIR, "test_processed.parquet")
metadata_save_path = os.path.join(WORKING_DIR, "feature_metadata.json")

X_train_final.to_parquet(train_save_path, index=False)
X_test_final.to_parquet(test_save_path, index=False)

feature_cols = [c for c in X_train_final.columns if c not in [target_col, "fold"]]
metadata = {
    "target_col": target_col,
    "feature_count": len(feature_cols),
    "feature_names": feature_cols,
    "n_splits": n_splits,
    "train_rows": len(X_train_final),
    "test_rows": len(X_test_final),
}

with open(metadata_save_path, "w") as f:
    json.dump(metadata, f, indent=2)

# ==============================================================================
# 4. Multi-Model Training and Validation Pipeline
# ==============================================================================
X = X_train_final[feature_cols].values
y = X_train_final[target_col].values
X_test = X_test_final[feature_cols].values

n_train = len(X_train_final)
n_test = len(X_test_final)

oof_lgb = np.zeros(n_train)
oof_xgb = np.zeros(n_train)
oof_cat = np.zeros(n_train)

test_preds_lgb = np.zeros(n_test)
test_preds_xgb = np.zeros(n_test)
test_preds_cat = np.zeros(n_test)

for fold in range(n_splits):
    val_idx = np.where(fold_indices == fold)[0]
    trn_idx = np.where(fold_indices != fold)[0]

    X_trn, y_trn = X[trn_idx], y[trn_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # --- LightGBM ---
    lgb_model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_trn,
        y_trn,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    val_pred_lgb = lgb_model.predict(X_val)
    oof_lgb[val_idx] = val_pred_lgb
    test_preds_lgb += lgb_model.predict(X_test) / n_splits

    # --- XGBoost ---
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + fold,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )
    xgb_model.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], verbose=False)
    val_pred_xgb = xgb_model.predict(X_val)
    oof_xgb[val_idx] = val_pred_xgb
    test_preds_xgb += xgb_model.predict(X_test) / n_splits

    # --- CatBoost ---
    cat_model = cb.CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42 + fold,
        verbose=False,
        thread_count=-1,
        early_stopping_rounds=50,
    )
    cat_model.fit(X_trn, y_trn, eval_set=(X_val, y_val), verbose=False)
    val_pred_cat = cat_model.predict(X_val)
    oof_cat[val_idx] = val_pred_cat
    test_preds_cat += cat_model.predict(X_test) / n_splits


# ==============================================================================
# 5. Optimal Ensemble Blending & Evaluation
# ==============================================================================
def ensemble_loss(weights):
    w1, w2, w3 = weights
    pred = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
    pred = np.clip(pred, 0, None)
    return root_mean_squared_error(y, pred)


initial_weights = [1.0 / 3, 1.0 / 3, 1.0 / 3]
bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    ensemble_loss,
    initial_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
weights = opt_res.x if opt_res.success else np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

oof_ensemble = weights[0] * oof_lgb + weights[1] * oof_xgb + weights[2] * oof_cat
oof_ensemble = np.clip(oof_ensemble, 0, None)
final_score = root_mean_squared_error(y, oof_ensemble)

# ==============================================================================
# 6. Test Inference & Submission Generation
# ==============================================================================
test_predictions = (
    weights[0] * test_preds_lgb
    + weights[1] * test_preds_xgb
    + weights[2] * test_preds_cat
)
test_predictions = np.clip(test_predictions, 0, None)

submission_df = pd.DataFrame(
    {"id": np.arange(n_test), "Size of Purchase (MW)": test_predictions}
)
submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df.to_csv(submission_path, index=False)

print(f"Final Validation Score: {final_score}")
