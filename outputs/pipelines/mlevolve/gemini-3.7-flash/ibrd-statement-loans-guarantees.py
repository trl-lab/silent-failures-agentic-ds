import gc
import math
import os
import random
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
import xgboost as xgb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 1. Global Configuration and Reproducibility
# =============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

INPUT_DIR = "./input"
SUBMISSION_DIR = "./submission"
WORKING_DIR = "./working"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(INPUT_DIR, "train.parquet")
TEST_PATH = os.path.join(INPUT_DIR, "test.parquet")
TARGET_COL = "Original Principal Amount (US$)"
N_SPLITS = 5

# =============================================================================
# 2. Data Loading & Feature Engineering
# =============================================================================
print("Loading raw data...")
train_raw = pd.read_parquet(TRAIN_PATH)
test_raw = pd.read_parquet(TEST_PATH)

if TARGET_COL not in train_raw.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in training data.")

y_train_arr = train_raw[TARGET_COL].astype(np.float64).values
train_df = train_raw.drop(columns=[TARGET_COL]).copy()
test_df = test_raw.copy()

train_len = len(train_df)
test_len = len(test_df)


# Date & Administrative Timeline Processing
def detect_and_parse_dates(df):
    date_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if "date" in col_lower or "period" in col_lower:
            date_cols.append(col)
    parsed_df = pd.DataFrame(index=df.index)
    for col in date_cols:
        parsed_df[col] = pd.to_datetime(df[col], errors="coerce")
    return parsed_df, date_cols


train_dates, date_columns = detect_and_parse_dates(train_df)
test_dates, _ = detect_and_parse_dates(test_df)


def engineer_date_features(df_dates, cols):
    features = pd.DataFrame(index=df_dates.index)
    for col in cols:
        s = df_dates[col]
        prefix = col.replace(" ", "_").replace("(", "").replace(")", "")
        features[f"{prefix}_year"] = s.dt.year.astype(np.float32)
        features[f"{prefix}_month"] = s.dt.month.astype(np.float32)
        features[f"{prefix}_quarter"] = s.dt.quarter.astype(np.float32)
        features[f"{prefix}_dayofweek"] = s.dt.dayofweek.astype(np.float32)
        features[f"{prefix}_is_month_end"] = s.dt.is_month_end.astype(np.float32)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1, col2 = cols[i], cols[j]
            p1 = col1.replace(" ", "_").replace("(", "").replace(")", "")
            p2 = col2.replace(" ", "_").replace("(", "").replace(")", "")
            diff_days = (df_dates[col2] - df_dates[col1]).dt.total_seconds() / (
                24 * 3600
            )
            features[f"diff_days_{p2}_minus_{p1}"] = diff_days.astype(np.float32)
    return features


date_feat_train = engineer_date_features(train_dates, date_columns)
date_feat_test = engineer_date_features(test_dates, date_columns)

# Column Categorization
non_date_cols = [c for c in train_df.columns if c not in date_columns]
cat_cols = []
num_cols = []
text_cols = []

for c in non_date_cols:
    is_cat = (
        train_df[c].dtype == "object"
        or isinstance(train_df[c].dtype, pd.CategoricalDtype)
    )
    if is_cat:
        sample_vals = train_df[c].dropna().astype(str)
        if (
            len(sample_vals) > 0
            and sample_vals.map(len).mean() > 40
            and sample_vals.nunique() > 200
        ):
            text_cols.append(c)
        else:
            cat_cols.append(c)
    elif pd.api.types.is_numeric_dtype(train_df[c]):
        num_cols.append(c)
    else:
        cat_cols.append(c)


# Numerical Engineering, Financial Interactions, and Hierarchical Group Aggregations
def engineer_numeric_features(df_train, df_test, cols):
    feat_train = pd.DataFrame(index=df_train.index)
    feat_test = pd.DataFrame(index=df_test.index)

    clean_cols = []
    for col in cols:
        val_tr = pd.to_numeric(df_train[col], errors="coerce").astype(np.float32)
        val_te = pd.to_numeric(df_test[col], errors="coerce").astype(np.float32)
        clean_name = (
            col.replace(" ", "_").replace("(", "").replace(")", "").replace("$", "USD")
        )
        clean_cols.append(clean_name)
        feat_train[clean_name] = val_tr
        feat_test[clean_name] = val_te
        feat_train[f"{clean_name}_is_missing"] = val_tr.isna().astype(np.float32)
        feat_test[f"{clean_name}_is_missing"] = val_te.isna().astype(np.float32)
        feat_train[f"{clean_name}_log1p"] = np.log1p(np.clip(val_tr.fillna(0), 0, None))
        feat_test[f"{clean_name}_log1p"] = np.log1p(np.clip(val_te.fillna(0), 0, None))

    # Financial pairwise interaction features
    for i in range(len(clean_cols)):
        for j in range(i + 1, len(clean_cols)):
            c1, c2 = clean_cols[i], clean_cols[j]
            v1_tr, v2_tr = feat_train[c1], feat_train[c2]
            v1_te, v2_te = feat_test[c1], feat_test[c2]

            feat_train[f"{c1}_plus_{c2}"] = v1_tr + v2_tr
            feat_test[f"{c1}_plus_{c2}"] = v1_te + v2_te

            feat_train[f"{c1}_minus_{c2}"] = v1_tr - v2_tr
            feat_test[f"{c1}_minus_{c2}"] = v1_te - v2_te

            feat_train[f"{c1}_ratio_{c2}"] = v1_tr / (np.abs(v2_tr) + 1.0)
            feat_test[f"{c1}_ratio_{c2}"] = v1_te / (np.abs(v2_te) + 1.0)

    # Hierarchical group-by entity aggregations across dataset
    combined_df = pd.concat([df_train, df_test], axis=0)
    for cat_c in cat_cols:
        nunique = combined_df[cat_c].nunique()
        if 1 < nunique <= 500:
            clean_cat = cat_c.replace(" ", "_").replace("(", "").replace(")", "")
            for num_c in cols:
                clean_num = (
                    num_c.replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("$", "USD")
                )
                grp = combined_df.groupby(cat_c)[num_c]
                agg_mean = grp.transform("mean").astype(np.float32)
                agg_std = grp.transform("std").fillna(0).astype(np.float32)
                agg_min = grp.transform("min").astype(np.float32)
                agg_max = grp.transform("max").astype(np.float32)
                agg_sum = grp.transform("sum").astype(np.float32)

                tr_mean = agg_mean.iloc[:len(df_train)]
                te_mean = agg_mean.iloc[len(df_train):]

                feat_train[f"grp_{clean_cat}_{clean_num}_mean"] = tr_mean
                feat_test[f"grp_{clean_cat}_{clean_num}_mean"] = te_mean
                feat_train[f"grp_{clean_cat}_{clean_num}_std"] = agg_std.iloc[:len(df_train)]
                feat_test[f"grp_{clean_cat}_{clean_num}_std"] = agg_std.iloc[len(df_train):]
                feat_train[f"grp_{clean_cat}_{clean_num}_min"] = agg_min.iloc[:len(df_train)]
                feat_test[f"grp_{clean_cat}_{clean_num}_min"] = agg_min.iloc[len(df_train):]
                feat_train[f"grp_{clean_cat}_{clean_num}_max"] = agg_max.iloc[:len(df_train)]
                feat_test[f"grp_{clean_cat}_{clean_num}_max"] = agg_max.iloc[len(df_train):]
                feat_train[f"grp_{clean_cat}_{clean_num}_sum"] = agg_sum.iloc[:len(df_train)]
                feat_test[f"grp_{clean_cat}_{clean_num}_sum"] = agg_sum.iloc[len(df_train):]

                feat_train[f"{clean_num}_diff_grp_{clean_cat}_mean"] = feat_train[clean_num] - tr_mean
                feat_test[f"{clean_num}_diff_grp_{clean_cat}_mean"] = feat_test[clean_num] - te_mean

    return feat_train, feat_test


num_feat_train, num_feat_test = engineer_numeric_features(train_df, test_df, num_cols)


# Frequency and Label Encodings
def engineer_cat_frequency(train_series, test_series):
    val_counts = train_series.astype(str).value_counts(normalize=True).to_dict()
    train_freq = train_series.astype(str).map(val_counts).fillna(0.0).astype(np.float32)
    test_freq = test_series.astype(str).map(val_counts).fillna(0.0).astype(np.float32)
    return train_freq, test_freq


freq_feat_train = pd.DataFrame(index=train_df.index)
freq_feat_test = pd.DataFrame(index=test_df.index)
label_feat_train = pd.DataFrame(index=train_df.index)
label_feat_test = pd.DataFrame(index=test_df.index)

for c in cat_cols:
    clean_name = c.replace(" ", "_").replace("(", "").replace(")", "")
    tr_freq, te_freq = engineer_cat_frequency(train_df[c], test_df[c])
    freq_feat_train[f"{clean_name}_freq"] = tr_freq
    freq_feat_test[f"{clean_name}_freq"] = te_freq

    le = LabelEncoder()
    combined = pd.concat(
        [
            train_df[c].astype(str).fillna("__MISSING__"),
            test_df[c].astype(str).fillna("__MISSING__"),
        ],
        axis=0,
    )
    le.fit(combined)
    label_feat_train[f"{clean_name}_le"] = le.transform(
        train_df[c].astype(str).fillna("__MISSING__")
    ).astype(np.int32)
    label_feat_test[f"{clean_name}_le"] = le.transform(
        test_df[c].astype(str).fillna("__MISSING__")
    ).astype(np.int32)

# Stratified Cross-Validation Splitting
log_target = np.log1p(np.clip(y_train_arr, 0, None))
strat_bins = pd.qcut(log_target, q=10, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
fold_indices = list(skf.split(train_df, strat_bins))

# Out-of-Fold Target Encoding
oof_target_enc_train = pd.DataFrame(index=train_df.index)
oof_target_enc_test = pd.DataFrame(index=test_df.index)

for c in cat_cols:
    clean_name = c.replace(" ", "_").replace("(", "").replace(")", "")
    oof_target_enc_train[f"{clean_name}_te_mean"] = 0.0
    oof_target_enc_train[f"{clean_name}_te_std"] = 0.0
    test_means_accum = np.zeros(test_len, dtype=np.float64)
    test_stds_accum = np.zeros(test_len, dtype=np.float64)

    global_mean = float(np.mean(log_target))
    global_std = float(np.std(log_target))

    for tr_idx, val_idx in fold_indices:
        tr_cat = train_df[c].iloc[tr_idx].astype(str)
        val_cat = train_df[c].iloc[val_idx].astype(str)
        tr_y = log_target[tr_idx]

        stats = (
            pd.DataFrame({"cat": tr_cat, "y": tr_y})
            .groupby("cat")["y"]
            .agg(["count", "mean", "std"])
        )
        smoothing_weight = 10.0
        smooth_mean = (
            stats["count"] * stats["mean"] + smoothing_weight * global_mean
        ) / (stats["count"] + smoothing_weight)
        smooth_std = stats["std"].fillna(global_std)

        mean_map = smooth_mean.to_dict()
        std_map = smooth_std.to_dict()

        oof_target_enc_train.loc[train_df.index[val_idx], f"{clean_name}_te_mean"] = (
            val_cat.map(mean_map).fillna(global_mean).values
        )
        oof_target_enc_train.loc[train_df.index[val_idx], f"{clean_name}_te_std"] = (
            val_cat.map(std_map).fillna(global_std).values
        )

        te_cat = test_df[c].astype(str)
        test_means_accum += te_cat.map(mean_map).fillna(global_mean).values / N_SPLITS
        test_stds_accum += te_cat.map(std_map).fillna(global_std).values / N_SPLITS

    oof_target_enc_test[f"{clean_name}_te_mean"] = test_means_accum.astype(np.float32)
    oof_target_enc_test[f"{clean_name}_te_std"] = test_stds_accum.astype(np.float32)

# Text Feature Decomposition with TF-IDF + TruncatedSVD
text_feat_train = pd.DataFrame(index=train_df.index)
text_feat_test = pd.DataFrame(index=test_df.index)

for tc in text_cols:
    clean_name = tc.replace(" ", "_").replace("(", "").replace(")", "")
    tr_text = train_df[tc].fillna("").astype(str).tolist()
    te_text = test_df[tc].fillna("").astype(str).tolist()

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    svd = TruncatedSVD(n_components=16, random_state=SEED)

    tfidf_train = tfidf.fit_transform(tr_text)
    svd_train = svd.fit_transform(tfidf_train)

    tfidf_test = tfidf.transform(te_text)
    svd_test = svd.transform(tfidf_test)

    for dim in range(16):
        text_feat_train[f"{clean_name}_svd_{dim}"] = svd_train[:, dim].astype(
            np.float32
        )
        text_feat_test[f"{clean_name}_svd_{dim}"] = svd_test[:, dim].astype(np.float32)

# Combine and Impute
X_train_full = pd.concat(
    [
        num_feat_train,
        date_feat_train,
        freq_feat_train,
        label_feat_train,
        oof_target_enc_train,
        text_feat_train,
    ],
    axis=1,
)

X_test_full = pd.concat(
    [
        num_feat_test,
        date_feat_test,
        freq_feat_test,
        label_feat_test,
        oof_target_enc_test,
        text_feat_test,
    ],
    axis=1,
)

common_cols = [c for c in X_train_full.columns if c in X_test_full.columns]
X_train_full = X_train_full[common_cols].copy()
X_test_full = X_test_full[common_cols].copy()

imputer = SimpleImputer(strategy="median")
X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train_full), columns=common_cols, index=train_df.index
)
X_test_imputed = pd.DataFrame(
    imputer.transform(X_test_full), columns=common_cols, index=test_df.index
)

n_train = len(X_train_imputed)
n_test = len(X_test_imputed)
n_features = X_train_imputed.shape[1]


# =============================================================================
# 3. Model Architectures & Parameter Configurations
# =============================================================================
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 45,
    "max_depth": 7,
    "min_child_samples": 20,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": -1,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 3.0,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.1,
    "reg_lambda": 1.5,
    "random_state": SEED,
    "n_jobs": -1,
}

cb_params = {
    "iterations": 2000,
    "learning_rate": 0.04,
    "depth": 6,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": SEED,
    "verbose": False,
    "early_stopping_rounds": 50,
}

hgb_params = {
    "loss": "squared_error",
    "learning_rate": 0.04,
    "max_iter": 600,
    "max_depth": 7,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "early_stopping": True,
    "n_iter_no_change": 30,
    "random_state": SEED,
}

# =============================================================================
# 4. Out-of-Fold Model Training & Cross Validation
# =============================================================================
oof_lgb = np.zeros(n_train, dtype=np.float64)
oof_xgb = np.zeros(n_train, dtype=np.float64)
oof_cb = np.zeros(n_train, dtype=np.float64)
oof_hgb = np.zeros(n_train, dtype=np.float64)

test_preds_lgb = np.zeros(n_test, dtype=np.float64)
test_preds_xgb = np.zeros(n_test, dtype=np.float64)
test_preds_cb = np.zeros(n_test, dtype=np.float64)
test_preds_hgb = np.zeros(n_test, dtype=np.float64)

X_test_mat = X_test_imputed.values.astype(np.float32)

print("Starting 5-Fold Training Loop across 4 Tree Ensembles...")

for fold, (tr_idx, val_idx) in enumerate(fold_indices):
    X_tr = X_train_imputed.iloc[tr_idx].values.astype(np.float32)
    y_tr = y_train_arr[tr_idx]
    X_va = X_train_imputed.iloc[val_idx].values.astype(np.float32)
    y_va = y_train_arr[val_idx]

    # --- LightGBM ---
    dtrain_lgb = lgb.Dataset(X_tr, label=y_tr)
    dval_lgb = lgb.Dataset(X_va, label=y_va, reference=dtrain_lgb)
    model_lgb = lgb.train(
        lgb_params,
        dtrain_lgb,
        num_boost_round=2000,
        valid_sets=[dval_lgb],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    val_pred_lgb = np.clip(
        model_lgb.predict(X_va, num_iteration=model_lgb.best_iteration), 0, None
    )
    oof_lgb[val_idx] = val_pred_lgb
    test_preds_lgb += (
        np.clip(
            model_lgb.predict(X_test_mat, num_iteration=model_lgb.best_iteration),
            0,
            None,
        )
        / N_SPLITS
    )

    # --- XGBoost ---
    dtr_xgb = xgb.DMatrix(X_tr, label=y_tr)
    dva_xgb = xgb.DMatrix(X_va, label=y_va)
    dte_xgb = xgb.DMatrix(X_test_mat)
    model_xgb = xgb.train(
        xgb_params,
        dtr_xgb,
        num_boost_round=2000,
        evals=[(dva_xgb, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    val_pred_xgb = np.clip(
        model_xgb.predict(dva_xgb, iteration_range=(0, model_xgb.best_iteration + 1)),
        0,
        None,
    )
    oof_xgb[val_idx] = val_pred_xgb
    test_preds_xgb += (
        np.clip(
            model_xgb.predict(
                dte_xgb, iteration_range=(0, model_xgb.best_iteration + 1)
            ),
            0,
            None,
        )
        / N_SPLITS
    )

    # --- CatBoost ---
    model_cb = CatBoostRegressor(**cb_params)
    model_cb.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        use_best_model=True,
    )
    val_pred_cb = np.clip(model_cb.predict(X_va), 0, None)
    oof_cb[val_idx] = val_pred_cb
    test_preds_cb += (
        np.clip(model_cb.predict(X_test_mat), 0, None) / N_SPLITS
    )

    # --- HistGradientBoosting ---
    model_hgb = HistGradientBoostingRegressor(**hgb_params)
    model_hgb.fit(X_tr, y_tr)
    val_pred_hgb = np.clip(model_hgb.predict(X_va), 0, None)
    oof_hgb[val_idx] = val_pred_hgb
    test_preds_hgb += (
        np.clip(model_hgb.predict(X_test_mat), 0, None) / N_SPLITS
    )

    fold_rmse_lgb = root_mean_squared_error(y_va, val_pred_lgb)
    fold_rmse_xgb = root_mean_squared_error(y_va, val_pred_xgb)
    fold_rmse_cb = root_mean_squared_error(y_va, val_pred_cb)
    fold_rmse_hgb = root_mean_squared_error(y_va, val_pred_hgb)
    print(
        f"Fold {fold+1}/{N_SPLITS} | LGB: {fold_rmse_lgb:.2f} | XGB: {fold_rmse_xgb:.2f} | CB: {fold_rmse_cb:.2f} | HGB: {fold_rmse_hgb:.2f}"
    )


# =============================================================================
# 5. Out-of-Fold Constrained Ensemble Optimization
# =============================================================================
def ensemble_objective(weights):
    w = np.array(weights)
    w = w / np.sum(w)
    pred = w[0] * oof_lgb + w[1] * oof_xgb + w[2] * oof_cb + w[3] * oof_hgb
    pred = np.clip(pred, 0, None)
    return root_mean_squared_error(y_train_arr, pred)


opt_res = minimize(
    ensemble_objective,
    [0.35, 0.35, 0.20, 0.10],
    bounds=[(0, 1), (0, 1), (0, 1), (0, 1)],
    method="SLSQP",
)
opt_w = opt_res.x / np.sum(opt_res.x)
print(
    f"Optimal Ensemble Weights: LGBM={opt_w[0]:.4f}, XGB={opt_w[1]:.4f}, CB={opt_w[2]:.4f}, HGB={opt_w[3]:.4f}"
)

oof_ensemble = (
    opt_w[0] * oof_lgb
    + opt_w[1] * oof_xgb
    + opt_w[2] * oof_cb
    + opt_w[3] * oof_hgb
)
oof_ensemble = np.clip(oof_ensemble, 0, None)
final_score = root_mean_squared_error(y_train_arr, oof_ensemble)

# =============================================================================
# 6. Submission Generation
# =============================================================================
test_final_preds = (
    opt_w[0] * test_preds_lgb
    + opt_w[1] * test_preds_xgb
    + opt_w[2] * test_preds_cb
    + opt_w[3] * test_preds_hgb
)
test_final_preds = np.clip(test_final_preds, 0, None)

submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df = pd.DataFrame(
    {
        "id": np.arange(len(test_final_preds), dtype=np.int64),
        TARGET_COL: test_final_preds,
    }
)
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path} with shape {submission_df.shape}")

# Final validation metric
print(f"Final Validation Score: {final_score}")