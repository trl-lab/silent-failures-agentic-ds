import os
import gc
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
np.random.seed(42)

# ==============================================================================
# 1. DATA LOADING & DIRECTORY SETUP
# ==============================================================================
input_dir = "./input"
working_dir = "./working"
submission_dir = "./submission"

os.makedirs(working_dir, exist_ok=True)
os.makedirs(submission_dir, exist_ok=True)

train_path = os.path.join(input_dir, "train.parquet")
test_path = os.path.join(input_dir, "test.parquet")

print("Loading raw datasets...")
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "projectPayment"
assert (
    target_col in train_df.columns
), f"Target column '{target_col}' not found in train data."

y_train_series = train_df[target_col].copy()
X_train_raw = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.copy()

test_ids = np.arange(len(X_test_raw), dtype=np.int32)
n_train = len(X_train_raw)
n_test = len(X_test_raw)

# ==============================================================================
# 2. FEATURE IDENTIFICATION & TYPE INFERENCE
# ==============================================================================
all_cols = list(X_train_raw.columns)
date_cols = []
numeric_cols = []
categorical_cols = []
text_cols = []

for col in all_cols:
    if pd.api.types.is_datetime64_any_dtype(X_train_raw[col]):
        date_cols.append(col)
    elif X_train_raw[col].dtype == "object" or pd.api.types.is_categorical_dtype(
        X_train_raw[col]
    ):
        sample_non_null = X_train_raw[col].dropna().head(50)
        is_date = False
        if len(sample_non_null) > 0:
            try:
                pd.to_datetime(sample_non_null, errors="raise")
                is_date = True
            except Exception:
                is_date = False
        if is_date:
            date_cols.append(col)
        else:
            unique_count = X_train_raw[col].nunique()
            if (
                unique_count > 5000
                and X_train_raw[col].astype(str).str.len().mean() > 50
            ):
                text_cols.append(col)
            else:
                categorical_cols.append(col)
    elif pd.api.types.is_numeric_dtype(X_train_raw[col]):
        numeric_cols.append(col)
    else:
        categorical_cols.append(col)


# ==============================================================================
# 3. BASE FEATURE EXTRACTION
# ==============================================================================
def extract_base_features(df_source):
    df = df_source.copy()
    extracted = pd.DataFrame(index=df.index)

    # A. Date / Time features
    for col in date_cols:
        dt_series = pd.to_datetime(df[col], errors="coerce")
        extracted[f"{col}_year"] = dt_series.dt.year.fillna(-1).astype(np.int32)
        extracted[f"{col}_month"] = dt_series.dt.month.fillna(-1).astype(np.int32)
        extracted[f"{col}_day"] = dt_series.dt.day.fillna(-1).astype(np.int32)
        extracted[f"{col}_quarter"] = dt_series.dt.quarter.fillna(-1).astype(np.int32)
        extracted[f"{col}_dayofweek"] = dt_series.dt.dayofweek.fillna(-1).astype(
            np.int32
        )
        extracted[f"{col}_is_year_end"] = dt_series.dt.is_year_end.astype(
            np.float32
        ).fillna(0.0)
        extracted[f"{col}_is_quarter_end"] = dt_series.dt.is_quarter_end.astype(
            np.float32
        ).fillna(0.0)
        extracted[f"{col}_month_sin"] = (
            np.sin(2 * np.pi * dt_series.dt.month / 12.0).fillna(0.0).astype(np.float32)
        )
        extracted[f"{col}_month_cos"] = (
            np.cos(2 * np.pi * dt_series.dt.month / 12.0).fillna(0.0).astype(np.float32)
        )

    # B. Text Metadata features
    for col in text_cols + categorical_cols:
        s = df[col].astype(str).fillna("")
        extracted[f"{col}_len"] = s.str.len().astype(np.int32)
        extracted[f"{col}_word_count"] = s.apply(lambda x: len(x.split())).astype(
            np.int32
        )
        extracted[f"{col}_digit_count"] = s.apply(
            lambda x: sum(c.isdigit() for c in x)
        ).astype(np.int32)
        extracted[f"{col}_upper_count"] = s.apply(
            lambda x: sum(c.isupper() for c in x)
        ).astype(np.int32)

    # C. Missingness Indicators & Numeric Transformations
    for col in numeric_cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        extracted[f"{col}_isna"] = vals.isna().astype(np.float32)
        extracted[col] = vals.fillna(0.0).astype(np.float32)
        min_val = vals.min()
        if min_val >= 0:
            extracted[f"{col}_log1p"] = np.log1p(
                np.maximum(vals.fillna(0.0), 0.0)
            ).astype(np.float32)

    # D. Categorical Pairwise Interaction Strings
    cat_subset = categorical_cols[: min(5, len(categorical_cols))]
    for i in range(len(cat_subset)):
        for j in range(i + 1, len(cat_subset)):
            c1, c2 = cat_subset[i], cat_subset[j]
            extracted[f"inter_{c1}_{c2}"] = (
                df[c1].astype(str) + "___" + df[c2].astype(str)
            )

    for col in categorical_cols:
        extracted[col] = df[col].astype(str).fillna("MISSING")

    return extracted


print("Extracting base features on train and test...")
X_train_fe = extract_base_features(X_train_raw)
X_test_fe = extract_base_features(X_test_raw)

# ==============================================================================
# 4. CROSS-VALIDATION SPLIT SETUP
# ==============================================================================
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_indices = list(kf.split(X_train_fe, y_train_series))

# ==============================================================================
# 5. STATEFUL LEAK-FREE FEATURE ENGINEERING
# ==============================================================================
# A. TF-IDF + TruncatedSVD Text Features
text_repr_cols = text_cols + categorical_cols
if len(text_repr_cols) > 0:
    combined_train_text = (
        X_train_raw[text_repr_cols].astype(str).fillna("").agg(" ".join, axis=1)
    )
    combined_test_text = (
        X_test_raw[text_repr_cols].astype(str).fillna("").agg(" ".join, axis=1)
    )

    tfidf = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b"
    )
    svd = TruncatedSVD(n_components=16, random_state=42)

    tfidf_train = tfidf.fit_transform(combined_train_text)
    tfidf_test = tfidf.transform(combined_test_text)

    svd_train = svd.fit_transform(tfidf_train)
    svd_test = svd.transform(tfidf_test)

    for i in range(svd_train.shape[1]):
        X_train_fe[f"text_svd_{i}"] = svd_train[:, i].astype(np.float32)
        X_test_fe[f"text_svd_{i}"] = svd_test[:, i].astype(np.float32)

# B. Frequency / Count Encoding
all_cat_cols = [
    c
    for c in X_train_fe.columns
    if X_train_fe[c].dtype == "object" or c in categorical_cols
]
for col in all_cat_cols:
    freq_map = X_train_fe[col].value_counts().to_dict()
    X_train_fe[f"{col}_freq"] = (
        X_train_fe[col].map(freq_map).fillna(0).astype(np.float32)
    )
    X_test_fe[f"{col}_freq"] = (
        X_test_fe[col].map(freq_map).fillna(0).astype(np.float32)
    )

# C. Hierarchical Group-by Aggregations & Within-Group Normalized Features
agg_num_cols = [c for c in numeric_cols if c in X_train_fe.columns]
if len(agg_num_cols) > 0 and len(categorical_cols) > 0:
    primary_group_cols = categorical_cols[: min(4, len(categorical_cols))]
    for grp_col in primary_group_cols:
        for num_col in agg_num_cols[: min(4, len(agg_num_cols))]:
            grp_stats = (
                X_train_fe.groupby(grp_col)[num_col]
                .agg(["mean", "std", "min", "max", "count"])
                .reset_index()
            )
            grp_stats.columns = [grp_col] + [
                f"{grp_col}_{num_col}_{stat}"
                for stat in ["mean", "std", "min", "max", "count"]
            ]

            X_train_fe = X_train_fe.merge(grp_stats, on=grp_col, how="left")
            X_test_fe = X_test_fe.merge(grp_stats, on=grp_col, how="left")

            mean_col = f"{grp_col}_{num_col}_mean"
            std_col = f"{grp_col}_{num_col}_std"
            min_col = f"{grp_col}_{num_col}_min"
            max_col = f"{grp_col}_{num_col}_max"
            count_col = f"{grp_col}_{num_col}_count"

            global_mean_val = float(X_train_fe[num_col].mean())
            X_train_fe[mean_col] = (
                X_train_fe[mean_col].fillna(global_mean_val).astype(np.float32)
            )
            X_test_fe[mean_col] = (
                X_test_fe[mean_col].fillna(global_mean_val).astype(np.float32)
            )

            X_train_fe[std_col] = (
                X_train_fe[std_col].fillna(0.0).astype(np.float32)
            )
            X_test_fe[std_col] = (
                X_test_fe[std_col].fillna(0.0).astype(np.float32)
            )

            X_train_fe[min_col] = (
                X_train_fe[min_col].fillna(global_mean_val).astype(np.float32)
            )
            X_test_fe[min_col] = (
                X_test_fe[min_col].fillna(global_mean_val).astype(np.float32)
            )

            X_train_fe[max_col] = (
                X_train_fe[max_col].fillna(global_mean_val).astype(np.float32)
            )
            X_test_fe[max_col] = (
                X_test_fe[max_col].fillna(global_mean_val).astype(np.float32)
            )

            X_train_fe[count_col] = (
                X_train_fe[count_col].fillna(0.0).astype(np.float32)
            )
            X_test_fe[count_col] = (
                X_test_fe[count_col].fillna(0.0).astype(np.float32)
            )

            # Within-group normalized features
            diff_col = f"{grp_col}_{num_col}_diff_mean"
            zscore_col = f"{grp_col}_{num_col}_zscore"
            ratio_col = f"{grp_col}_{num_col}_ratio_mean"

            X_train_fe[diff_col] = (
                X_train_fe[num_col] - X_train_fe[mean_col]
            ).astype(np.float32)
            X_test_fe[diff_col] = (
                X_test_fe[num_col] - X_test_fe[mean_col]
            ).astype(np.float32)

            X_train_fe[zscore_col] = (
                (X_train_fe[num_col] - X_train_fe[mean_col])
                / (X_train_fe[std_col] + 1e-5)
            ).astype(np.float32)
            X_test_fe[zscore_col] = (
                (X_test_fe[num_col] - X_test_fe[mean_col])
                / (X_test_fe[std_col] + 1e-5)
            ).astype(np.float32)

            X_train_fe[ratio_col] = (
                (X_train_fe[num_col] + 1e-5)
                / (X_train_fe[mean_col] + 1e-5)
            ).astype(np.float32)
            X_test_fe[ratio_col] = (
                (X_test_fe[num_col] + 1e-5)
                / (X_test_fe[mean_col] + 1e-5)
            ).astype(np.float32)

# D. Smooth Bayesian Out-of-Fold (OOF) Target Encoding
smooth_m = 10.0
global_mean = float(y_train_series.mean())

te_targets = categorical_cols + [
    c for c in X_train_fe.columns if c.startswith("inter_")
][:5]

for col in te_targets:
    te_col = f"{col}_target_enc"
    X_train_fe[te_col] = np.nan

    for trn_idx, val_idx in fold_indices:
        trn_df_fold = pd.DataFrame(
            {
                col: X_train_fe.iloc[trn_idx][col].values,
                "target": y_train_series.iloc[trn_idx].values,
            }
        )
        stats = trn_df_fold.groupby(col)["target"].agg(["count", "mean"])
        smooth_enc = (
            stats["count"] * stats["mean"] + smooth_m * global_mean
        ) / (stats["count"] + smooth_m)
        enc_dict = smooth_enc.to_dict()

        val_mapped = (
            X_train_fe.iloc[val_idx][col]
            .map(enc_dict)
            .fillna(global_mean)
            .values.astype(np.float32)
        )
        X_train_fe.iloc[val_idx, X_train_fe.columns.get_loc(te_col)] = val_mapped

    full_stats = (
        pd.DataFrame(
            {col: X_train_fe[col].values, "target": y_train_series.values}
        )
        .groupby(col)["target"]
        .agg(["count", "mean"])
    )
    full_smooth_enc = (
        full_stats["count"] * full_stats["mean"] + smooth_m * global_mean
    ) / (full_stats["count"] + smooth_m)
    full_enc_dict = full_smooth_enc.to_dict()

    X_test_fe[te_col] = (
        X_test_fe[col].map(full_enc_dict).fillna(global_mean).astype(np.float32)
    )
    X_train_fe[te_col] = X_train_fe[te_col].astype(np.float32)

# E. Label Encoding for remaining categoricals
cat_columns_final = [
    c for c in X_train_fe.columns if X_train_fe[c].dtype == "object"
]
for col in cat_columns_final:
    le = LabelEncoder()
    train_vals = X_train_fe[col].astype(str).fillna("MISSING")
    le.fit(list(train_vals.unique()) + ["MISSING", "UNKNOWN_UNSEEN"])

    mapping = {cls_: idx for idx, cls_ in enumerate(le.classes_)}
    unknown_idx = mapping["UNKNOWN_UNSEEN"]

    X_train_fe[col] = (
        train_vals.map(mapping).fillna(unknown_idx).values.astype(np.int32)
    )
    test_vals = X_test_fe[col].astype(str).fillna("MISSING")
    X_test_fe[col] = (
        test_vals.map(mapping).fillna(unknown_idx).values.astype(np.int32)
    )

X_train_fe = X_train_fe.replace([np.inf, -np.inf], np.nan).fillna(0.0)
X_test_fe = X_test_fe.replace([np.inf, -np.inf], np.nan).fillna(0.0)

feature_names = list(X_train_fe.columns)
X_test_fe = X_test_fe[feature_names]

y_train = y_train_series.values.astype(np.float64)


# ==============================================================================
# 6. MODEL CONFIGURATIONS
# ==============================================================================
def get_lgb_params(seed: int = 42) -> dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_jobs": -1,
        "random_state": seed,
        "verbose": -1,
    }


def get_xgb_params(seed: int = 42) -> dict:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def get_cat_params(seed: int = 42) -> dict:
    return {
        "iterations": 3000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": seed,
        "verbose": False,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "early_stopping_rounds": 50,
        "thread_count": -1,
    }


# ==============================================================================
# 7. CROSS-VALIDATED TRAINING & EVALUATION
# ==============================================================================
oof_preds_lgb = np.zeros(n_train, dtype=np.float64)
oof_preds_xgb = np.zeros(n_train, dtype=np.float64)
oof_preds_cb = np.zeros(n_train, dtype=np.float64)

test_preds_lgb = np.zeros(n_test, dtype=np.float64)
test_preds_xgb = np.zeros(n_test, dtype=np.float64)
test_preds_cb = np.zeros(n_test, dtype=np.float64)

print(
    f"Training models on {n_train} samples, {n_test} test samples with {len(feature_names)} features..."
)

for fold, (trn_idx, val_idx) in enumerate(fold_indices):
    X_tr, y_tr = X_train_fe.iloc[trn_idx], y_train[trn_idx]
    X_va, y_va = X_train_fe.iloc[val_idx], y_train[val_idx]

    # --- A. LightGBM ---
    lgb_train_data = lgb.Dataset(X_tr, label=y_tr)
    lgb_val_data = lgb.Dataset(X_va, label=y_va, reference=lgb_train_data)
    lgb_params = get_lgb_params(seed=42 + fold)

    lgb_model = lgb.train(
        params=lgb_params,
        train_set=lgb_train_data,
        num_boost_round=3000,
        valid_sets=[lgb_val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    val_pred_lgb = lgb_model.predict(X_va, num_iteration=lgb_model.best_iteration)
    oof_preds_lgb[val_idx] = val_pred_lgb
    test_preds_lgb += (
        lgb_model.predict(X_test_fe, num_iteration=lgb_model.best_iteration) / n_splits
    )

    # --- B. XGBoost ---
    xgb_params = get_xgb_params(seed=42 + fold)
    xgb_train_data = xgb.DMatrix(X_tr, label=y_tr)
    xgb_val_data = xgb.DMatrix(X_va, label=y_va)
    xgb_test_data = xgb.DMatrix(X_test_fe)

    xgb_model = xgb.train(
        params=xgb_params,
        dtrain=xgb_train_data,
        num_boost_round=3000,
        evals=[(xgb_val_data, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    val_pred_xgb = xgb_model.predict(
        xgb_val_data, iteration_range=(0, xgb_model.best_iteration + 1)
    )
    oof_preds_xgb[val_idx] = val_pred_xgb
    test_preds_xgb += (
        xgb_model.predict(
            xgb_test_data, iteration_range=(0, xgb_model.best_iteration + 1)
        )
        / n_splits
    )

    # --- C. CatBoost ---
    cat_params = get_cat_params(seed=42 + fold)
    cb_model = CatBoostRegressor(**cat_params)
    cb_model.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=False,
    )
    val_pred_cb = cb_model.predict(X_va)
    oof_preds_cb[val_idx] = val_pred_cb
    test_preds_cb += cb_model.predict(X_test_fe) / n_splits

    fold_rmse_lgb = np.sqrt(mean_squared_error(y_va, val_pred_lgb))
    fold_rmse_xgb = np.sqrt(mean_squared_error(y_va, val_pred_xgb))
    fold_rmse_cb = np.sqrt(mean_squared_error(y_va, val_pred_cb))
    fold_rmse_blend = np.sqrt(
        mean_squared_error(
            y_va, (val_pred_lgb + val_pred_xgb + val_pred_cb) / 3.0
        )
    )
    print(
        f"Fold {fold+1}/{n_splits} | LGB RMSE: {fold_rmse_lgb:.4f} | XGB RMSE: {fold_rmse_xgb:.4f} | CB RMSE: {fold_rmse_cb:.4f} | Blend RMSE: {fold_rmse_blend:.4f}"
    )


# ==============================================================================
# 8. ENSEMBLE OPTIMIZATION & SUBMISSION
# ==============================================================================
oof_matrix = np.column_stack([oof_preds_lgb, oof_preds_xgb, oof_preds_cb])
test_matrix = np.column_stack([test_preds_lgb, test_preds_xgb, test_preds_cb])


def blend_objective(weights):
    w = np.array(weights)
    pred = np.dot(oof_matrix, w)
    return np.sqrt(mean_squared_error(y_train, pred))


init_weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
bounds = [(0.0, 1.0)] * 3
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    blend_objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
best_weights = opt_res.x if opt_res.success else init_weights
best_weights = np.maximum(0, best_weights)
best_weights /= np.sum(best_weights)

print(
    f"Optimal Blending Weights: LGB={best_weights[0]:.4f}, XGB={best_weights[1]:.4f}, CB={best_weights[2]:.4f}"
)

overall_oof_blend = np.dot(oof_matrix, best_weights)
final_validation_score = float(np.sqrt(mean_squared_error(y_train, overall_oof_blend)))

final_test_preds = np.dot(test_matrix, best_weights).astype(np.float64)

if np.min(y_train) >= 0.0:
    final_test_preds = np.clip(final_test_preds, a_min=0.0, a_max=None)

submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "projectPayment": final_test_preds,
    }
)

submission_path = os.path.join(submission_dir, "submission.csv")
submission_df.to_csv(submission_path, index=False)

assert os.path.exists(submission_path), "Submission file was not created."
assert (
    len(submission_df) == n_test
), f"Submission row count mismatch: {len(submission_df)} vs {n_test}"
assert list(submission_df.columns) == [
    "id",
    "projectPayment",
], f"Columns mismatch: {submission_df.columns}"
assert (
    not submission_df["projectPayment"].isna().any()
), "Submission contains NaN values."

print(f"Final Validation Score: {final_validation_score}")
