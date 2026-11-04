import json
import os
import re
import warnings
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge, RidgeCV
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. Configuration & Reproducibility
# -----------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# -----------------------------------------------------------------------------
# 2. Data Ingestion & Preprocessing
# -----------------------------------------------------------------------------
print("Loading train and test datasets from ./input...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "installed_capacity"
if target_col not in train_df.columns:
    matching = [c for c in train_df.columns if "capacity" in c.lower()]
    target_col = matching[0] if matching else train_df.columns[-1]

print(
    f"Dataset loaded. Train shape: {train_df.shape}, Test shape: {test_df.shape}, Target: {target_col}"
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

y_train = train_df[target_col].copy()
X_train_raw = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.copy()
if target_col in X_test_raw.columns:
    X_test_raw = X_test_raw.drop(columns=[target_col])

combined_df = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)
n_train = len(train_df)
n_test = len(test_df)

# Column type categorization
datetime_cols = []
numeric_cols = []
categorical_cols = []
text_cols = []
coord_cols = []

for col in combined_df.columns:
    col_lower = col.lower()
    if "lat" in col_lower or "lon" in col_lower or "coord" in col_lower:
        coord_cols.append(col)
    elif pd.api.types.is_datetime64_any_dtype(combined_df[col]):
        datetime_cols.append(col)
    elif pd.api.types.is_numeric_dtype(combined_df[col]):
        sample_vals = combined_df[col].dropna()
        if (
            len(sample_vals) > 0
            and sample_vals.between(1900, 2100).mean() > 0.8
            and ("year" in col_lower or "date" in col_lower)
        ):
            datetime_cols.append(col)
        elif combined_df[col].nunique() < 10 and not pd.api.types.is_float_dtype(
            combined_df[col]
        ):
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)
    else:
        avg_len = combined_df[col].astype(str).str.len().mean()
        num_unique = combined_df[col].nunique()
        if avg_len > 40 or (num_unique > 0.6 * len(combined_df) and num_unique > 50):
            text_cols.append(col)
        else:
            categorical_cols.append(col)

print(
    f"Identified features: Numeric={len(numeric_cols)}, Categorical={len(categorical_cols)}, "
    f"Date/Year={len(datetime_cols)}, Text={len(text_cols)}, Coordinates={len(coord_cols)}"
)

features_df = pd.DataFrame(index=combined_df.index)

# -----------------------------------------------------------------------------
# 3. Feature Engineering
# -----------------------------------------------------------------------------
# Temporal Features
year_numeric_cols = []
for col in datetime_cols:
    if pd.api.types.is_datetime64_any_dtype(combined_df[col]):
        dt_series = pd.to_datetime(combined_df[col], errors="coerce")
        features_df[f"{col}_year"] = dt_series.dt.year
        features_df[f"{col}_month"] = dt_series.dt.month
        features_df[f"{col}_day"] = dt_series.dt.day
        features_df[f"{col}_dayofweek"] = dt_series.dt.dayofweek
        features_df[f"{col}_is_missing"] = dt_series.isna().astype(int)
        year_numeric_cols.append(f"{col}_year")
    else:
        col_vals = pd.to_numeric(combined_df[col], errors="coerce")
        features_df[f"{col}_numeric_year"] = col_vals
        features_df[f"{col}_is_missing"] = col_vals.isna().astype(int)
        year_numeric_cols.append(f"{col}_numeric_year")

if len(year_numeric_cols) >= 2:
    for i in range(len(year_numeric_cols)):
        for j in range(i + 1, len(year_numeric_cols)):
            col1, col2 = year_numeric_cols[i], year_numeric_cols[j]
            features_df[f"diff_{col1}_minus_{col2}"] = (
                features_df[col1] - features_df[col2]
            )

# Spatial Features
lat_col = next((c for c in coord_cols if "lat" in c.lower()), None)
lon_col = next((c for c in coord_cols if "lon" in c.lower()), None)

if lat_col and lon_col:
    lat = pd.to_numeric(combined_df[lat_col], errors="coerce")
    lon = pd.to_numeric(combined_df[lon_col], errors="coerce")
    features_df["geo_latitude"] = lat
    features_df["geo_longitude"] = lon
    features_df["geo_lat_missing"] = lat.isna().astype(int)
    features_df["geo_lon_missing"] = lon.isna().astype(int)

    lat_rad = np.radians(lat.fillna(lat.median()))
    lon_rad = np.radians(lon.fillna(lon.median()))
    features_df["geo_x"] = np.cos(lat_rad) * np.cos(lon_rad)
    features_df["geo_y"] = np.cos(lat_rad) * np.sin(lon_rad)
    features_df["geo_z"] = np.sin(lat_rad)

# Numeric Features - Clean & Compact representation
for col in numeric_cols:
    num_series = pd.to_numeric(combined_df[col], errors="coerce")
    features_df[f"{col}_raw"] = num_series
    features_df[f"{col}_isna"] = num_series.isna().astype(int)
    valid_min = num_series.dropna().min() if len(num_series.dropna()) > 0 else 0
    if valid_min >= 0:
        features_df[f"{col}_log1p"] = np.log1p(np.maximum(0, num_series.fillna(0)))

# Categorical Features - Frequency & Categorical Codes
for col in categorical_cols:
    str_series = combined_df[col].fillna("__MISSING__").astype(str)
    freq_map = str_series.value_counts(normalize=True).to_dict()
    features_df[f"{col}_freq"] = str_series.map(freq_map).astype(float)
    features_df[f"{col}_catcode"] = pd.factorize(str_series)[0]

# Text NLP Features - Lightweight character and word counts (removing noisy SVD)
for col in text_cols:
    text_data = combined_df[col].fillna("").astype(str)
    features_df[f"{col}_char_len"] = text_data.str.len()
    features_df[f"{col}_word_count"] = text_data.str.split().str.len()

# Group-Level Aggregation Features (Technology, Fuel, Country, Region baselines)
group_cat_candidates = [
    c for c in categorical_cols if 2 <= combined_df[c].nunique() <= 50
]
top_group_cats = group_cat_candidates[:min(4, len(group_cat_candidates))]
top_num_for_agg = numeric_cols[:min(6, len(numeric_cols))]

for cat_c in top_group_cats:
    cat_series = combined_df[cat_c].fillna("__MISSING__").astype(str)
    for num_c in top_num_for_agg:
        num_series = pd.to_numeric(combined_df[num_c], errors="coerce")
        grp_mean = num_series.groupby(cat_series).transform("mean")
        grp_std = num_series.groupby(cat_series).transform("std").fillna(0.0)
        grp_median = num_series.groupby(cat_series).transform("median")
        features_df[f"grp_{cat_c}_{num_c}_mean"] = grp_mean
        features_df[f"grp_{cat_c}_{num_c}_std"] = grp_std
        features_df[f"grp_{cat_c}_{num_c}_diff"] = num_series - grp_median

# Ratio & Interaction Features
if len(numeric_cols) >= 2:
    for i in range(min(4, len(numeric_cols))):
        for j in range(i + 1, min(4, len(numeric_cols))):
            c1, c2 = numeric_cols[i], numeric_cols[j]
            s1 = pd.to_numeric(combined_df[c1], errors="coerce").fillna(0)
            s2 = pd.to_numeric(combined_df[c2], errors="coerce").fillna(0)
            features_df[f"ratio_{c1}_{c2}"] = s1 / (s2.abs() + 1.0)
            features_df[f"prod_{c1}_{c2}"] = s1 * s2

# Train / Test Splitting of Processed Features
X_train_processed = features_df.iloc[:n_train].copy().reset_index(drop=True)
X_test_processed = features_df.iloc[n_train:].copy().reset_index(drop=True)

# Missing Value Imputation
for c in X_train_processed.columns:
    med_val = X_train_processed[c].median()
    if pd.isna(med_val):
        med_val = 0.0
    X_train_processed[c] = X_train_processed[c].fillna(med_val)
    X_test_processed[c] = X_test_processed[c].fillna(med_val)

# Feature Pruning via Variance Thresholding and Collinearity Filtering
var_series = X_train_processed.var()
non_zero_cols = var_series[var_series > 1e-5].index.tolist()
X_train_processed = X_train_processed[non_zero_cols]
X_test_processed = X_test_processed[non_zero_cols]

# Drop highly collinear features (corr > 0.95) to retain ~45-60 robust features
corr_matrix = X_train_processed.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
kept_cols = [c for c in X_train_processed.columns if c not in to_drop]

if len(kept_cols) < 40:
    kept_cols = non_zero_cols[:min(60, len(non_zero_cols))]

X_train_processed = X_train_processed[kept_cols]
X_test_processed = X_test_processed[kept_cols]
print(f"Features pruned from {features_df.shape[1]} to {len(kept_cols)} high-signal features.")

# Out-of-Fold Target Encoding (Strictly on Train Folds)
num_bins = 5
binned_y = pd.qcut(y_train.rank(method="first"), q=num_bins, labels=False)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_target_enc = pd.DataFrame(index=X_train_processed.index)
test_target_enc = pd.DataFrame(index=X_test_processed.index)

high_imp_cats = categorical_cols[: min(5, len(categorical_cols))]
for col in high_imp_cats:
    oof_target_enc[f"{col}_target_mean"] = np.nan
    test_fold_means = []

    cat_series_train = combined_df[col].iloc[:n_train].fillna("__MISSING__").astype(str)
    cat_series_test = combined_df[col].iloc[n_train:].fillna("__MISSING__").astype(str)
    global_mean = y_train.mean()

    for train_idx, val_idx in skf.split(X_train_processed, binned_y):
        tr_cats = cat_series_train.iloc[train_idx]
        tr_y = y_train.iloc[train_idx]

        cat_stats = tr_y.groupby(tr_cats).agg(["mean", "count"])
        smoothing = 15.0
        smoothed = (
            cat_stats["count"] * cat_stats["mean"] + smoothing * global_mean
        ) / (cat_stats["count"] + smoothing)

        val_cats = cat_series_train.iloc[val_idx]
        oof_target_enc.loc[val_idx, f"{col}_target_mean"] = val_cats.map(
            smoothed
        ).fillna(global_mean)
        test_fold_means.append(cat_series_test.map(smoothed).fillna(global_mean))

    test_target_enc[f"{col}_target_mean"] = np.mean(test_fold_means, axis=0)

X_train_final = pd.concat([X_train_processed, oof_target_enc], axis=1)
X_test_final = pd.concat([X_test_processed, test_target_enc], axis=1)

folds_arr = np.zeros(n_train, dtype=int)
for fold_num, (_, val_idx) in enumerate(skf.split(X_train_final, binned_y)):
    folds_arr[val_idx] = fold_num

feature_names = list(X_train_final.columns)
num_input_features = len(feature_names)
print(
    f"Engineered feature set dimensions: Train={X_train_final.shape}, Test={X_test_final.shape}"
)


# -----------------------------------------------------------------------------
# 4. Model Architectures & Parameter Specifications
# -----------------------------------------------------------------------------
def get_lgb_params(seed: int = 42) -> dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 12,
        "max_depth": 3,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 16,
        "reg_alpha": 0.5,
        "reg_lambda": 3.0,
        "verbosity": -1,
        "random_state": seed,
        "n_estimators": 1500,
    }


def get_xgb_params(seed: int = 42) -> dict:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_child_weight": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 3.0,
        "random_state": seed,
        "n_estimators": 1500,
    }


def get_cat_params(seed: int = 42) -> dict:
    return {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "learning_rate": 0.03,
        "depth": 4,
        "l2_leaf_reg": 5.0,
        "random_seed": seed,
        "iterations": 1500,
        "verbose": False,
    }


def get_et_params(seed: int = 42) -> dict:
    return {
        "n_estimators": 350,
        "max_depth": 8,
        "min_samples_split": 6,
        "min_samples_leaf": 3,
        "max_features": 0.7,
        "random_state": seed,
        "n_jobs": -1,
    }


# -----------------------------------------------------------------------------
# 5. Model Training & Cross-Validation
# -----------------------------------------------------------------------------
X = np.nan_to_num(
    X_train_final.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
)
y = y_train.values.astype(np.float32)
X_test = np.nan_to_num(
    X_test_final.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
)

n_samples = len(X)
n_folds = 5

oof_preds_lgb = np.zeros(n_samples, dtype=np.float32)
oof_preds_xgb = np.zeros(n_samples, dtype=np.float32)
oof_preds_cat = np.zeros(n_samples, dtype=np.float32)
oof_preds_et = np.zeros(n_samples, dtype=np.float32)
oof_preds_ridge = np.zeros(n_samples, dtype=np.float32)
oof_preds_huber = np.zeros(n_samples, dtype=np.float32)

test_preds_lgb = np.zeros(n_test, dtype=np.float32)
test_preds_xgb = np.zeros(n_test, dtype=np.float32)
test_preds_cat = np.zeros(n_test, dtype=np.float32)
test_preds_et = np.zeros(n_test, dtype=np.float32)
test_preds_ridge = np.zeros(n_test, dtype=np.float32)
test_preds_huber = np.zeros(n_test, dtype=np.float32)

for fold in range(n_folds):
    val_idx = np.where(folds_arr == fold)[0]
    train_idx = np.where(folds_arr != fold)[0]

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[val_idx], y[val_idx]

    # --- 1. LightGBM ---
    lgb_params = get_lgb_params(seed=SEED + fold)
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    val_pred_lgb = lgb_model.predict(X_va)
    oof_preds_lgb[val_idx] = val_pred_lgb
    test_preds_lgb += lgb_model.predict(X_test) / n_folds

    # --- 2. XGBoost ---
    xgb_params = get_xgb_params(seed=SEED + fold)
    xgb_params["early_stopping_rounds"] = 50
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    val_pred_xgb = xgb_model.predict(X_va)
    oof_preds_xgb[val_idx] = val_pred_xgb
    test_preds_xgb += xgb_model.predict(X_test) / n_folds

    # --- 3. CatBoost ---
    cat_params = get_cat_params(seed=SEED + fold)
    cat_params["early_stopping_rounds"] = 50
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    val_pred_cat = cat_model.predict(X_va)
    oof_preds_cat[val_idx] = val_pred_cat
    test_preds_cat += cat_model.predict(X_test) / n_folds

    # --- 4. ExtraTrees ---
    et_params = get_et_params(seed=SEED + fold)
    et_model = ExtraTreesRegressor(**et_params)
    et_model.fit(X_tr, y_tr)
    val_pred_et = et_model.predict(X_va)
    oof_preds_et[val_idx] = val_pred_et
    test_preds_et += et_model.predict(X_test) / n_folds

    # --- 5. RidgeCV with RobustScaler ---
    ridge_pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-2, 4, 30)))
    ])
    ridge_pipeline.fit(X_tr, y_tr)
    val_pred_ridge = ridge_pipeline.predict(X_va)
    oof_preds_ridge[val_idx] = val_pred_ridge
    test_preds_ridge += ridge_pipeline.predict(X_test) / n_folds

    # --- 6. HuberRegressor with RobustScaler ---
    huber_pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("huber", HuberRegressor(max_iter=1000, alpha=1.0))
    ])
    huber_pipeline.fit(X_tr, y_tr)
    val_pred_huber = huber_pipeline.predict(X_va)
    oof_preds_huber[val_idx] = val_pred_huber
    test_preds_huber += huber_pipeline.predict(X_test) / n_folds

    fold_rmse_lgb = root_mean_squared_error(y_va, val_pred_lgb)
    fold_rmse_xgb = root_mean_squared_error(y_va, val_pred_xgb)
    fold_rmse_cat = root_mean_squared_error(y_va, val_pred_cat)
    fold_rmse_et = root_mean_squared_error(y_va, val_pred_et)
    fold_rmse_ridge = root_mean_squared_error(y_va, val_pred_ridge)
    fold_rmse_huber = root_mean_squared_error(y_va, val_pred_huber)
    print(
        f"Fold {fold+1}/{n_folds} - LGB: {fold_rmse_lgb:.4f} | XGB: {fold_rmse_xgb:.4f} | CAT: {fold_rmse_cat:.4f} | ET: {fold_rmse_et:.4f} | Ridge: {fold_rmse_ridge:.4f} | Huber: {fold_rmse_huber:.4f}"
    )

# -----------------------------------------------------------------------------
# 6. Out-of-Fold Blending Optimization & Final Predictions
# -----------------------------------------------------------------------------
rmse_lgb_oof = root_mean_squared_error(y, oof_preds_lgb)
rmse_xgb_oof = root_mean_squared_error(y, oof_preds_xgb)
rmse_cat_oof = root_mean_squared_error(y, oof_preds_cat)
rmse_et_oof = root_mean_squared_error(y, oof_preds_et)
rmse_ridge_oof = root_mean_squared_error(y, oof_preds_ridge)
rmse_huber_oof = root_mean_squared_error(y, oof_preds_huber)

print(
    f"OOF RMSE Scores -> LGBM: {rmse_lgb_oof:.4f} | XGBoost: {rmse_xgb_oof:.4f} | CatBoost: {rmse_cat_oof:.4f} | ExtraTrees: {rmse_et_oof:.4f} | Ridge: {rmse_ridge_oof:.4f} | Huber: {rmse_huber_oof:.4f}"
)

oof_matrix = np.column_stack([oof_preds_lgb, oof_preds_xgb, oof_preds_cat, oof_preds_et, oof_preds_ridge, oof_preds_huber])
test_matrix = np.column_stack([test_preds_lgb, test_preds_xgb, test_preds_cat, test_preds_et, test_preds_ridge, test_preds_huber])
n_models = oof_matrix.shape[1]


def blend_objective(weights):
    pred = np.dot(oof_matrix, weights)
    return root_mean_squared_error(y, np.maximum(0.0, pred))


initial_weights = np.ones(n_models) / float(n_models)
bounds = [(0.0, 1.0) for _ in range(n_models)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    blend_objective,
    initial_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
opt_w = opt_res.x if opt_res.success else np.ones(n_models) / float(n_models)

print(
    f"Optimized Blend Weights: LGBM={opt_w[0]:.4f}, XGBoost={opt_w[1]:.4f}, CatBoost={opt_w[2]:.4f}, ExtraTrees={opt_w[3]:.4f}, Ridge={opt_w[4]:.4f}, Huber={opt_w[5]:.4f}"
)

final_oof_blend = np.maximum(0.0, np.dot(oof_matrix, opt_w))
final_validation_score = root_mean_squared_error(y, final_oof_blend)

final_test_preds = np.maximum(0.0, np.dot(test_matrix, opt_w))

# -----------------------------------------------------------------------------
# 7. Generate Submission & Log Final Score
# -----------------------------------------------------------------------------
submission_df = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "installed_capacity": final_test_preds}
)

sub_path = "./submission/submission.csv"
submission_df.to_csv(sub_path, index=False)
print(f"Submission saved to {sub_path} with {len(submission_df)} rows.")

print(f"Final Validation Score: {final_validation_score}")
