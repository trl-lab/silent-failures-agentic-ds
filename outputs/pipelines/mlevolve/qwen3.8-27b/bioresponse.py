import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import log_loss
from sklearn.feature_selection import mutual_info_classif
import os
import joblib
import warnings

warnings.filterwarnings("ignore")

# Create directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("Loading data...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Identify target and features
target = train_df["Activity"].values
feature_cols = [c for c in train_df.columns if c != "Activity"]
X_full = train_df[feature_cols].values

print(f"Target unique values: {np.unique(target)}")
print(f"Target distribution: {dict(zip(*np.unique(target, return_counts=True)))}")
print(f"Number of features: {len(feature_cols)}")

n_classes_check = len(np.unique(target))
if n_classes_check != 2:
    raise ValueError(
        f"This solution is designed for binary classification (2 classes), "
        f"but found {n_classes_check} classes: {np.unique(target)}. "
        f"All model objectives are set to 'binary' which will produce incorrect results."
    )

# Test features
X_test = (
    test_df[feature_cols].values
    if all(c in test_df.columns for c in feature_cols)
    else test_df.values
)

print(f"Missing values in train: {np.isnan(X_full).sum()}")
print(f"Missing values in test: {np.isnan(X_test).sum()}")

# ============================================================
# 2. CREATE STRATIFIED SPLIT (BEFORE any processing)
# ============================================================
print("\nCreating stratified train/val split...")
n_classes = len(np.unique(target))
print(f"Number of classes: {n_classes}")

X_train, X_val, y_train, y_val = train_test_split(
    X_full, target, test_size=0.2, random_state=42, stratify=target
)

print(f"Train split: {X_train.shape}, Val split: {X_val.shape}")
print(f"Train class dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"Val class dist: {dict(zip(*np.unique(y_val, return_counts=True)))}")

# ============================================================
# 3. FEATURE ENGINEERING (fit on train only)
# ============================================================
print("\n--- Feature Engineering Pipeline ---")

# Step 3a: Handle missing values (impute with median)
print("Handling missing values...")
medians = np.nanmedian(X_train, axis=0)
medians = np.where(np.isnan(medians), 0, medians)

X_train_imp = np.where(np.isnan(X_train), medians, X_train)
X_val_imp = np.where(np.isnan(X_val), medians, X_val)
X_test_imp = np.where(np.isnan(X_test), medians, X_test)

# Step 3b: Handle infinite values
X_train_imp = np.nan_to_num(X_train_imp, nan=0.0, posinf=0.0, neginf=0.0)
X_val_imp = np.nan_to_num(X_val_imp, nan=0.0, posinf=0.0, neginf=0.0)
X_test_imp = np.nan_to_num(X_test_imp, nan=0.0, posinf=0.0, neginf=0.0)

# Step 3c: Variance-based feature selection
print("Variance-based feature selection...")
var_threshold = VarianceThreshold(threshold=0.0)
var_threshold.fit(X_train_imp)
var_mask = var_threshold.get_support()
n_before_var = X_train_imp.shape[1]
X_train_var = X_train_imp[:, var_mask]
X_val_var = X_val_imp[:, var_mask]
X_test_var = X_test_imp[:, var_mask]
print(f"  Features after variance filter: {var_mask.sum()} / {n_before_var}")

# Step 3d: Correlation-based feature selection
print("Correlation-based feature selection...")
corr_matrix = np.corrcoef(X_train_var, rowvar=False)
corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

n_features = X_train_var.shape[1]
avg_abs_corr = np.abs(corr_matrix).sum(axis=0) / (n_features - 1)

corr_keep_mask = avg_abs_corr < 0.3
n_after_corr = corr_keep_mask.sum()
print(
    f"  Features after correlation filter (avg_corr < 0.3): {n_after_corr} / {n_features}"
)

if n_after_corr < 50:
    corr_keep_mask = avg_abs_corr < 0.5
    n_after_corr = corr_keep_mask.sum()
    print(f"  Relaxed to avg_corr < 0.5: {n_after_corr} features")

if n_after_corr < 50:
    corr_keep_mask = avg_abs_corr < 0.7
    n_after_corr = corr_keep_mask.sum()
    print(f"  Relaxed to avg_corr < 0.7: {n_after_corr} features")

if n_after_corr < 50:
    corr_keep_mask = np.ones(n_features, dtype=bool)
    print(f"  Keeping all {n_features} features, relying on PCA")

X_train_corr = X_train_var[:, corr_keep_mask]
X_val_corr = X_val_var[:, corr_keep_mask]
X_test_corr = X_test_var[:, corr_keep_mask]

# Step 3e: Standard Scaling (fit on train)
print("Standard scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_corr)
X_val_scaled = scaler.transform(X_val_corr)
X_test_scaled = scaler.transform(X_test_corr)

# Step 3f: Feature importance-based filter (stronger model, top 300)
print("Feature importance-based selection (top 300)...")
quick_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 8,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}
quick_train_data = lgb.Dataset(X_train_scaled, label=y_train)
quick_model = lgb.train(
    quick_params,
    quick_train_data,
    num_boost_round=500,
    valid_sets=[quick_train_data],
    valid_names=["train"],
    callbacks=[lgb.log_evaluation(period=500)],
)
feat_imp = quick_model.feature_importance(importance_type="gain")

# Mutual information as secondary criterion
print("  Computing mutual information scores...")
mi_scores = mutual_info_classif(X_train_scaled, y_train, random_state=42, n_neighbors=3)

# Combine: rank by gain importance, then filter by MI > 0
gain_rank = np.argsort(feat_imp)[::-1]
mi_positive_mask = mi_scores > 0

# Select top 300 from features with positive MI, ranked by gain importance
candidates = [i for i in gain_rank if mi_positive_mask[i]]
if len(candidates) < 300:
    candidates = list(gain_rank)
top_300_idx = np.array(candidates[:300])
print(f"  Keeping top 300 features by combined importance")

X_train_final = X_train_scaled[:, top_300_idx]
X_val_final = X_val_scaled[:, top_300_idx]
X_test_final = X_test_scaled[:, top_300_idx]

print(f"\nFinal shapes after feature engineering:")
print(f"  X_train: {X_train_final.shape}")
print(f"  X_val: {X_val_final.shape}")
print(f"  X_test: {X_test_final.shape}")

# ============================================================
# 4. MODEL PARAMETERS
# ============================================================
class_counts = np.bincount(y_train, minlength=n_classes)
class_weights = len(y_train) / (n_classes * class_counts)
class_weights = np.clip(class_weights, 0.5, 5.0)
print(f"\nClass weights: {class_weights}")

# LightGBM: 3 diverse configs
lgb_params_list = [
    {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 5,
        "min_child_samples": 30,
        "min_child_weight": 5,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "path_smoothing": 1.0,
        "verbose": -1,
        "n_jobs": -1,
    },
    {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 30,
        "min_child_weight": 5,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "path_smoothing": 1.0,
        "verbose": -1,
        "n_jobs": -1,
    },
    {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.03,
        "num_leaves": 127,
        "max_depth": 12,
        "min_child_samples": 30,
        "min_child_weight": 5,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "path_smoothing": 1.0,
        "verbose": -1,
        "n_jobs": -1,
    },
]

# XGBoost: 2 diverse configs
xgb_params_list = [
    {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.03,
        "max_depth": 5,
        "subsample": 0.7,
        "colsample_bytree": 0.6,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "n_jobs": -1,
        "verbosity": 0,
    },
    {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.03,
        "max_depth": 7,
        "subsample": 0.7,
        "colsample_bytree": 0.6,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "n_jobs": -1,
        "verbosity": 0,
    },
]

print(f"LightGBM configs: 3 (num_leaves: 31, 63, 127)")
print(f"XGBoost configs: 2 (max_depth: 5, 7)")
print(f"CatBoost: 1 (ordered boosting)")

# ============================================================
# 5. TRAINING: Diverse Ensemble (3 LightGBM + 2 XGBoost + 1 CatBoost)
# ============================================================
LGB_SEEDS = [42, 123, 777]
XGB_SEEDS = [42, 123]
EARLY_STOPPING_ROUNDS = 200
MAX_ROUNDS = 3000

print(f"\n--- Training Ensemble: 3 LightGBM + 2 XGBoost + 1 CatBoost ---")

models = []
val_scores_per_model = []
best_iterations = []
model_types = []  # 'lgb', 'xgb', 'cat'

# Train LightGBM models (3 diverse configs)
for i, (params_base, seed) in enumerate(zip(lgb_params_list, LGB_SEEDS)):
    params = params_base.copy()
    params["random_state"] = seed
    params["bagging_seed"] = seed
    params["feature_fraction_seed"] = seed

    train_data = lgb.Dataset(X_train_final, label=y_train)
    val_data = lgb.Dataset(X_val_final, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[val_data],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=500),
        ],
    )

    best_score = model.best_score["valid"]["binary_logloss"]
    val_scores_per_model.append(best_score)
    models.append(model)
    best_iterations.append(model.best_iteration)
    model_types.append("lgb")
    print(
        f"  LightGBM Config {i+1} (seed={seed}): best_iter={model.best_iteration}, val_logloss={best_score:.6f}"
    )

# Train XGBoost models (2 diverse configs)
for i, (params_base, seed) in enumerate(zip(xgb_params_list, XGB_SEEDS)):
    params = params_base.copy()
    params["seed"] = seed

    dtrain = xgb.DMatrix(X_train_final, label=y_train)
    dval = xgb.DMatrix(X_val_final, label=y_val)

    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=MAX_ROUNDS,
        evals=[(dval, "valid")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=500,
        evals_result=evals_result,
    )
    best_iteration = model.best_iteration
    best_score = min(evals_result["valid"]["logloss"])
    val_scores_per_model.append(best_score)
    models.append(model)
    best_iterations.append(best_iteration)
    model_types.append("xgb")
    print(
        f"  XGBoost Config {i+1} (seed={seed}): best_iter={best_iteration}, val_logloss={best_score:.6f}"
    )

# Train CatBoost model
print("  Training CatBoost...")
cat_model = CatBoostClassifier(
    iterations=3000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=500,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
)
cat_model.fit(
    X_train_final, y_train,
    eval_set=(X_val_final, y_val),
    verbose=500,
)
cat_best_iter = cat_model.get_best_iteration()
cat_val_pred = cat_model.predict_proba(X_val_final)[:, 1]
cat_val_score = log_loss(y_val, cat_val_pred)
val_scores_per_model.append(cat_val_score)
models.append(cat_model)
best_iterations.append(cat_best_iter)
model_types.append("cat")
print(f"  CatBoost: best_iter={cat_best_iter}, val_logloss={cat_val_score:.6f}")

# ============================================================
# 6. VALIDATION: Weighted Ensemble (no calibration)
# ============================================================
print("\n--- Validation (Weighted Ensemble, no calibration) ---")


def predict_ensemble(models, X, best_iterations, model_types, weights=None):
    """Generate weighted ensemble predictions for any input."""
    n_models = len(models)
    if weights is None:
        weights = np.ones(n_models) / n_models

    all_preds = np.zeros(X.shape[0])
    for idx, model in enumerate(models):
        mtype = model_types[idx]
        best_iter = best_iterations[idx]
        if mtype == "lgb":
            pred = model.predict(X, num_iteration=best_iter)
        elif mtype == "xgb":
            dmat = xgb.DMatrix(X)
            pred = model.predict(dmat, iteration_range=(0, best_iter))
        elif mtype == "cat":
            pred = model.predict_proba(X)[:, 1]
        pred = np.asarray(pred).flatten()
        all_preds += weights[idx] * pred
    all_preds = np.clip(all_preds, 0.001, 0.999)
    return all_preds


# Compute weights: inverse log_loss squared (better models get higher weight)
inv_scores_sq = (1.0 / np.array(val_scores_per_model)) ** 2
weights = inv_scores_sq / inv_scores_sq.sum()
print(f"  Model weights (inv-logloss^2): {[f'{w:.4f}' for w in weights]}")

# Validation predictions (weighted ensemble)
val_pred_proba = predict_ensemble(models, X_val_final, best_iterations, model_types, weights=weights)

# Compute validation log_loss directly (no calibration)
val_score = log_loss(y_val, val_pred_proba)

print(f"  Ensemble Validation Log Loss: {val_score:.6f}")
print(f"  Per-model scores: {[f'{s:.6f}' for s in val_scores_per_model]}")

# ============================================================
# 7. RETRAIN ON FULL DATA & GENERATE TEST PREDICTIONS
# ============================================================
print("\n--- Retraining on Full Data (train+val) for Test Predictions ---")

# Combine train and val for final training
X_full_train = np.vstack([X_train_final, X_val_final])
y_full_train = np.concatenate([y_train, y_val])

final_models = []
final_best_iterations = []
final_model_types = []

# Retrain LightGBM models on full data
for i, (params_base, seed) in enumerate(zip(lgb_params_list, LGB_SEEDS)):
    params = params_base.copy()
    params["random_state"] = seed
    params["bagging_seed"] = seed
    params["feature_fraction_seed"] = seed

    train_data = lgb.Dataset(X_full_train, label=y_full_train)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=best_iterations[i],
        valid_sets=[train_data],
        valid_names=["train"],
        callbacks=[lgb.log_evaluation(period=500)],
    )
    final_models.append(model)
    final_best_iterations.append(best_iterations[i])
    final_model_types.append("lgb")
    print(f"  LightGBM Config {i+1} retrained on full data (iter={best_iterations[i]})")

# Retrain XGBoost models on full data
for i, (params_base, seed) in enumerate(zip(xgb_params_list, XGB_SEEDS)):
    params = params_base.copy()
    params["seed"] = seed

    dtrain_full = xgb.DMatrix(X_full_train, label=y_full_train)
    offset = 3  # offset for LightGBM models
    model = xgb.train(
        params,
        dtrain_full,
        num_boost_round=best_iterations[offset + i],
        verbose_eval=500,
    )
    final_models.append(model)
    final_best_iterations.append(best_iterations[offset + i])
    final_model_types.append("xgb")
    print(f"  XGBoost Config {i+1} retrained on full data (iter={best_iterations[offset + i]})")

# Retrain CatBoost on full data
cat_model_full = CatBoostClassifier(
    iterations=best_iterations[5],
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=500,
)
cat_model_full.fit(X_full_train, y_full_train, verbose=500)
final_models.append(cat_model_full)
final_best_iterations.append(best_iterations[5])
final_model_types.append("cat")
print(f"  CatBoost retrained on full data (iter={best_iterations[5]})")

# Generate test predictions with retrained models
print("\n--- Generating Test Predictions (retrained models) ---")
test_pred_proba = predict_ensemble(
    final_models, X_test_final, final_best_iterations, final_model_types, weights=weights
)

submission = pd.DataFrame(
    {"id": range(len(test_pred_proba)), "Activity": test_pred_proba}
)

submission.to_csv("./submission/submission.csv", index=False)
print(f"  Submission saved: {submission.shape}")
print(f"  Sample:\n{submission.head()}")

# ============================================================
# 8. FINAL SCORE
# ============================================================
score = val_score
print(f"Final Validation Score: {score}")