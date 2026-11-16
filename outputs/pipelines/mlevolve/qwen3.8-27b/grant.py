import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import os
import warnings

warnings.filterwarnings("ignore")
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============================================================
# DATA LOADING
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

target_col = "Federal Funds"
y_train_full = train[target_col].values
train = train.drop(columns=[target_col])

test_ids = test.index.values if "id" not in test.columns else test["id"].values
if "id" in test.columns:
    test = test.drop(columns=["id"])
if "id" in train.columns:
    train = train.drop(columns=["id"])

# ============================================================
# FEATURE ENGINEERING
# ============================================================
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()
bool_cols = [c for c in train.columns if train[c].dtype == "bool"]

n_train = len(train)
combined = pd.concat([train, test], axis=0, ignore_index=True)

# Boolean -> numeric
for col in bool_cols:
    combined[col] = combined[col].astype(int)

# Categorical encoding (frequency + label)
for col in categorical_cols:
    train_part = combined.loc[: n_train - 1, col]
    freq_map = train_part.value_counts(normalize=True).to_dict()
    combined[col + "_freq"] = combined[col].map(freq_map).fillna(0)

    le = LabelEncoder()
    combined.loc[: n_train - 1, col + "_label"] = le.fit_transform(train_part.astype(str)).astype(float)
    test_part = combined.loc[n_train:, col]
    known_classes = set(train_part.astype(str).unique())
    test_encoded = test_part.astype(str).map(
        {c: i for i, c in enumerate(sorted(known_classes))}
    )
    combined.loc[n_train:, col + "_label"] = test_encoded.fillna(-1).values
    combined = combined.drop(columns=[col])

# Fill missing numeric values (fit on train only)
numeric_cols_updated = combined.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols_updated:
    train_vals = combined.loc[: n_train - 1, col]
    med = train_vals.median()
    combined[col] = combined[col].fillna(med)

# Identify base numeric columns (exclude derived)
key_numeric = [
    c
    for c in numeric_cols_updated
    if c not in [c + "_freq" for c in categorical_cols]
    and c not in [c + "_label" for c in categorical_cols]
]

# Log transforms for positive skewed features
for col in numeric_cols_updated:
    train_vals = combined.loc[: n_train - 1, col]
    if (train_vals > 0).sum() > 0.9 * len(train_vals) and train_vals.max() > 100:
        combined[col + "_log"] = np.log1p(combined[col].clip(lower=0))

# Percentile rank features (fit on train only to avoid leakage)
for col in key_numeric[:30]:
    train_vals = combined.loc[: n_train - 1, col]
    if train_vals.nunique() > 5:
        train_rank = train_vals.rank(pct=True)
        # Map train values to their percentile ranks, apply to all
        rank_map = dict(zip(train_vals.values, train_rank.values))
        combined[col + "_pct"] = combined[col].map(rank_map)
        # For unseen values (in test), clip to [0, 1] range based on train min/max
        train_min, train_max = train_vals.min(), train_vals.max()
        mask = combined[col + "_pct"].isna()
        if mask.any():
            clipped = combined.loc[mask, col].clip(lower=train_min, upper=train_max)
            combined.loc[mask, col + "_pct"] = (clipped - train_min) / (train_max - train_min + 1e-8)

# Interaction features between top-variance features
variances = (
    combined.loc[: n_train - 1, key_numeric[:30]].var().sort_values(ascending=False)
)
top_features = variances.index[:8].tolist()

for i in range(len(top_features)):
    for j in range(i + 1, len(top_features)):
        f1, f2 = top_features[i], top_features[j]
        combined[f1 + "_" + f2 + "_ratio"] = combined[f1] / (combined[f2].abs() + 1e-8)
        combined[f1 + "_" + f2 + "_prod"] = combined[f1] * combined[f2]

# Domain-specific ratio features between top-variance key numeric columns
# These capture multiplicative relationships between funding components
top_ratio_features = variances.index[:10].tolist()
for i in range(len(top_ratio_features)):
    for j in range(i + 1, len(top_ratio_features)):
        f1, f2 = top_ratio_features[i], top_ratio_features[j]
        combined[f1 + "_div_" + f2] = combined[f1] / (combined[f2].abs() + 1e-8)

# Polynomial features (degree 2) for top 5 most important numeric features
top5_features = key_numeric[:5]
for col in top5_features:
    combined[col + "_poly2"] = combined[col] ** 2
    combined[col + "_poly2_scaled"] = (combined[col] - combined.loc[: n_train - 1, col].mean()) ** 2

# Squared features
for col in key_numeric[:15]:
    combined[col + "_sq"] = combined[col] ** 2

# Handle infinities
combined = combined.replace([np.inf, -np.inf], np.nan)
for col in combined.select_dtypes(include=[np.number]).columns:
    if combined[col].isnull().sum() > 0:
        train_vals = combined.loc[: n_train - 1, col]
        combined[col] = combined[col].fillna(train_vals.median())

# ============================================================
# SPLIT AND SCALE
# ============================================================
train_feat = combined.loc[: n_train - 1].reset_index(drop=True)
test_feat = combined.loc[n_train:].reset_index(drop=True)

y_full = pd.Series(y_train_full, index=train_feat.index)
train_final, val_final, y_train_split, y_val_split = train_test_split(
    train_feat, y_full, test_size=0.15, random_state=42
)
y_train_split = y_train_split.reset_index(drop=True).values
y_val_split = y_val_split.reset_index(drop=True).values
train_final = train_final.reset_index(drop=True)
val_final = val_final.reset_index(drop=True)

print(f"Final train shape: {train_final.shape}")
print(f"Final val shape: {val_final.shape}")
print(f"Final test shape: {test_feat.shape}")

scaler = StandardScaler()
scaler.fit(train_final)
train_X = pd.DataFrame(
    scaler.transform(train_final),
    columns=train_final.columns,
    index=train_final.index,
)
val_X = pd.DataFrame(
    scaler.transform(val_final), columns=val_final.columns, index=val_final.index
)
test_X = pd.DataFrame(
    scaler.transform(test_feat),
    columns=test_feat.columns,
    index=test_feat.index,
)

feature_names = train_final.columns.tolist()

# Baseline
mean_pred = np.full_like(y_val_split, np.mean(y_train_split), dtype=np.float64)
baseline_rmse = np.sqrt(mean_squared_error(y_val_split, mean_pred))
print(f"Baseline RMSE (mean prediction): {baseline_rmse:.4f}")
print(
    f"Target - mean: {y_train_split.mean():.2f}, std: {y_train_split.std():.2f}, min: {y_train_split.min():.2f}, max: {y_train_split.max():.2f}"
)

# ============================================================
# MODEL 1: Original-scale LightGBM
# ============================================================
y_train_log = np.log1p(y_train_split)
y_val_log = np.log1p(y_val_split)

params_orig = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "min_child_weight": 1e-3,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 5000,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}

train_ds_orig = lgb.Dataset(
    train_X.values, label=y_train_split, feature_name=feature_names
)
val_ds_orig = lgb.Dataset(
    val_X.values, label=y_val_split, feature_name=feature_names, reference=train_ds_orig
)

model_orig = lgb.train(
    params_orig,
    train_ds_orig,
    valid_sets=[val_ds_orig],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=500),
    ],
)

val_pred_orig = model_orig.predict(
    val_X.values, num_iteration=model_orig.best_iteration
)
val_pred_orig = np.maximum(val_pred_orig, 0)
rmse_orig_model = np.sqrt(mean_squared_error(y_val_split, val_pred_orig))
print(
    f"[LGBM-Orig] Best iter: {model_orig.best_iteration}, Val RMSE: {rmse_orig_model:.4f}"
)

# ============================================================
# MODEL 2: LightGBM with smaller trees and higher regularization
# ============================================================
params_lgbm_small = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.02,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 30,
    "min_child_weight": 5e-3,
    "subsample": 0.7,
    "subsample_freq": 1,
    "colsample_bytree": 0.5,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "n_estimators": 5000,
    "verbose": -1,
    "random_state": 7,
    "n_jobs": -1,
}

train_ds_small = lgb.Dataset(
    train_X.values, label=y_train_split, feature_name=feature_names
)
val_ds_small = lgb.Dataset(
    val_X.values, label=y_val_split, feature_name=feature_names, reference=train_ds_small
)

model_lgbm_small = lgb.train(
    params_lgbm_small,
    train_ds_small,
    valid_sets=[val_ds_small],
    callbacks=[
        lgb.early_stopping(stopping_rounds=150, verbose=False),
        lgb.log_evaluation(period=500),
    ],
)

val_pred_lgbm_small = model_lgbm_small.predict(
    val_X.values, num_iteration=model_lgbm_small.best_iteration
)
val_pred_lgbm_small = np.maximum(val_pred_lgbm_small, 0)
rmse_lgbm_small = np.sqrt(mean_squared_error(y_val_split, val_pred_lgbm_small))
print(
    f"[LGBM-Small] Best iter: {model_lgbm_small.best_iteration}, Val RMSE: {rmse_lgbm_small:.4f}"
)

# ============================================================
# MODEL 3: XGBoost
# ============================================================
dtrain_xgb = xgb.DMatrix(train_X.values, label=y_train_split, feature_names=feature_names)
dval_xgb = xgb.DMatrix(val_X.values, label=y_val_split, feature_names=feature_names)

params_xgb = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "random_state": 42,
    "nthread": -1,
    "verbosity": 0,
}

model_xgb = xgb.train(
    params_xgb,
    dtrain_xgb,
    num_boost_round=5000,
    evals=[(dval_xgb, "val")],
    early_stopping_rounds=100,
    verbose_eval=500,
)

val_pred_xgb = model_xgb.predict(
    xgb.DMatrix(val_X.values, feature_names=feature_names),
    iteration_range=(0, model_xgb.best_iteration + 1),
)
val_pred_xgb = np.maximum(val_pred_xgb, 0)
rmse_xgb = np.sqrt(mean_squared_error(y_val_split, val_pred_xgb))
print(f"[XGBoost] Best iter: {model_xgb.best_iteration}, Val RMSE: {rmse_xgb:.4f}")

# ============================================================
# WEIGHTED ENSEMBLE BLEND
# ============================================================
# Weights inversely proportional to each model's validation RMSE
rmse_list = [rmse_orig_model, rmse_lgbm_small, rmse_xgb]
inv_rmse = [1.0 / r for r in rmse_list]
weight_sum = sum(inv_rmse)
weights = [w / weight_sum for w in inv_rmse]

print(f"\nEnsemble weights: LGBM-Orig={weights[0]:.4f}, LGBM-Small={weights[1]:.4f}, XGBoost={weights[2]:.4f}")

# Blend validation predictions
val_pred_ensemble = (
    weights[0] * val_pred_orig
    + weights[1] * val_pred_lgbm_small
    + weights[2] * val_pred_xgb
)
rmse_ensemble = np.sqrt(mean_squared_error(y_val_split, val_pred_ensemble))
print(f"[Ensemble] Val RMSE: {rmse_ensemble:.4f}")

# Also check individual models vs ensemble
print(f"\n--- Model Comparison ---")
print(f"  LGBM-Orig:   {rmse_orig_model:.4f}")
print(f"  LGBM-Small:  {rmse_lgbm_small:.4f}")
print(f"  XGBoost:     {rmse_xgb:.4f}")
print(f"  Ensemble:    {rmse_ensemble:.4f}")

# Use ensemble if it's better than best single model
best_single_rmse = min(rmse_orig_model, rmse_lgbm_small, rmse_xgb)
if rmse_ensemble <= best_single_rmse:
    best_rmse = rmse_ensemble
    use_ensemble = True
    print(f"\nUsing ENSEMBLE (RMSE={rmse_ensemble:.4f})")
else:
    best_rmse = best_single_rmse
    use_ensemble = False
    print(f"\nUsing best single model (RMSE={best_rmse:.4f})")

# ============================================================
# GENERATE TEST PREDICTIONS
# ============================================================
test_pred_orig = np.maximum(
    model_orig.predict(test_X.values, num_iteration=model_orig.best_iteration), 0
)
test_pred_lgbm_small = np.maximum(
    model_lgbm_small.predict(test_X.values, num_iteration=model_lgbm_small.best_iteration), 0
)
test_pred_xgb = np.maximum(
    model_xgb.predict(
        xgb.DMatrix(test_X.values, feature_names=feature_names),
        iteration_range=(0, model_xgb.best_iteration + 1),
    ),
    0,
)

if use_ensemble:
    test_pred = (
        weights[0] * test_pred_orig
        + weights[1] * test_pred_lgbm_small
        + weights[2] * test_pred_xgb
    )
else:
    # Use best single model
    if best_rmse == rmse_orig_model:
        test_pred = test_pred_orig
    elif best_rmse == rmse_lgbm_small:
        test_pred = test_pred_lgbm_small
    else:
        test_pred = test_pred_xgb

# ============================================================
# SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame({"id": range(len(test_pred)), "Federal Funds": test_pred})
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {len(submission)} predictions")
print(
    f"Test pred stats - mean: {test_pred.mean():.2f}, std: {test_pred.std():.2f}, min: {test_pred.min():.2f}, max: {test_pred.max():.2f}"
)

# ============================================================
# FINAL VALIDATION SCORE (official metric: RMSE on original scale)
# ============================================================
score = best_rmse
print(f"Final Validation Score: {score}")