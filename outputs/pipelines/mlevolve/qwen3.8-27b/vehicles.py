import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import os
import json
import warnings

warnings.filterwarnings("ignore")

# Create output directory
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# ============================================================
# 2. IDENTIFY FEATURE TYPES & SPLIT
# ============================================================
target_col = "youSaveSpend"
id_col = "id"

y_train_full = train[target_col].values
train_ids = train[id_col].values if id_col in train.columns else np.arange(len(train))
test_ids = test[id_col].values if id_col in test.columns else np.arange(len(test))

train_features = train.drop(
    columns=[c for c in [target_col, id_col] if c in train.columns], errors="ignore"
)
test_features = test.drop(
    columns=[c for c in [id_col] if c in test.columns], errors="ignore"
)

# Identify feature types
numeric_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train_features.select_dtypes(
    include=["object", "category"]
).columns.tolist()

# Low-cardinality numerics → categorical
for col in numeric_cols[:]:
    if train_features[col].nunique() < 15:
        categorical_cols.append(col)
        numeric_cols.remove(col)

print(f"Numeric features: {numeric_cols}")
print(f"Categorical features: {categorical_cols}")

# ============================================================
# 3. TRAIN/VALIDATION SPLIT
# ============================================================
try:
    target_quantiles = pd.qcut(y_train_full, q=5, labels=False, duplicates="drop")
    (
        train_feat_split,
        val_feat_split,
        y_train_split,
        y_val_split,
        _,
        _,
    ) = train_test_split(
        train_features,
        y_train_full,
        target_quantiles,
        test_size=0.15,
        random_state=42,
        stratify=target_quantiles,
    )
except Exception:
    train_feat_split, val_feat_split, y_train_split, y_val_split = train_test_split(
        train_features, y_train_full, test_size=0.15, random_state=42
    )

print(f"Train split: {train_feat_split.shape}, Val split: {val_feat_split.shape}")

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================


def engineer_features(
    df, fit_df=None, y_fit=None, is_fitting=True, encoders=None, freq_maps=None
):
    """Apply feature engineering. Fit on train, transform on val/test."""
    df = df.copy()

    # Handle missing values
    for col in df.columns:
        if df[col].isnull().any():
            if col in df.select_dtypes(include=[np.number]).columns:
                fill_val = (
                    fit_df[col].median() if fit_df is not None else df[col].median()
                )
                df[col] = df[col].fillna(fill_val)
            else:
                df[col] = df[col].fillna("MISSING")

    # Categorical encoding
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].astype(str)

        if is_fitting:
            # Fit target encoding
            global_mean = y_fit.mean()
            agg = pd.DataFrame({"cat": df[col], "target": y_fit})
            stats = agg.groupby("cat")["target"].agg(["mean", "count"])
            smoothing = 10
            stats["smoothed"] = (
                stats["count"] * stats["mean"] + smoothing * global_mean
            ) / (stats["count"] + smoothing)
            encoders[col] = (stats["smoothed"], global_mean)

            # Fit frequency encoding
            freq_maps[col] = df[col].value_counts(normalize=True)

        # Apply target encoding
        if col in encoders:
            smoothed_map, global_mean = encoders[col]
            df[col + "_te"] = df[col].map(smoothed_map).fillna(global_mean)
        else:
            df[col + "_te"] = 0.0

        # Apply frequency encoding
        if col in freq_maps:
            df[col + "_freq"] = df[col].map(freq_maps[col]).fillna(0)
        else:
            df[col + "_freq"] = 0.0

        # One-hot for low cardinality, drop original for high
        if df[col].nunique() <= 10:
            dummies = pd.get_dummies(df[col], prefix=col, dtype=float)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        else:
            df = df.drop(columns=[col])

    # Numerical feature engineering
    num_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns.tolist()
        if not c.endswith("_te") and not c.endswith("_freq") and not c.startswith("id")
    ]

    # Log transform for skewed features
    for col in num_cols:
        vals = df[col].dropna()
        if len(vals) > 0 and vals.min() > 0 and vals.skew() > 1:
            df[col + "_log"] = np.log1p(df[col])
        elif len(vals) > 0 and vals.min() < 0 and vals.skew() < -1:
            df[col + "_log"] = -np.log1p(-df[col])

    # Determine top numeric features for interactions
    if is_fitting:
        corr_with_target = (
            df[num_cols].corrwith(y_fit).abs().sort_values(ascending=False)
        )
        top_numeric = corr_with_target[corr_with_target > 0.1].index.tolist()[:8]
    else:
        top_numeric = num_cols[:8]

    # Pairwise interactions
    if len(top_numeric) >= 2:
        for i in range(min(len(top_numeric), 6)):
            for j in range(i + 1, min(len(top_numeric), 6)):
                c1, c2 = top_numeric[i], top_numeric[j]
                if c1 in df.columns and c2 in df.columns:
                    df[f"{c1}_x_{c2}"] = df[c1] * df[c2]
                    df[f"{c1}_div_{c2}"] = df[c1] / (df[c2] + 1e-8)

    # Polynomial features
    if len(top_numeric) >= 1:
        for col in top_numeric[:3]:
            if col in df.columns:
                df[col + "_sq"] = df[col] ** 2

    # Ratio features
    if len(num_cols) >= 2:
        if is_fitting:
            corr = df[num_cols].corrwith(y_fit).abs().sort_values(ascending=False)
            top_for_ratio = corr.index[:5].tolist()
        else:
            top_for_ratio = num_cols[:5]
        for i in range(len(top_for_ratio)):
            for j in range(i + 1, len(top_for_ratio)):
                c1, c2 = top_for_ratio[i], top_for_ratio[j]
                if c1 in df.columns and c2 in df.columns:
                    df[f"{c1}_ratio_{c2}"] = df[c1] / (df[c2].abs() + 1e-8)

    return df


# Fit on train split
encoders = {}
freq_maps = {}
train_processed = engineer_features(
    train_feat_split,
    fit_df=train_feat_split,
    y_fit=pd.Series(y_train_split),
    is_fitting=True,
    encoders=encoders,
    freq_maps=freq_maps,
)

# Transform val
val_processed = engineer_features(
    val_feat_split,
    fit_df=train_feat_split,
    y_fit=None,
    is_fitting=False,
    encoders=encoders,
    freq_maps=freq_maps,
)

# Transform test
test_processed = engineer_features(
    test_features,
    fit_df=train_feat_split,
    y_fit=None,
    is_fitting=False,
    encoders=encoders,
    freq_maps=freq_maps,
)

# Align columns
train_cols = train_processed.columns.tolist()
for col in train_cols:
    if col not in val_processed.columns:
        val_processed[col] = 0.0
    if col not in test_processed.columns:
        test_processed[col] = 0.0

val_processed = val_processed[train_cols]
test_processed = test_processed[train_cols]

print(f"Processed train: {train_processed.shape}")
print(f"Processed val: {val_processed.shape}")
print(f"Processed test: {test_processed.shape}")

# ============================================================
# 5. MODEL DESIGN & TRAINING (ENSEMBLE)
# ============================================================
X_train = train_processed[train_cols].values
X_val = val_processed[train_cols].values
X_test = test_processed[train_cols].values

# --- Model 1: LightGBM (reduced complexity) ---
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 10,
    "min_child_weight": 1.0,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# --- Model 2: XGBoost ---
xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 4,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "min_child_weight": 1.0,
    "random_state": 42,
    "nthread": -1,
    "verbosity": 0,
}

# --- Model 3: CatBoost ---
cat_params = {
    "iterations": 3000,
    "learning_rate": 0.01,
    "depth": 4,
    "l2_leaf_reg": 10,
    "random_seed": 42,
    "verbose": 0,
}

# Train LightGBM
print("\nTraining LightGBM with early stopping...")
train_data_lgb = lgb.Dataset(X_train, label=y_train_split)
val_data_lgb = lgb.Dataset(X_val, label=y_val_split, reference=train_data_lgb)

lgb_model = lgb.train(
    lgb_params,
    train_data_lgb,
    num_boost_round=5000,
    valid_sets=[val_data_lgb],
    valid_names=["valid"],
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
)

lgb_best_iter = lgb_model.best_iteration
lgb_val_pred = lgb_model.predict(X_val, num_iteration=lgb_best_iter)
lgb_val_rmse = np.sqrt(np.mean((y_val_split - lgb_val_pred) ** 2))
print(f"LightGBM best iteration: {lgb_best_iter}, Val RMSE: {lgb_val_rmse:.6f}")

# Train XGBoost
print("\nTraining XGBoost with early stopping...")
xgb_train = xgb.DMatrix(X_train, label=y_train_split)
xgb_val = xgb.DMatrix(X_val, label=y_val_split)

xgb_model = xgb.train(
    xgb_params,
    xgb_train,
    num_boost_round=5000,
    evals=[(xgb_val, "valid")],
    early_stopping_rounds=100,
    verbose_eval=False,
)

xgb_best_iter = xgb_model.best_iteration
xgb_val_pred = xgb_model.predict(xgb.DMatrix(X_val), iteration_range=(0, xgb_best_iter + 1))
xgb_val_rmse = np.sqrt(np.mean((y_val_split - xgb_val_pred) ** 2))
print(f"XGBoost best iteration: {xgb_best_iter}, Val RMSE: {xgb_val_rmse:.6f}")

# Train CatBoost
print("\nTraining CatBoost with early stopping...")
cat_model = CatBoostRegressor(**cat_params)
cat_model.fit(
    X_train,
    y_train_split,
    eval_set=(X_val, y_val_split),
    early_stopping_rounds=100,
    verbose=0,
)

cat_best_iter = cat_model.get_best_iteration()
cat_val_pred = cat_model.predict(X_val)
cat_val_rmse = np.sqrt(np.mean((y_val_split - cat_val_pred) ** 2))
print(f"CatBoost best iteration: {cat_best_iter}, Val RMSE: {cat_val_rmse:.6f}")

# ============================================================
# 6. ENSEMBLE PREDICTION (inverse-RMSE weighted)
# ============================================================
# Compute inverse-RMSE weights
rmse_values = np.array([lgb_val_rmse, xgb_val_rmse, cat_val_rmse])
weights = (1.0 / rmse_values) / np.sum(1.0 / rmse_values)
print(f"\nEnsemble weights: LGB={weights[0]:.4f}, XGB={weights[1]:.4f}, CAT={weights[2]:.4f}")

# Ensemble validation prediction
ensemble_val_pred = (
    weights[0] * lgb_val_pred
    + weights[1] * xgb_val_pred
    + weights[2] * cat_val_pred
)
ensemble_val_rmse = np.sqrt(np.mean((y_val_split - ensemble_val_pred) ** 2))
print(f"Ensemble Validation RMSE: {ensemble_val_rmse:.6f}")

# ============================================================
# 7. RETRAIN ON FULL DATA & PREDICT TEST
# ============================================================
X_full = np.vstack([X_train, X_val])
y_full = np.concatenate([y_train_split, y_val_split])

# Retrain LightGBM on full data
print(f"\nRetraining LightGBM on full data ({len(X_full)} samples) with {lgb_best_iter} iterations...")
full_data_lgb = lgb.Dataset(X_full, label=y_full)
final_lgb_model = lgb.train(lgb_params, full_data_lgb, num_boost_round=lgb_best_iter)
lgb_test_pred = final_lgb_model.predict(X_test)

# Retrain XGBoost on full data
print(f"Retraining XGBoost on full data with {xgb_best_iter} iterations...")
xgb_full = xgb.DMatrix(X_full, label=y_full)
final_xgb_model = xgb.train(
    xgb_params,
    xgb_full,
    num_boost_round=xgb_best_iter,
    verbose_eval=False,
)
xgb_test_pred = final_xgb_model.predict(xgb.DMatrix(X_test))

# Retrain CatBoost on full data
print(f"Retraining CatBoost on full data with {cat_best_iter} iterations...")
final_cat_model = CatBoostRegressor(
    iterations=cat_best_iter,
    learning_rate=0.01,
    depth=4,
    l2_leaf_reg=10,
    random_seed=42,
    verbose=0,
)
final_cat_model.fit(X_full, y_full, verbose=0)
cat_test_pred = final_cat_model.predict(X_test)

# Ensemble test prediction
test_pred = (
    weights[0] * lgb_test_pred
    + weights[1] * xgb_test_pred
    + weights[2] * cat_test_pred
)

# Clip to reasonable range
test_pred = np.clip(
    test_pred,
    y_full.min() - 0.5 * (y_full.max() - y_full.min()),
    y_full.max() + 0.5 * (y_full.max() - y_full.min()),
)

print(f"Test predictions range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

# ============================================================
# 8. SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame({"id": test_ids.astype(int), "youSaveSpend": test_pred})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")

# Feature importance
importance = lgb_model.feature_importance(importance_type="gain")
feat_imp = pd.DataFrame({"feature": train_cols, "importance": importance}).sort_values(
    "importance", ascending=False
)
print(f"\nTop 10 features by gain (LightGBM):")
print(feat_imp.head(10).to_string(index=False))

# ============================================================
# 9. FINAL SCORE
# ============================================================
print(f"\n--- Summary ---")
print(f"Split: 85/15 stratified by target quantiles")
print(f"Metric: RMSE (official competition metric)")
print(f"Ensemble: LightGBM + XGBoost + CatBoost (inverse-RMSE weighted)")
print(f"  LightGBM: best_iter={lgb_best_iter}, RMSE={lgb_val_rmse:.6f}, weight={weights[0]:.4f}")
print(f"  XGBoost:  best_iter={xgb_best_iter}, RMSE={xgb_val_rmse:.6f}, weight={weights[1]:.4f}")
print(f"  CatBoost: best_iter={cat_best_iter}, RMSE={cat_val_rmse:.6f}, weight={weights[2]:.4f}")
print(f"Ensemble Validation RMSE: {ensemble_val_rmse:.6f}")
print(f"Final Validation Score: {ensemble_val_rmse:.6f}")
