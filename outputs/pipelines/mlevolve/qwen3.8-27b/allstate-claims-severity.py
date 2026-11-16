import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
import json
import os

# Create directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(
    f"Target stats: mean={train_df['loss'].mean():.2f}, median={train_df['loss'].median():.2f}, max={train_df['loss'].max():.2f}"
)

# ============================================================
# 2. IDENTIFY FEATURE TYPES
# ============================================================
target_col = "loss"
id_col = "id"

train_features = train_df.drop(columns=[target_col, id_col], errors="ignore")
test_features = test_df.drop(columns=[target_col, id_col], errors="ignore")

cat_cols = train_features.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = train_features.select_dtypes(
    include=["int64", "int32", "float64", "float32"]
).columns.tolist()

print(f"Categorical features ({len(cat_cols)}): {cat_cols[:10]}...")
print(f"Numerical features ({len(num_cols)}): {num_cols[:10]}...")

# ============================================================
# 3. SPLIT TRAIN INTO TRAIN/VALIDATION (BEFORE any processing)
# ============================================================
train_idx, val_idx = train_test_split(
    range(len(train_features)), test_size=0.15, random_state=42
)

train_split = train_features.iloc[train_idx].reset_index(drop=True)
val_split = train_features.iloc[val_idx].reset_index(drop=True)

y_train_raw = train_df[target_col].iloc[train_idx].values
y_val_raw = train_df[target_col].iloc[val_idx].values

print(f"Train split: {len(train_split)}, Val split: {len(val_split)}")

# ============================================================
# 4. FEATURE ENGINEERING (fit on train only, transform all)
# ============================================================


def engineer_features(
    df, cat_cols, num_cols, freq_maps=None, cardinality=None, is_fit=False
):
    """Apply feature engineering. If is_fit=True, compute stats from this df."""
    df = df.copy()

    # --- Frequency Encoding (fit on train, apply to all) ---
    if is_fit:
        freq_maps = {}
        for col in cat_cols:
            freq = df[col].value_counts(normalize=True)
            freq_maps[col] = freq
    if freq_maps is not None:
        for col in cat_cols:
            if col in freq_maps:
                df[f"{col}_freq"] = df[col].map(freq_maps[col]).fillna(0)

    # --- Categorical cardinality features ---
    if is_fit:
        cardinality = {}
        for col in cat_cols:
            cardinality[col] = df[col].nunique()
    if cardinality is not None:
        for col in cat_cols:
            df[f"{col}_nunique"] = cardinality[col]

    # --- Missing value indicators ---
    for col in cat_cols + num_cols:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isnull().astype(int)

    # --- Numerical feature engineering ---
    for col in num_cols:
        if col in df.columns and df[col].min() > 0:
            df[f"{col}_log"] = np.log1p(df[col])

    # --- Interaction features between top categorical pairs ---
    if len(cat_cols) >= 2:
        top_cats = cat_cols[:7]
        for i in range(len(top_cats)):
            for j in range(i + 1, len(top_cats)):
                col_a, col_b = top_cats[i], top_cats[j]
                if col_a in df.columns and col_b in df.columns:
                    df[f"{col_a}_x_{col_b}"] = (
                        df[col_a].astype(str) + "_" + df[col_b].astype(str)
                    )

    # --- Aggregate statistics for numerical features ---
    if num_cols:
        num_subset = df[num_cols].fillna(0)
        df["num_mean"] = num_subset.mean(axis=1)
        df["num_std"] = num_subset.std(axis=1)
        df["num_max"] = num_subset.max(axis=1)
        df["num_min"] = num_subset.min(axis=1)
        df["num_range"] = df["num_max"] - df["num_min"]
        df["num_skew"] = num_subset.skew(axis=1)

    return df, freq_maps, cardinality


# Fit on train split
train_split, freq_maps, cardinality = engineer_features(
    train_split, cat_cols, num_cols, is_fit=True
)

# Transform validation and test (using stats from train)
val_split, _, _ = engineer_features(
    val_split,
    cat_cols,
    num_cols,
    freq_maps=freq_maps,
    cardinality=cardinality,
    is_fit=False,
)
test_features, _, _ = engineer_features(
    test_features,
    cat_cols,
    num_cols,
    freq_maps=freq_maps,
    cardinality=cardinality,
    is_fit=False,
)

# ============================================================
# 5. IDENTIFY FINAL CATEGORICAL COLUMNS (for LightGBM native handling)
# ============================================================
final_cat_cols = cat_cols + [
    c for c in train_split.columns if "x_" in c and c not in cat_cols
]
final_cat_cols = [c for c in final_cat_cols if c in train_split.columns]

# Convert final categorical columns to category dtype
for col in final_cat_cols:
    train_split[col] = train_split[col].astype("category")
    val_split[col] = val_split[col].astype("category")
    test_features[col] = test_features[col].astype("category")

# Ensure categories are aligned (use train categories)
for col in final_cat_cols:
    train_cats = train_split[col].cat.categories
    val_split[col] = pd.Categorical(val_split[col], categories=train_cats)
    test_features[col] = pd.Categorical(test_features[col], categories=train_cats)

# ============================================================
# 6. TARGET TRANSFORMATION (log1p for skewed distribution)
# ============================================================
y_train = np.log1p(y_train_raw)
y_val = np.log1p(y_val_raw)

print(f"Target (log1p): train mean={y_train.mean():.4f}, val mean={y_val.mean():.4f}")
print(f"Train features: {train_split.shape}")
print(f"Val features:   {val_split.shape}")
print(f"Test features:  {test_features.shape}")
print(f"Categorical cols: {len(final_cat_cols)}")
print(f"Total feature columns: {train_split.shape[1]}")

# ============================================================
# 7. MODEL DESIGN: 4-Model Ensemble for MAE Optimization
# ============================================================
# Ensemble of diverse models to reduce variance and improve generalization.
# All models use log1p target transformation.
# The ensemble averages predictions in log-space before inverse transformation.
# Model diversity: 2 LightGBM (quantile) + 1 XGBoost (reg:squarederror) + 1 LightGBM (huber)

model_configs = [
    # Model 1: LightGBM quantile alpha=0.5, moderate complexity
    {
        "type": "lgb",
        "params": {
            "objective": "quantile",
            "alpha": 0.5,
            "metric": "mae",
            "learning_rate": 0.03,
            "num_leaves": 255,
            "max_depth": -1,
            "min_child_samples": 20,
            "min_child_weight": 1e-3,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "n_estimators": 5000,
            "verbose": -1,
            "random_state": 42,
            "n_jobs": -1,
            "cat_smooth": 10.0,
        },
    },
    # Model 2: LightGBM quantile alpha=0.5, high complexity for more interactions
    {
        "type": "lgb",
        "params": {
            "objective": "quantile",
            "alpha": 0.5,
            "metric": "mae",
            "learning_rate": 0.02,
            "num_leaves": 511,
            "max_depth": -1,
            "min_child_samples": 20,
            "min_child_weight": 1e-3,
            "feature_fraction": 0.5,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "n_estimators": 5000,
            "verbose": -1,
            "random_state": 42,
            "n_jobs": -1,
            "cat_smooth": 10.0,
        },
    },
    # Model 3: XGBoost reg:squarederror on log1p target (different tree-growing and regularization)
    {
        "type": "xgb",
        "params": {
            "objective": "reg:squarederror",
            "learning_rate": 0.02,
            "max_depth": 8,
            "colsample_bytree": 0.6,
            "subsample": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "n_estimators": 5000,
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
            "enable_categorical": True,
            "early_stopping_rounds": 150,
        },
    },
    # Model 4: LightGBM MAE (huber) objective, lower complexity for different bias-variance tradeoff
    {
        "type": "lgb",
        "params": {
            "objective": "huber",
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "min_child_weight": 1e-3,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "n_estimators": 5000,
            "verbose": -1,
            "random_state": 42,
            "n_jobs": -1,
            "cat_smooth": 10.0,
        },
    },
]

# ============================================================
# 8. TRAIN 4-MODEL ENSEMBLE WITH EARLY STOPPING
# ============================================================
print("\n--- Training 4-Model Ensemble ---")

models = []
val_preds_log = []
test_preds_log = []

for i, config in enumerate(model_configs):
    mtype = config["type"]
    params = config["params"]

    if mtype == "lgb":
        print(f"\n  Model {i+1} [LightGBM]: objective={params['objective']}, num_leaves={params['num_leaves']}, lr={params['learning_rate']}")

        model = lgb.LGBMRegressor(**params)

        model.fit(
            train_split,
            y_train,
            eval_set=[(val_split, y_val)],
            categorical_feature=final_cat_cols,
            callbacks=[
                lgb.early_stopping(stopping_rounds=150, verbose=False),
                lgb.log_evaluation(period=500),
            ],
        )

        print(f"  Best iteration: {model.best_iteration_}")

        # Validation prediction in log-space
        val_pred_log = model.predict(val_split)
        # Test prediction in log-space
        test_pred_log = model.predict(test_features)

    else:
        print(f"\n  Model {i+1} [XGBoost]: objective={params['objective']}, max_depth={params['max_depth']}, lr={params['learning_rate']}")

        model = xgb.XGBRegressor(**params)

        model.fit(
            train_split,
            y_train,
            eval_set=[(val_split, y_val)],
            verbose=500,
        )

        print(f"  Best iteration: {model.best_iteration}")
        val_pred_log = model.predict(val_split, iteration_range=(0, model.best_iteration + 1))
        test_pred_log = model.predict(test_features, iteration_range=(0, model.best_iteration + 1))

        # Validation prediction in log-space
    val_preds_log.append(val_pred_log)

    # Individual model MAE in original space
    val_pred_orig = np.expm1(val_pred_log)
    val_pred_orig = np.maximum(val_pred_orig, 0)
    val_mae_i = np.mean(np.abs(y_val_raw - val_pred_orig))
    print(f"  Individual Model {i+1} Validation MAE: {val_mae_i:.4f}")

    test_preds_log.append(test_pred_log)
    models.append(model)

# ============================================================
# 9. ENSEMBLE VALIDATION PREDICTIONS (original space)
# ============================================================
# Average predictions in log-space, then inverse transform
val_pred_log_ensemble = np.mean(val_preds_log, axis=0)
val_pred = np.expm1(val_pred_log_ensemble)
val_pred = np.maximum(val_pred, 0)

# Compute ensemble MAE in original space (official metric)
val_mae = np.mean(np.abs(y_val_raw - val_pred))
print(f"\n--- Ensemble Results ---")
print(f"Ensemble Validation MAE (original space): {val_mae:.4f}")

# ============================================================
# 10. ENSEMBLE TEST PREDICTIONS (identical processing to validation)
# ============================================================
test_pred_log_ensemble = np.mean(test_preds_log, axis=0)
test_pred = np.expm1(test_pred_log_ensemble)
test_pred = np.maximum(test_pred, 0)

# ============================================================
# 11. SAVE SUBMISSION
# ============================================================
test_ids = (
    test_df[id_col].values
    if id_col in test_df.columns
    else np.arange(len(test_features))
)

submission = pd.DataFrame({"id": test_ids.astype(int), "loss": test_pred})
submission.to_csv("./submission/submission.csv", index=False)

print(f"\nSubmission saved: {submission.shape}")
print(
    f"Prediction stats: mean={test_pred.mean():.2f}, median={np.median(test_pred):.2f}, "
    f"min={test_pred.min():.2f}, max={test_pred.max():.2f}"
)

# ============================================================
# 12. SAVE MODELS
# ============================================================
for i, model in enumerate(models):
    if isinstance(model, lgb.LGBMRegressor):
        model.booster_.save_model(f"./working/model_ensemble_{i+1}.txt")
        print(f"Saved model {i+1} to ./working/model_ensemble_{i+1}.txt")
    else:
        model.save_model(f"./working/model_ensemble_{i+1}.json")
        print(f"Saved model {i+1} to ./working/model_ensemble_{i+1}.json")

# ============================================================
# 13. FINAL SCORE
# ============================================================
score = val_mae
print(f"Final Validation Score: {score:.4f}")
