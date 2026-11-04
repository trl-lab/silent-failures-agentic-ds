import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import os
import json
import warnings

warnings.filterwarnings("ignore")

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
print(f"Columns: {list(train_df.columns)}")
print(f"\nTarget stats:\n{train_df['Debt-to-Earnings Annual Rate'].describe()}")
print(f"\nNull counts (train):\n{train_df.isnull().sum()[train_df.isnull().sum() > 0]}")
print(f"\nDtypes:\n{train_df.dtypes}")

# ============================================================
# 2. SPLIT DATA FIRST (before any transformation)
# ============================================================
TARGET = "Debt-to-Earnings Annual Rate"

target = train_df[TARGET].copy()
train_features = train_df.drop(columns=[TARGET])

train_indices, val_indices = train_test_split(
    np.arange(len(train_features)),
    test_size=0.2,
    random_state=42,
    stratify=pd.qcut(target, q=10, duplicates="drop"),
)

X_train = train_features.iloc[train_indices].copy()
X_val = train_features.iloc[val_indices].copy()
y_train = target.iloc[train_indices].copy()
y_val = target.iloc[val_indices].copy()
X_test = test_df.copy()

print(f"\nSplit: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================


def engineer_features(
    df, is_training=False, scaler=None, encoders=None, col_stats=None
):
    """Apply feature engineering pipeline."""
    df = df.copy()

    # Separate column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    bool_cols = [c for c in df.columns if df[c].dtype == "bool"]

    # Convert bool to int
    for c in bool_cols:
        df[c] = df[c].astype(int)
        if c not in numeric_cols:
            numeric_cols.append(c)

    # --- Missing value indicators ---
    for c in numeric_cols:
        if df[c].isnull().sum() > 0:
            df[f"{c}_missing"] = df[c].isnull().astype(int)

    for c in categorical_cols:
        if df[c].isnull().sum() > 0:
            df[f"{c}_missing"] = df[c].isnull().astype(int)

    # --- Numeric feature engineering ---
    log_cols = []
    for c in numeric_cols:
        if df[c].isnull().sum() < len(df) * 0.5:
            non_null = df[c].dropna()
            if len(non_null) > 10:
                if (non_null > 0).sum() / len(non_null) > 0.8:
                    skew = non_null.skew()
                    if abs(skew) > 0.5:
                        log_cols.append(c)

    for c in log_cols:
        df[f"{c}_log"] = np.log1p(df[c].clip(lower=0))

    sqrt_cols = []
    for c in numeric_cols:
        if c not in log_cols:
            non_null = df[c].dropna()
            if len(non_null) > 10 and (non_null >= 0).sum() / len(non_null) > 0.9:
                if 0.2 < abs(non_null.skew()) <= 0.5:
                    sqrt_cols.append(c)

    for c in sqrt_cols:
        df[f"{c}_sqrt"] = np.sqrt(df[c].clip(lower=0))

    # --- Interaction features ---
    if is_training:
        interaction_pairs = []
        for i, c1 in enumerate(numeric_cols):
            for c2 in numeric_cols[i + 1 :]:
                if len(interaction_pairs) < 50:
                    interaction_pairs.append((c1, c2))
        col_stats = col_stats or {}
        col_stats["interaction_pairs"] = interaction_pairs
        col_stats["log_cols"] = log_cols
        col_stats["sqrt_cols"] = sqrt_cols
        col_stats["numeric_cols"] = numeric_cols
        col_stats["categorical_cols"] = categorical_cols
    else:
        interaction_pairs = col_stats.get("interaction_pairs", [])

    # Add interaction features (limited to top pairs)
    for c1, c2 in interaction_pairs[:30]:
        if c1 in df.columns and c2 in df.columns:
            df[f"{c1}_x_{c2}"] = df[c1].fillna(0) * df[c2].fillna(0)

    # Ratio features
    for c1, c2 in interaction_pairs[:30]:
        if c1 in df.columns and c2 in df.columns:
            denom = df[c2].replace(0, np.nan)
            df[f"{c1}_div_{c2}"] = (
                (df[c1].fillna(0) / denom).fillna(0).clip(-1000, 1000)
            )

    # --- Categorical encoding ---
    for c in categorical_cols:
        if is_training:
            freq = df[c].value_counts(normalize=True)
            encoders = encoders or {}
            encoders[c] = freq
            df[f"{c}_freq"] = df[c].map(freq).fillna(0)

            try:
                oe = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
                df[f"{c}_ord"] = oe.fit_transform(
                    df[c].fillna("__MISSING__").values.reshape(-1, 1)
                )
                encoders[f"{c}_oe"] = oe
            except:
                df[f"{c}_ord"] = 0
        else:
            freq = encoders.get(c, pd.Series())
            df[f"{c}_freq"] = df[c].map(freq).fillna(0)

            oe = encoders.get(f"{c}_oe")
            if oe is not None:
                df[f"{c}_ord"] = oe.transform(
                    df[c].fillna("__MISSING__").values.reshape(-1, 1)
                )
            else:
                df[f"{c}_ord"] = 0

    # --- Remove original categorical columns ---
    df = df.drop(columns=categorical_cols, errors="ignore")

    # --- Remove original bool columns ---
    df = df.drop(columns=bool_cols, errors="ignore")

    # --- Fill remaining NaNs in numeric columns ---
    numeric_cols_final = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols_final:
        if df[c].isnull().sum() > 0:
            if is_training:
                col_stats = col_stats or {}
                col_stats["medians"] = col_stats.get("medians", {})
                col_stats["medians"][c] = df[c].median()
                df[c] = df[c].fillna(col_stats["medians"][c])
            else:
                median_val = col_stats.get("medians", {}).get(c, 0)
                df[c] = df[c].fillna(median_val)

    # Handle infinities
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df, scaler, encoders, col_stats


# Apply feature engineering
X_train, _, encoders, col_stats = engineer_features(X_train, is_training=True)
X_val, _, _, _ = engineer_features(
    X_val, is_training=False, encoders=encoders, col_stats=col_stats
)
X_test, _, _, _ = engineer_features(
    X_test, is_training=False, encoders=encoders, col_stats=col_stats
)

# Ensure same columns
all_cols = X_train.columns.tolist()
for col in all_cols:
    if col not in X_val.columns:
        X_val[col] = 0
    if col not in X_test.columns:
        X_test[col] = 0

X_val = X_val[all_cols]
X_test = X_test[all_cols]

print(f"\nFinal feature count: {len(all_cols)}")
print(f"Train features shape: {X_train.shape}")
print(f"Val features shape: {X_val.shape}")
print(f"Test features shape: {X_test.shape}")

# ============================================================
# 4. SAVE PROCESSED DATA
# ============================================================
X_train.to_parquet("./working/X_train.parquet")
X_val.to_parquet("./working/X_val.parquet")
X_test.to_parquet("./working/X_test.parquet")
y_train.to_frame(TARGET).to_parquet("./working/y_train.parquet")
y_val.to_frame(TARGET).to_parquet("./working/y_val.parquet")

# ============================================================
# 5. MODEL DESIGN: LightGBM with Huber Loss
# ============================================================
# Rationale:
# - Huber loss: robust to outliers in financial ratios
# - Low learning rate + many rounds with early stopping: prevents overfitting
# - Conservative depth and min_child_samples: regularizes for small data

q1, q3 = np.percentile(y_train, [25, 75])
huber_delta = 1.5 * (q3 - q1)
print(f"\nHuber delta (1.5x IQR): {huber_delta:.4f}")

base_params = {
    "objective": "huber",
    "alpha": huber_delta,
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 20,
    "min_child_weight": 0.1,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_jobs": -1,
}

# ============================================================
# 6. TRAINING: Multi-seed LightGBM ensemble for robustness
# ============================================================
seeds = [42, 123, 456, 789, 1024]
models = []
val_preds_list = []
val_rmses = []

for i, seed in enumerate(seeds):
    params = base_params.copy()
    params["random_state"] = seed

    train_data = lgb.Dataset(X_train.values, label=y_train.values)
    val_data = lgb.Dataset(X_val.values, label=y_val.values, reference=train_data)

    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=5000,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    best_iter = model.best_iteration
    val_pred = model.predict(X_val.values, num_iteration=best_iter)
    val_rmse = np.sqrt(mean_squared_error(y_val.values, val_pred))

    models.append(model)
    val_preds_list.append(val_pred)
    val_rmses.append(val_rmse)

    print(f"  Seed {seed}: best_iter={best_iter}, val_rmse={val_rmse:.4f}")

# ============================================================
# 7. ENSEMBLE: Average predictions from all seeds
# ============================================================
val_ensemble_pred = np.mean(val_preds_list, axis=0)
val_ensemble_rmse = np.sqrt(mean_squared_error(y_val.values, val_ensemble_pred))
print(f"\nEnsemble validation RMSE: {val_ensemble_rmse:.4f}")
print(f"Individual RMSEs: {[f'{r:.4f}' for r in val_rmses]}")

# ============================================================
# 8. TEST INFERENCE (identical processing to validation)
# ============================================================
test_preds_list = []
for model in models:
    test_pred = model.predict(X_test.values, num_iteration=model.best_iteration)
    test_preds_list.append(test_pred)

test_ensemble_pred = np.mean(test_preds_list, axis=0)

print(
    f"\nTest predictions: mean={test_ensemble_pred.mean():.4f}, std={test_ensemble_pred.std():.4f}"
)
print(f"Validation target: mean={y_val.mean():.4f}, std={y_val.std():.4f}")

# ============================================================
# 9. SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame(
    {
        "id": range(len(test_ensemble_pred)),
        "Debt-to-Earnings Annual Rate": test_ensemble_pred,
    }
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")

# Save best model for reference
models[0].save_model("./working/lgb_model_final.txt")

# Feature importance (top 10)
importance = models[0].feature_importance(importance_type="gain")
feat_imp = pd.Series(importance, index=X_train.columns).sort_values(ascending=False)
print(f"\nTop 10 features by gain:")
print(feat_imp.head(10))

# ============================================================
# FINAL VALIDATION SCORE (official metric: RMSE)
# ============================================================
score = val_ensemble_rmse
print(f"Final Validation Score: {score}")
