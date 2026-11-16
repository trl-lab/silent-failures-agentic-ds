import pandas as pd
import numpy as np
import os
import json
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

warnings.filterwarnings("ignore")

# Create directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("Loading data...")
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# ============================================================
# 2. IDENTIFY COLUMN TYPES
# ============================================================
target_col = "usdprice"
id_col = "id" if "id" in train.columns else None

# Separate target
y_train = train[target_col].copy()
train = train.drop(columns=[target_col])

# Identify column types
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()
bool_cols = [c for c in train.columns if train[c].dtype == bool]

# Remove id from feature columns
if id_col and id_col in numeric_cols:
    numeric_cols.remove(id_col)
    train[id_col] = train[id_col].astype(str)
    test[id_col] = test[id_col].astype(str)
    if id_col in categorical_cols:
        categorical_cols.remove(id_col)

if id_col and id_col in test.columns:
    test[id_col] = test[id_col].astype(str)

print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Boolean columns ({len(bool_cols)}): {bool_cols}")

# ============================================================
# 3. CREATE TRAIN/VAL SPLIT
# ============================================================
if y_train.nunique() > 10:
    train_idx, val_idx = train_test_split(
        range(len(train)), test_size=0.15, random_state=42
    )
else:
    train_idx, val_idx = train_test_split(
        range(len(train)), test_size=0.15, random_state=42, stratify=y_train
    )

train_split = train.iloc[train_idx].copy()
val_split = train.iloc[val_idx].copy()
y_train_split = y_train.iloc[train_idx].copy()
y_val_split = y_train.iloc[val_idx].copy()

print(f"Train split: {len(train_split)}, Val split: {len(val_split)}")
print(
    f"Target stats - Train: mean={y_train_split.mean():.4f}, std={y_train_split.std():.4f}"
)
print(f"Target stats - Val: mean={y_val_split.mean():.4f}, std={y_val_split.std():.4f}")

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

# Pre-compute top numeric columns by variance (used consistently across all splits)
variances = {
    col: train_split[col].var() for col in numeric_cols if col in train_split.columns
}
top_numeric = sorted(variances, key=variances.get, reverse=True)[:5]
print(f"Top numeric columns by variance: {top_numeric}")


def engineer_features(df, train_df, y_train_ref, is_train=True):
    """Apply feature engineering to a dataframe."""
    df = df.copy()

    # Handle boolean columns
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(np.int8)

    # Handle missing values in numeric columns
    for col in numeric_cols:
        if col in df.columns:
            median_val = train_df[col].median()
            df[col] = df[col].fillna(median_val)

    # Handle missing values in categorical columns
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("MISSING").astype(str)

    # Log transform numeric columns that are non-negative
    for col in numeric_cols:
        if col in df.columns:
            if df[col].min() >= 0:
                df[f"{col}_log"] = np.log1p(df[col])

    # Interaction features between top numeric columns
    for i in range(len(top_numeric)):
        for j in range(i + 1, min(i + 3, len(top_numeric))):
            col_a, col_b = top_numeric[i], top_numeric[j]
            if col_a in df.columns and col_b in df.columns:
                df[f"{col_a}_x_{col_b}"] = df[col_a] * df[col_b]
                denom = df[col_b].replace(0, np.nan)
                df[f"{col_a}_div_{col_b}"] = df[col_a] / denom
                df[f"{col_a}_div_{col_b}"] = df[f"{col_a}_div_{col_b}"].fillna(0)

    # Squared features for top numeric
    for col in top_numeric[:3]:
        if col in df.columns:
            df[f"{col}_sq"] = df[col] ** 2

    # Target encoding for categorical columns (fit on train only)
    for col in categorical_cols:
        if col in df.columns:
            temp = pd.DataFrame(
                {"cat": train_df[col].values, "target": y_train_ref.values}
            )
            cat_means = temp.groupby("cat")["target"].mean()
            cat_counts = temp.groupby("cat")["target"].count()
            global_mean = y_train_ref.mean()

            smoothing = 10
            enc = df[col].map(cat_means).fillna(global_mean)
            counts = df[col].map(cat_counts).fillna(0)
            enc = (enc * counts + global_mean * smoothing) / (counts + smoothing)
            df[f"{col}_target_enc"] = enc

    # Group-based statistical features
    for cat_col in categorical_cols[:3]:
        if cat_col in df.columns:
            for num_col in numeric_cols[:3]:
                if num_col in df.columns:
                    group_stats = train_df.groupby(cat_col)[num_col].agg(
                        ["mean", "std"]
                    )
                    df[f"{cat_col}_{num_col}_mean"] = df[cat_col].map(
                        group_stats["mean"]
                    )
                    df[f"{cat_col}_{num_col}_std"] = df[cat_col].map(group_stats["std"])
                    df[f"{cat_col}_{num_col}_mean"] = df[
                        f"{cat_col}_{num_col}_mean"
                    ].fillna(train_df[num_col].mean())
                    df[f"{cat_col}_{num_col}_std"] = df[
                        f"{cat_col}_{num_col}_std"
                    ].fillna(train_df[num_col].std())

    return df


print("\nEngineering features...")
train_processed = engineer_features(
    train_split, train_split, y_train_split, is_train=True
)
val_processed = engineer_features(val_split, train_split, y_train_split, is_train=False)
test_processed = engineer_features(test, train_split, y_train_split, is_train=False)

print(
    f"Processed shapes - Train: {train_processed.shape}, Val: {val_processed.shape}, Test: {test_processed.shape}"
)

# ============================================================
# 5. FINAL CLEANUP
# ============================================================
for df_ref in [train_processed, val_processed, test_processed]:
    df_ref.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_ref.fillna(0, inplace=True)

# Ensure all features are numeric
for col in train_processed.columns:
    if train_processed[col].dtype == object:
        train_processed[col] = pd.to_numeric(
            train_processed[col], errors="coerce"
        ).fillna(0)
        val_processed[col] = pd.to_numeric(val_processed[col], errors="coerce").fillna(
            0
        )
        test_processed[col] = pd.to_numeric(
            test_processed[col], errors="coerce"
        ).fillna(0)

# Define feature columns
feature_cols = train_processed.columns.tolist()
print(f"\nTotal features: {len(feature_cols)}")

# ============================================================
# 6. PREPARE DATA ARRAYS
# ============================================================
X_train = train_processed[feature_cols].values.astype(np.float32)
X_val = val_processed[feature_cols].values.astype(np.float32)
X_test = test_processed[feature_cols].values.astype(np.float32)

y_train_orig = y_train_split.values.astype(np.float64)
y_val_orig = y_val_split.values.astype(np.float64)

# Log transform target
y_train_log = np.log1p(np.clip(y_train_orig, 0, None))
y_val_log = np.log1p(np.clip(y_val_orig, 0, None))

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(
    f"Target (orig) - Train: mean={y_train_orig.mean():.4f}, std={y_train_orig.std():.4f}"
)
print(
    f"Target (orig) - Val:   mean={y_val_orig.mean():.4f}, std={y_val_orig.std():.4f}"
)

# ============================================================
# 7. TRAIN XGBOOST MODEL
# ============================================================
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}

print("\nTraining XGBoost...")

model = xgb.XGBRegressor(
    **params,
    n_estimators=5000,
    early_stopping_rounds=100,
)


model.fit(
    X_train,
    y_train_log,
    eval_set=[(X_val, y_val_log)],
    verbose=False,
)

best_iter = model.best_iteration
print(f"Best iteration: {best_iter}")

# ============================================================
# 8. VALIDATION PREDICTIONS (using best iteration)
# ============================================================
val_pred_log = model.predict(X_val, iteration_range=(0, best_iter + 1))
val_pred_orig = np.expm1(np.clip(val_pred_log, -20, 20))
val_pred_orig = np.clip(val_pred_orig, 0, None)

# Compute RMSE on original scale (official metric)
rmse_val = np.sqrt(mean_squared_error(y_val_orig, val_pred_orig))
print(f"Validation RMSE (original scale): {rmse_val:.6f}")

# ============================================================
# 9. TEST PREDICTIONS (identical pipeline to validation)
# ============================================================
test_pred_log = model.predict(X_test, iteration_range=(0, best_iter + 1))
test_pred_orig = np.expm1(np.clip(test_pred_log, -20, 20))
test_pred_orig = np.clip(test_pred_orig, 0, None)

# ============================================================
# 10. SAVE SUBMISSION
# ============================================================
if id_col and id_col in test.columns:
    test_ids = test[id_col].values.astype(int)
else:
    test_ids = np.arange(len(test))

submission = pd.DataFrame({"id": test_ids, "usdprice": test_pred_orig})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(
    f"Predictions: mean={test_pred_orig.mean():.4f}, std={test_pred_orig.std():.4f}, "
    f"min={test_pred_orig.min():.4f}, max={test_pred_orig.max():.4f}"
)

# ============================================================
# 11. FINAL VALIDATION SCORE
# ============================================================
score = rmse_val
print(f"Final Validation Score: {score}")