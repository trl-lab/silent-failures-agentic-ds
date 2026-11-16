import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

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

# ============================================================
# 2. CREATE TRAIN/VAL SPLIT
# ============================================================
target_col = "Amount in USD"
y_full = train_df[target_col].values
train_df = train_df.drop(columns=[target_col])

# Check for id column
id_col = None
if "id" in train_df.columns:
    id_col = "id"
    train_df = train_df.drop(columns=[id_col])

X_train, X_val, y_train, y_val = train_test_split(
    train_df, y_full, test_size=0.15, random_state=42
)

print(f"Train split: {X_train.shape}, Val split: {X_val.shape}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
X_combined = pd.concat([X_train, X_val], axis=0, ignore_index=True)

# Identify column types
numeric_cols = X_combined.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_combined.select_dtypes(
    include=["object", "category"]
).columns.tolist()
datetime_cols = X_combined.select_dtypes(include=["datetime64"]).columns.tolist()

# Handle mixed-type object columns
for col in categorical_cols[:]:
    parsed = pd.to_datetime(X_combined[col], errors="coerce")
    if parsed.notna().sum() > len(X_combined) * 0.5:
        X_combined[col] = parsed
        datetime_cols.append(col)
        categorical_cols.remove(col)
    else:
        parsed_num = pd.to_numeric(X_combined[col], errors="coerce")
        if parsed_num.notna().sum() > len(X_combined) * 0.5:
            X_combined[col] = parsed_num
            numeric_cols.append(col)
            categorical_cols.remove(col)

numeric_cols = list(set(numeric_cols))
categorical_cols = list(set(categorical_cols))
datetime_cols = list(set(datetime_cols))

print(f"Numeric cols ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical cols ({len(categorical_cols)}): {categorical_cols}")
print(f"Datetime cols ({len(datetime_cols)}): {datetime_cols}")

# --- Datetime Feature Engineering ---
for col in datetime_cols:
    if X_combined[col].dtype == "object":
        X_combined[col] = pd.to_datetime(X_combined[col], errors="coerce")

    X_combined[f"{col}_year"] = X_combined[col].dt.year
    X_combined[f"{col}_month"] = X_combined[col].dt.month
    X_combined[f"{col}_day"] = X_combined[col].dt.day
    X_combined[f"{col}_dayofweek"] = X_combined[col].dt.dayofweek
    X_combined[f"{col}_quarter"] = X_combined[col].dt.quarter
    X_combined[f"{col}_weekofyear"] = X_combined[col].dt.isocalendar().week.astype(int)
    X_combined[f"{col}_dayofyear"] = X_combined[col].dt.dayofyear
    X_combined[f"{col}_is_weekend"] = (X_combined[col].dt.dayofweek >= 5).astype(int)
    X_combined[f"{col}_ordinal"] = X_combined[col].astype(np.int64) // 10**9
    X_combined = X_combined.drop(columns=[col])

# --- Categorical Feature Engineering ---
if len(categorical_cols) > 0:
    # Replace None with NaN so SimpleImputer handles them correctly
    for col in categorical_cols:
        X_combined[col] = X_combined[col].where(X_combined[col].notna(), np.nan)
        X_combined[col] = X_combined[col].apply(lambda x: np.nan if x is None else x)

    cat_imputer = SimpleImputer(strategy="most_frequent")
    train_cat_raw = X_combined.iloc[: len(X_train)][categorical_cols]
    cat_imputer.fit(train_cat_raw)
    X_combined[categorical_cols] = cat_imputer.transform(X_combined[categorical_cols])

    # Safety: replace any remaining None with a placeholder string
    for col in categorical_cols:
        X_combined[col] = X_combined[col].apply(lambda x: "__MISSING__" if x is None else x)

    train_cat = X_combined.iloc[: len(X_train)][categorical_cols]
    cat_categories = {}
    for col in categorical_cols:
        cats = [x for x in train_cat[col].unique().tolist() if x is not None]
        cats = sorted(cats)
        cat_categories[col] = cats

    ordinal_encoder = OrdinalEncoder(
        categories=[cat_categories[col] for col in categorical_cols],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    # Fit on train only to avoid data leakage
    ordinal_encoder.fit(X_combined.iloc[: len(X_train)][categorical_cols])
    X_combined[categorical_cols] = ordinal_encoder.transform(X_combined[categorical_cols])

# --- Numeric Feature Engineering ---
if len(numeric_cols) > 0:
    train_numeric = X_combined.iloc[: len(X_train)][numeric_cols]
    num_imputer = SimpleImputer(strategy="median")
    num_imputer.fit(train_numeric)
    X_combined[numeric_cols] = num_imputer.transform(X_combined[numeric_cols])

    # Log transform for skewed positive numeric features
    for col in numeric_cols:
        vals = X_combined[col].values
        positive_ratio = (vals > 0).sum() / len(vals)
        if positive_ratio > 0.9:
            X_combined[f"{col}_log"] = np.log1p(np.maximum(vals, 0))

    # Interaction features
    if len(numeric_cols) <= 20:
        for i in range(min(len(numeric_cols), 5)):
            for j in range(i + 1, min(len(numeric_cols), 5)):
                col_i, col_j = numeric_cols[i], numeric_cols[j]
                X_combined[f"{col_i}_div_{col_j}"] = X_combined[col_i] / (
                    X_combined[col_j].abs() + 1e-8
                )

# Use raw target to align with official RMSE metric
y_train_log = y_train.astype(np.float64)
y_val_log = y_val.astype(np.float64)

# ============================================================
# 4. PREPARE FINAL ARRAYS
# ============================================================
feature_cols = X_combined.columns.tolist()
print(f"Total features: {len(feature_cols)}")

X_train_final = X_combined.iloc[: len(X_train)].reset_index(drop=True)
X_val_final = X_combined.iloc[len(X_train) :].reset_index(drop=True)

# Process test with same pipeline
X_test_proc = test_df.copy()
if id_col and id_col in X_test_proc.columns:
    test_ids = X_test_proc[id_col].values
    X_test_proc = X_test_proc.drop(columns=[id_col])
else:
    test_ids = np.arange(len(X_test_proc))

for col in datetime_cols:
    if col in X_test_proc.columns:
        if X_test_proc[col].dtype == "object":
            X_test_proc[col] = pd.to_datetime(X_test_proc[col], errors="coerce")
        X_test_proc[f"{col}_year"] = X_test_proc[col].dt.year
        X_test_proc[f"{col}_month"] = X_test_proc[col].dt.month
        X_test_proc[f"{col}_day"] = X_test_proc[col].dt.day
        X_test_proc[f"{col}_dayofweek"] = X_test_proc[col].dt.dayofweek
        X_test_proc[f"{col}_quarter"] = X_test_proc[col].dt.quarter
        X_test_proc[f"{col}_weekofyear"] = (
            X_test_proc[col].dt.isocalendar().week.astype(int)
        )
        X_test_proc[f"{col}_dayofyear"] = X_test_proc[col].dt.dayofyear
        X_test_proc[f"{col}_is_weekend"] = (X_test_proc[col].dt.dayofweek >= 5).astype(
            int
        )
        X_test_proc[f"{col}_ordinal"] = X_test_proc[col].astype(np.int64) // 10**9
        X_test_proc = X_test_proc.drop(columns=[col])

if len(categorical_cols) > 0:
    for col in categorical_cols:
        X_test_proc[col] = X_test_proc[col].apply(lambda x: np.nan if x is None else x)
    X_test_proc[categorical_cols] = cat_imputer.transform(X_test_proc[categorical_cols])
    for col in categorical_cols:
        X_test_proc[col] = X_test_proc[col].apply(lambda x: "__MISSING__" if x is None else x)
    X_test_proc[categorical_cols] = ordinal_encoder.transform(
        X_test_proc[categorical_cols]
    )

if len(numeric_cols) > 0:
    X_test_proc[numeric_cols] = num_imputer.transform(X_test_proc[numeric_cols])
    for col in numeric_cols:
        vals = X_test_proc[col].values
        positive_ratio = (vals > 0).sum() / len(vals)
        if positive_ratio > 0.9:
            X_test_proc[f"{col}_log"] = np.log1p(np.maximum(vals, 0))
    if len(numeric_cols) <= 20:
        for i in range(min(len(numeric_cols), 5)):
            for j in range(i + 1, min(len(numeric_cols), 5)):
                col_i, col_j = numeric_cols[i], numeric_cols[j]
                X_test_proc[f"{col_i}_div_{col_j}"] = X_test_proc[col_i] / (
                    X_test_proc[col_j].abs() + 1e-8
                )

# Align test columns to feature_cols
for col in feature_cols:
    if col not in X_test_proc.columns:
        X_test_proc[col] = 0.0
X_test_final = X_test_proc[feature_cols].reset_index(drop=True)

# Convert to numpy
X_train_np = X_train_final.values.astype(np.float32)
X_val_np = X_val_final.values.astype(np.float32)
X_test_np = X_test_final.values.astype(np.float32)

X_train_np = np.nan_to_num(X_train_np, nan=0.0, posinf=1e6, neginf=-1e6)
X_val_np = np.nan_to_num(X_val_np, nan=0.0, posinf=1e6, neginf=-1e6)
X_test_np = np.nan_to_num(X_test_np, nan=0.0, posinf=1e6, neginf=-1e6)

print(
    f"Final shapes - Train: {X_train_np.shape}, Val: {X_val_np.shape}, Test: {X_test_np.shape}"
)

# ============================================================
# 5. TRAIN LIGHTGBM MODEL
# ============================================================
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_gain_to_split": 0.01,
    "verbose": -1,
    "n_estimators": 5000,
    "random_state": 42,
    "n_jobs": -1,
}

model = lgb.LGBMRegressor(**lgb_params)

print("Training with early stopping...")
model.fit(
    X_train_np,
    y_train_log,
    eval_set=[(X_val_np, y_val_log)],
    eval_metric="rmse",
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=500),
    ],
)

best_iter = model.best_iteration_
best_val_rmse_log = model.best_score_["valid_0"]["rmse"]
print(f"Best iteration: {best_iter}")
print(f"Best validation RMSE: {best_val_rmse_log:.4f}")

# ============================================================
# 6. VALIDATION: RMSE on ORIGINAL scale
# ============================================================
val_pred_orig = model.predict(X_val_np, num_iteration=best_iter)
val_pred_orig = np.maximum(val_pred_orig, 0.0)

val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred_orig)))
print(f"Validation RMSE: {val_rmse:.4f}")

baseline_rmse = float(
    np.sqrt(mean_squared_error(y_val, np.full(len(y_val), y_train.mean())))
)
print(f"Baseline (mean) RMSE: {baseline_rmse:.4f}")
print(f"Improvement over baseline: {(1 - val_rmse / baseline_rmse) * 100:.1f}%")

# ============================================================
# 7. TEST INFERENCE
# ============================================================
test_pred_orig = model.predict(X_test_np, num_iteration=best_iter)
test_pred_orig = np.maximum(test_pred_orig, 0.0)

print(
    f"Test predictions: min={test_pred_orig.min():.2f}, max={test_pred_orig.max():.2f}, mean={test_pred_orig.mean():.2f}"
)

# ============================================================
# 8. SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame({"id": test_ids.astype(int), "Amount in USD": test_pred_orig})
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape[0]} rows")

# ============================================================
# FINAL VALIDATION SCORE
# ============================================================
score = val_rmse
print(f"Final Validation Score: {score}")