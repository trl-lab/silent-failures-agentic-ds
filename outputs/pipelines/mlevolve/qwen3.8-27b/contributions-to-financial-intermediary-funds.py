import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import os

# Create directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# ============================================================
# 2. IDENTIFY FEATURE TYPES
# ============================================================
target_col = "Amount in USD"

datetime_cols = []
categorical_cols = []
numerical_cols = []

for col in train_df.columns:
    if col == target_col:
        continue
    if train_df[col].dtype == "object" or train_df[col].dtype.name == "category":
        categorical_cols.append(col)
    elif pd.api.types.is_datetime64_any_dtype(train_df[col]):
        datetime_cols.append(col)
    elif train_df[col].dtype in ["int64", "int32", "float64", "float32"]:
        numerical_cols.append(col)

# Check for string columns that might be dates
for col in categorical_cols[:]:
    try:
        non_null = train_df[col].dropna()
        if len(non_null) > 0:
            converted = pd.to_datetime(non_null, errors="coerce")
            if converted.notna().sum() / len(non_null) > 0.8:
                datetime_cols.append(col)
                categorical_cols.remove(col)
    except:
        pass

print(f"Datetime cols: {datetime_cols}")
print(f"Categorical cols: {categorical_cols}")
print(f"Numerical cols: {numerical_cols}")

# ============================================================
# 3. CREATE TRAIN/VAL SPLIT (before any processing)
# ============================================================
train_work = train_df.copy()
train_work["target_quantile"] = pd.qcut(
    train_work[target_col], q=10, labels=False, duplicates="drop"
)

X_train, X_val, y_train, y_val = train_test_split(
    train_work.drop(columns=[target_col, "target_quantile"]),
    train_work[target_col],
    test_size=0.2,
    random_state=42,
    stratify=train_work["target_quantile"],
)

print(f"Train split: {X_train.shape}, Val split: {X_val.shape}")

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================


def engineer_features(
    df,
    is_train=False,
    target_encodings=None,
    datetime_cols=None,
    categorical_cols=None,
    numerical_cols=None,
):
    """Apply feature engineering. Fit on train, transform on val/test."""
    df = df.copy()
    if datetime_cols is None:
        datetime_cols = []
    if categorical_cols is None:
        categorical_cols = []
    if numerical_cols is None:
        numerical_cols = []

    # Store medians for numerical fill
    medians = {}

    # --- Datetime Feature Engineering ---
    for col in datetime_cols:
        if col not in df.columns:
            continue
        df[col] = pd.to_datetime(df[col], errors="coerce")

        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_day"] = df[col].dt.day
        df[f"{col}_dayofweek"] = df[col].dt.dayofweek
        df[f"{col}_dayofyear"] = df[col].dt.dayofyear
        df[f"{col}_quarter"] = df[col].dt.quarter
        df[f"{col}_weekofyear"] = df[col].dt.isocalendar().week.astype(int)

        df[f"{col}_month_sin"] = np.sin(2 * np.pi * df[col].dt.month / 12)
        df[f"{col}_month_cos"] = np.cos(2 * np.pi * df[col].dt.month / 12)
        df[f"{col}_dayofweek_sin"] = np.sin(2 * np.pi * df[col].dt.dayofweek / 7)
        df[f"{col}_dayofweek_cos"] = np.cos(2 * np.pi * df[col].dt.dayofweek / 7)
        df[f"{col}_dayofyear_sin"] = np.sin(2 * np.pi * df[col].dt.dayofyear / 365)
        df[f"{col}_dayofyear_cos"] = np.cos(2 * np.pi * df[col].dt.dayofyear / 365)

        df[f"{col}_days_since_epoch"] = (df[col] - pd.Timestamp("2000-01-01")).dt.days
        df[f"{col}_days_since_epoch"] = df[f"{col}_days_since_epoch"].fillna(0)

        df.drop(columns=[col], inplace=True)

    # --- Categorical Feature Engineering (keep as strings for LightGBM) ---
    for col in categorical_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna("MISSING").astype(str)

    # --- Numerical Feature Engineering ---
    for col in numerical_cols:
        if col not in df.columns:
            continue
        if is_train:
            median_val = df[col].median()
            medians[col] = median_val
            df[col] = df[col].fillna(median_val)
        else:
            df[col] = df[col].fillna(medians.get(col, 0))

    # --- Target Encoding Features (median target per category) ---
    if is_train:
        # target_encodings will be computed externally and passed back
        pass
    else:
        if target_encodings is not None:
            for col in categorical_cols:
                if col not in df.columns:
                    continue
                if col in target_encodings:
                    mapping = target_encodings[col]
                    global_median = mapping.get("__global__", 0.0)
                    df[f"{col}_target_enc"] = df[col].map(mapping).fillna(global_median)

    return df, medians


# Apply feature engineering
X_train_fe, medians = engineer_features(
    X_train,
    is_train=True,
    datetime_cols=datetime_cols,
    categorical_cols=categorical_cols,
    numerical_cols=numerical_cols,
)
X_val_fe, _ = engineer_features(
    X_val,
    is_train=False,
    datetime_cols=datetime_cols,
    categorical_cols=categorical_cols,
    numerical_cols=numerical_cols,
)
X_test_fe, _ = engineer_features(
    test_df,
    is_train=False,
    datetime_cols=datetime_cols,
    categorical_cols=categorical_cols,
    numerical_cols=numerical_cols,
)

# --- Compute target encodings (fit on train split only) ---
target_encodings = {}
for col in categorical_cols:
    if col in X_train_fe.columns:
        combined = pd.DataFrame({"cat": X_train_fe[col].values, "target": y_train.values})
        med_per_cat = combined.groupby("cat")["target"].median().to_dict()
        global_median = y_train.median()
        med_per_cat["__global__"] = global_median
        target_encodings[col] = med_per_cat

# Apply target encoding to all splits
for col in categorical_cols:
    if col in X_train_fe.columns and col in target_encodings:
        mapping = target_encodings[col]
        global_median = mapping.get("__global__", 0.0)
        X_train_fe[f"{col}_target_enc"] = X_train_fe[col].map(mapping).fillna(global_median)
        X_val_fe[f"{col}_target_enc"] = X_val_fe[col].map(mapping).fillna(global_median)
        X_test_fe[f"{col}_target_enc"] = X_test_fe[col].map(mapping).fillna(global_median)

print(
    f"After feature engineering: Train={X_train_fe.shape}, Val={X_val_fe.shape}, Test={X_test_fe.shape}"
)

# ============================================================
# 5. TARGET TRANSFORMATION
# ============================================================
y_train_log = np.log1p(np.clip(y_train.values, 0, None))
y_val_log = np.log1p(np.clip(y_val.values, 0, None))

print(
    f"Target (log) - Train: mean={y_train_log.mean():.4f}, std={y_train_log.std():.4f}"
)

# ============================================================
# 6. PREPARE DATA FOR LIGHTGBM
# ============================================================
n_features = X_train_fe.shape[1]
print(f"Number of features: {n_features}")

# Identify categorical columns for LightGBM native handling
cat_feature_cols = [col for col in categorical_cols if col in X_train_fe.columns]
print(f"Categorical features for LightGBM: {cat_feature_cols}")

# Convert categorical columns to pandas 'category' dtype (required by LightGBM sklearn API)
for col in cat_feature_cols:
    X_train_fe[col] = X_train_fe[col].astype("category")
    X_val_fe[col] = X_val_fe[col].astype("category")
    X_test_fe[col] = X_test_fe[col].astype("category")

# ============================================================
# 7. LIGHTGBM MODEL CONFIGURATION
# ============================================================
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
}

# ============================================================
# 8. TRAIN WITH EARLY STOPPING
# ============================================================
y_val_original = y_val.values

model = lgb.LGBMRegressor(**lgb_params)

model.fit(
    X_train_fe,
    y_train_log,
    eval_set=[(X_val_fe, y_val_log)],
    eval_metric="rmse",
    categorical_feature=cat_feature_cols,
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)],
)

print(f"Best iteration: {model.best_iteration_}")

# ============================================================
# 9. GENERATE PREDICTIONS
# ============================================================
# Validation predictions
val_pred_log = model.predict(X_val_fe, categorical_feature=cat_feature_cols)
val_pred_original = np.expm1(val_pred_log)
val_pred_original = np.maximum(val_pred_original, 0)

final_rmse = np.sqrt(np.mean((val_pred_original - y_val_original) ** 2))
print(f"\nBest Validation RMSE (original scale): {final_rmse:.4f}")

# Test predictions
test_pred_log = model.predict(X_test_fe, categorical_feature=cat_feature_cols)
test_pred_original = np.expm1(test_pred_log)
test_pred_original = np.maximum(test_pred_original, 0)

# ============================================================
# 11. SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame(
    {"id": range(len(test_pred_original)), "Amount in USD": test_pred_original}
)
submission.to_csv("./submission/submission.csv", index=False)

print(f"Submission saved: {len(submission)} predictions")
print(
    f"Test pred stats: mean={test_pred_original.mean():.2f}, std={test_pred_original.std():.2f}, min={test_pred_original.min():.2f}, max={test_pred_original.max():.2f}"
)

# ============================================================
# FINAL OUTPUT
# ============================================================
score = final_rmse
print(f"Final Validation Score: {score}")