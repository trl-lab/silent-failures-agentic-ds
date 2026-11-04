import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import json
import warnings

warnings.filterwarnings("ignore")
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Train columns: {list(train.columns)}")
print(f"Target stats:\n{train['time_to_approve'].describe()}")

# ============================================================
# 2. IDENTIFY COLUMN TYPES
# ============================================================
target_col = "time_to_approve"

# Identify date columns
date_cols = []
for c in train.columns:
    if train[c].dtype == "datetime64[ns]":
        date_cols.append(c)
    elif train[c].dtype == "object" and ("date" in c.lower() or "time" in c.lower()):
        try:
            train[c] = pd.to_datetime(train[c])
            test[c] = pd.to_datetime(test[c])
            date_cols.append(c)
        except:
            pass

# Identify numeric and categorical columns
numeric_cols = [
    c
    for c in train.columns
    if train[c].dtype in ["int64", "float64", "int32", "float32"] and c != target_col
]
categorical_cols = [
    c for c in train.columns if train[c].dtype == "object" and c != target_col
]
id_cols = [c for c in train.columns if "id" in c.lower()]
bool_cols = [c for c in train.columns if train[c].dtype == "bool" and c != target_col]

numeric_cols = [c for c in numeric_cols if c not in id_cols and c != target_col]
categorical_cols = [c for c in categorical_cols if c not in id_cols and c != target_col]
numeric_cols = numeric_cols + bool_cols

print(f"Date cols: {date_cols}")
print(f"Numeric cols ({len(numeric_cols)}): {numeric_cols[:10]}...")
print(f"Categorical cols ({len(categorical_cols)}): {categorical_cols[:10]}...")
print(f"ID cols: {id_cols}")

# ============================================================
# 3. CREATE TRAIN/VAL SPLIT (BEFORE any feature engineering)
# ============================================================
train_data, val_data = train_test_split(train, test_size=0.15, random_state=42)
print(f"Train split: {train_data.shape}, Val split: {val_data.shape}")


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
# High-cardinality categoricals for frequency encoding
high_card_cats = ["Street Name", "Block", "Lot", "Permit Number", "Description"]

def engineer_features(
    df, target_encoders=None, train_stats=None, is_train=False, target=None
):
    df = df.copy()

    if is_train:
        target_encoders = {}
        train_stats = {}

    # --- Temporal features from date columns ---
    for c in date_cols:
        if c in df.columns:
            if df[c].dtype == "object":
                df[c] = pd.to_datetime(df[c], errors="coerce")
            if df[c].dtype == "datetime64[ns]":
                df[f"{c}_year"] = df[c].dt.year
                df[f"{c}_month"] = df[c].dt.month
                df[f"{c}_day"] = df[c].dt.day
                df[f"{c}_dayofweek"] = df[c].dt.dayofweek
                df[f"{c}_quarter"] = df[c].dt.quarter
                df[f"{c}_weekofyear"] = df[c].dt.isocalendar().week.fillna(0).astype(int)
                df[f"{c}_is_weekend"] = (df[c].dt.dayofweek >= 5).astype(int)
                df[f"{c}_dayofyear"] = df[c].dt.dayofyear
                df[f"{c}_month_sin"] = np.sin(2 * np.pi * df[c].dt.month / 12)
                df[f"{c}_month_cos"] = np.cos(2 * np.pi * df[c].dt.month / 12)
                df[f"{c}_dow_sin"] = np.sin(2 * np.pi * df[c].dt.dayofweek / 7)
                df[f"{c}_dow_cos"] = np.cos(2 * np.pi * df[c].dt.dayofweek / 7)
                df.drop(columns=[c], inplace=True, errors="ignore")

    # --- Days-difference features between key dates ---
    base_date = "Permit Creation Date"
    if base_date in [c for c in date_cols]:
        # We need to keep the base date for diff calculation, so re-parse if needed
        if base_date not in df.columns:
            # It was dropped above, so we need to handle this differently
            pass
    # Re-parse dates for diff calculation (before dropping)
    # Actually dates are already dropped, so we compute diffs from the original
    # Let's restructure: compute diffs BEFORE dropping dates

    # Since dates were already dropped above, we need a different approach
    # Compute day differences using the year/month/day features we created
    # Better approach: compute diffs from the original date columns before dropping
    # Let's add a pre-processing step

    # --- Target encoding with Bayesian smoothing ---
    smoothing = 20  # Bayesian smoothing parameter
    for c in categorical_cols:
        if c not in df.columns:
            continue
        if is_train and target is not None and target in df.columns:
            grouped = df.groupby(c)[target]
            cat_mean = grouped.mean()
            cat_count = grouped.count()
            global_mean = df[target].mean()
            # Bayesian smoothed target encoding
            smoothed = (cat_count * cat_mean + smoothing * global_mean) / (cat_count + smoothing)
            target_encoders[c] = {
                "smoothed": smoothed,
                "mean": cat_mean,
                "count": cat_count,
                "global_mean": global_mean,
            }
            df[f"{c}_target_enc"] = df[c].map(smoothed).fillna(global_mean)
            df[f"{c}_target_count"] = df[c].map(cat_count).fillna(0)
        elif (
            target_encoders is not None
            and c in target_encoders
            and target_encoders[c] is not None
            and "smoothed" in target_encoders[c]
        ):
            enc = target_encoders[c]
            df[f"{c}_target_enc"] = df[c].map(enc["smoothed"]).fillna(enc["global_mean"])
            df[f"{c}_target_count"] = df[c].map(enc["count"]).fillna(0)
        else:
            df[f"{c}_target_enc"] = 0
            df[f"{c}_target_count"] = 0

    # --- Frequency encoding for high-cardinality categoricals ---
    for c in high_card_cats:
        if c not in df.columns:
            continue
        if is_train:
            freq = df[c].value_counts(normalize=True)
            target_encoders[f"{c}_freq"] = freq
            df[f"{c}_freq"] = df[c].map(freq).fillna(0)
        elif target_encoders is not None and f"{c}_freq" in target_encoders:
            df[f"{c}_freq"] = df[c].map(target_encoders[f"{c}_freq"]).fillna(0)
        else:
            df[f"{c}_freq"] = 0

    # --- Label encoding for tree models ---
    for c in categorical_cols:
        if c in df.columns:
            if is_train:
                le = LabelEncoder()
                df[c] = df[c].fillna("__NAN__")
                df[f"{c}_label"] = le.fit_transform(df[c].astype(str))
                target_encoders[f"{c}_le"] = le
            elif target_encoders is not None and f"{c}_le" in target_encoders:
                le = target_encoders[f"{c}_le"]
                df[c] = df[c].fillna("__NAN__")
                known = set(le.classes_)
                df[f"{c}_label"] = df[c].apply(
                    lambda x: le.transform([x])[0] if x in known else -1
                )
            else:
                df[f"{c}_label"] = 0

    # --- Numeric feature engineering ---
    for c in numeric_cols:
        if c not in df.columns:
            continue
        if is_train:
            train_stats[c] = {
                "mean": df[c].mean(),
                "std": df[c].std(),
                "median": df[c].median(),
            }
            if df[c].min() >= 0 and df[c].skew() > 1:
                df[f"{c}_log"] = np.log1p(df[c])
        else:
            if train_stats is not None and c in train_stats:
                stats = train_stats[c]
                df[c] = df[c].fillna(stats["mean"])
                if df[c].min() >= 0 and df[c].skew() > 1:
                    df[f"{c}_log"] = np.log1p(df[c])
            else:
                df[c] = df[c].fillna(0)

    # --- Interaction features between key categoricals ---
    if len(categorical_cols) >= 2:
        top_cats = [c for c in categorical_cols if c in df.columns][:5]
        for i in range(len(top_cats)):
            for j in range(i + 1, len(top_cats)):
                c1, c2 = top_cats[i], top_cats[j]
                if c1 in df.columns and c2 in df.columns:
                    interaction = df[c1].astype(str) + "_" + df[c2].astype(str)
                    key = f"{c1}_x_{c2}"
                    if is_train and target is not None and target in df.columns:
                        tmp = pd.DataFrame({"inter": interaction, "target": df[target]})
                        inter_mean = tmp.groupby("inter")["target"].mean()
                        inter_count = tmp.groupby("inter")["target"].count()
                        global_mean = df[target].mean()
                        smoothed_inter = (inter_count * inter_mean + 20 * global_mean) / (inter_count + 20)
                        target_encoders[key] = smoothed_inter
                        df[f"{key}_target"] = interaction.map(smoothed_inter).fillna(global_mean)
                    elif (
                        target_encoders is not None
                        and key in target_encoders
                        and target_encoders[key] is not None
                    ):
                        df[f"{key}_target"] = interaction.map(
                            target_encoders[key]
                        ).fillna(target_encoders.get("_global_mean", 0))
                    else:
                        df[f"{key}_target"] = 0

    # --- Missing value indicators ---
    for c in numeric_cols + categorical_cols:
        if c in df.columns:
            df[f"{c}_missing"] = df[c].isnull().astype(int)

    return df, target_encoders, train_stats


# --- Compute days-difference features BEFORE feature engineering ---
# We need to compute these from the original date columns
def add_date_diff_features(df, ref_date_col="Permit Creation Date"):
    df = df.copy()
    for c in date_cols:
        if c == ref_date_col or c not in df.columns:
            continue
        if df[c].dtype == "object":
            df[c] = pd.to_datetime(df[c], errors="coerce")
        if df[ref_date_col].dtype == "object":
            df[ref_date_col] = pd.to_datetime(df[ref_date_col], errors="coerce")
        if c in df.columns and ref_date_col in df.columns:
            diff = (df[c] - df[ref_date_col]).dt.days
            df[f"days_{ref_date_col}_to_{c}"] = diff.fillna(-9999)
    return df

print("Adding date difference features...")
train_data = add_date_diff_features(train_data)
val_data = add_date_diff_features(val_data)
test = add_date_diff_features(test)

# Add date diff columns to numeric_cols so they get processed
date_diff_cols = [c for c in train_data.columns if c.startswith("days_")]
numeric_cols = numeric_cols + date_diff_cols

print("\nEngineering features on train split...")
train_processed, target_encoders, train_stats = engineer_features(
    train_data, target_encoders=None, train_stats=None, is_train=True, target=target_col
)

print("Engineering features on val split...")
val_processed, _, _ = engineer_features(
    val_data,
    target_encoders=target_encoders,
    train_stats=train_stats,
    is_train=False,
    target=None,
)

print("Engineering features on test...")
test_processed, _, _ = engineer_features(
    test,
    target_encoders=target_encoders,
    train_stats=train_stats,
    is_train=False,
    target=None,
)

# --- Add zero-inflation features ---
# is_zero indicator and log_positive
train_processed["is_zero_target"] = (train_processed[target_col] == 0).astype(np.float32)
val_processed["is_zero_target"] = (val_processed[target_col] == 0).astype(np.float32)
# log_positive: log of target where > 0, 0 otherwise
train_processed["log_positive_target"] = np.where(
    train_processed[target_col] > 0,
    np.log1p(train_processed[target_col]),
    0.0
).astype(np.float32)
val_processed["log_positive_target"] = np.where(
    val_processed[target_col] > 0,
    np.log1p(val_processed[target_col]),
    0.0
).astype(np.float32)
# For test, we don't have target, so set to 0 (these are only used as targets, not features)
test_processed["is_zero_target"] = 0.0
test_processed["log_positive_target"] = 0.0

# ============================================================
# 5. PREPARE FEATURE MATRICES
# ============================================================
exclude_cols = (
    [target_col, "is_zero_target", "log_positive_target"]
    + id_cols
    + date_cols
    + categorical_cols
    + [c for c in numeric_cols if c in train.columns and train[c].dtype == "bool"]
)

all_cols = set(train_processed.columns)
feature_cols = [c for c in all_cols if c not in exclude_cols and c != target_col]

for c in numeric_cols:
    if c in train_processed.columns and c not in feature_cols:
        feature_cols.append(c)

# Ensure consistency
for c in feature_cols:
    if c not in val_processed.columns:
        val_processed[c] = 0
    if c not in test_processed.columns:
        test_processed[c] = 0

print(f"Total features: {len(feature_cols)}")

# ============================================================
# 6. BUILD MODEL MATRICES
# ============================================================
X_train = train_processed[feature_cols].values.astype(np.float32)
y_train = train_processed[target_col].values.astype(np.float32)
X_val = val_processed[feature_cols].values.astype(np.float32)
y_val = val_processed[target_col].values.astype(np.float32)
X_test = test_processed[feature_cols].values.astype(np.float32)

# Clean NaN/inf
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

# Log-transform target
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

# Zero-inflation targets
y_train_is_zero = (y_train == 0).astype(np.float32)
y_val_is_zero = (y_val == 0).astype(np.float32)

# Positive-only targets for the positive regressor
pos_mask_train = y_train > 0
pos_mask_val = y_val > 0
y_train_pos = y_train_log[pos_mask_train]
y_val_pos = y_val_log[pos_mask_val]
X_train_pos = X_train[pos_mask_train]
X_val_pos = X_val[pos_mask_val]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(
    f"Target raw: mean={y_train.mean():.2f}, std={y_train.std():.2f}, max={y_train.max():.2f}"
)
print(f"Target log: mean={y_train_log.mean():.4f}, std={y_train_log.std():.4f}")
print(f"Zero fraction: train={y_train_is_zero.mean():.3f}, val={y_val_is_zero.mean():.3f}")
print(f"Positive samples: train={pos_mask_train.sum()}, val={pos_mask_val.sum()}")

# ============================================================
# 7. MODEL 1: LightGBM with Tweedie loss (zero-inflated continuous target)
# ============================================================
print("\n=== Training LightGBM Tweedie (variance_power=1.5, raw target) ===")

lgb_tweedie = lgb.LGBMRegressor(
    objective="tweedie",
    tweedie_variance_power=1.5,
    n_estimators=5000,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=63,
    min_child_samples=30,
    subsample=0.75,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=1.0,
    min_split_gain=0.01,
    random_state=42,
    verbose=-1,
    n_jobs=-1,
)

lgb_tweedie.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    callbacks=[
        lgb.early_stopping(200, verbose=False),
        lgb.log_evaluation(period=500),
    ],
)

print(f"LightGBM Tweedie best iteration: {lgb_tweedie.best_iteration_}")

val_pred_tweedie = np.clip(lgb_tweedie.predict(X_val), 0, None)
test_pred_tweedie = np.clip(lgb_tweedie.predict(X_test), 0, None)

rmse_tweedie = root_mean_squared_error(y_val, val_pred_tweedie)
print(f"LightGBM Tweedie RMSE: {rmse_tweedie:.4f}")

# ============================================================
# 8. MODEL 2: LightGBM with Quantile Regression (alpha=0.5)
# ============================================================
print("\n=== Training LightGBM Quantile (alpha=0.5, raw target) ===")

lgb_quantile = lgb.LGBMRegressor(
    objective="quantile",
    alpha=0.5,
    n_estimators=5000,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=63,
    min_child_samples=30,
    subsample=0.75,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=1.0,
    min_split_gain=0.01,
    random_state=42,
    verbose=-1,
    n_jobs=-1,
)

lgb_quantile.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    callbacks=[
        lgb.early_stopping(200, verbose=False),
        lgb.log_evaluation(period=500),
    ],
)

print(f"LightGBM Quantile best iteration: {lgb_quantile.best_iteration_}")

val_pred_quantile = np.clip(lgb_quantile.predict(X_val), 0, None)
test_pred_quantile = np.clip(lgb_quantile.predict(X_test), 0, None)

rmse_quantile = root_mean_squared_error(y_val, val_pred_quantile)
print(f"LightGBM Quantile RMSE: {rmse_quantile:.4f}")

# ============================================================
# 9. MODEL 3: LightGBM Tweedie with deeper trees, lower learning rate
# ============================================================
print("\n=== Training LightGBM Tweedie (deeper, lower LR) ===")

lgb_tweedie_deep = lgb.LGBMRegressor(
    objective="tweedie",
    tweedie_variance_power=1.5,
    n_estimators=5000,
    learning_rate=0.02,
    max_depth=12,
    num_leaves=255,
    min_child_samples=50,
    subsample=0.7,
    subsample_freq=1,
    colsample_bytree=0.5,
    reg_alpha=1.0,
    reg_lambda=2.0,
    min_split_gain=0.01,
    random_state=123,
    verbose=-1,
    n_jobs=-1,
)

lgb_tweedie_deep.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    callbacks=[
        lgb.early_stopping(200, verbose=False),
        lgb.log_evaluation(period=500),
    ],
)

print(f"LightGBM Tweedie Deep best iteration: {lgb_tweedie_deep.best_iteration_}")

val_pred_tweedie_deep = np.clip(lgb_tweedie_deep.predict(X_val), 0, None)
test_pred_tweedie_deep = np.clip(lgb_tweedie_deep.predict(X_test), 0, None)

rmse_tweedie_deep = root_mean_squared_error(y_val, val_pred_tweedie_deep)
print(f"LightGBM Tweedie Deep RMSE: {rmse_tweedie_deep:.4f}")

# ============================================================
# 10. BLEND: Simple average of top-2 models
# ============================================================
print("\n=== Simple Average Blend of Top-2 Models ===")

# Collect all model predictions
model_results = [
    ("LGB_Tweedie", rmse_tweedie, val_pred_tweedie, test_pred_tweedie),
    ("LGB_Quantile", rmse_quantile, val_pred_quantile, test_pred_quantile),
    ("LGB_Tweedie_Deep", rmse_tweedie_deep, val_pred_tweedie_deep, test_pred_tweedie_deep),
]

# Sort by validation RMSE (lower is better)
model_results.sort(key=lambda x: x[1])

print("Model rankings:")
for name, rmse, _, _ in model_results:
    print(f"  {name}: RMSE = {rmse:.4f}")

# Take top-2 models and average their predictions
top2 = model_results[:2]
print(f"\nTop-2 models for blending: {top2[0][0]}, {top2[1][0]}")

val_pred_final = np.clip((top2[0][2] + top2[1][2]) / 2.0, 0, None)
test_pred_final = np.clip((top2[0][3] + top2[1][3]) / 2.0, 0, None)

final_rmse = root_mean_squared_error(y_val, val_pred_final)
print(f"Final Blend RMSE (top-2 average): {final_rmse:.4f}")

# ============================================================
# 11. SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame(
    {"id": range(len(test_processed)), "time_to_approve": test_pred_final}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(submission.head())

# Save model artifacts
lgb_tweedie.booster_.save_model("./working/lgb_tweedie.txt")
lgb_quantile.booster_.save_model("./working/lgb_quantile.txt")
lgb_tweedie_deep.booster_.save_model("./working/lgb_tweedie_deep.txt")

print(f"\nValidation Summary:")
print(f"  LightGBM Tweedie RMSE:       {rmse_tweedie:.4f}")
print(f"  LightGBM Quantile RMSE:      {rmse_quantile:.4f}")
print(f"  LightGBM Tweedie Deep RMSE:  {rmse_tweedie_deep:.4f}")
print(f"  Blend RMSE (top-2 avg):      {final_rmse:.4f}")
print(f"  Top-2 models: {top2[0][0]}, {top2[1][0]}")

score = final_rmse
print(f"Final Validation Score: {score}")