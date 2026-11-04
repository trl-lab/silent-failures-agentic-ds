import numpy as np
import pandas as pd
import lightgbm as lgb
import os
from sklearn.metrics import mean_absolute_error

# Create directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("Loading data...")
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# ============================================================
# 2. IDENTIFY TARGET AND HANDLE NaN TARGETS
# ============================================================
target_col = "target"
train_has_target = train[target_col].notna()

print(f"Train rows with target: {train_has_target.sum()}")
print(f"Train rows without target: {(~train_has_target).sum()}")

# ============================================================
# 3. TIME-BASED SPLIT (last 10% of dates as validation)
# ============================================================
all_dates = sorted(train["date_id"].unique())
n_dates = len(all_dates)
n_val_dates = max(1, n_dates // 10)
val_dates = set(all_dates[-n_val_dates:])
train_dates = set(all_dates[:-n_val_dates])

print(
    f"Total dates: {n_dates}, Train dates: {len(train_dates)}, Val dates: {len(val_dates)}"
)

train_mask = train["date_id"].isin(train_dates) & train_has_target
val_mask = train["date_id"].isin(val_dates) & train_has_target

df_train = train[train_mask].copy()
df_val = train[val_mask].copy()

print(f"df_train shape: {df_train.shape}")
print(f"df_val shape: {df_val.shape}")

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================


def engineer_features(df):
    """Engineer features for the trading data."""
    df = df.copy()

    # --- Basic microstructure features ---
    df["spread"] = df["ask_price"] - df["bid_price"]
    df["mid_price"] = (df["ask_price"] + df["bid_price"]) / 2.0
    df["wap_deviation"] = df["wap"] - df["mid_price"]
    df["ref_deviation"] = df["reference_price"] - df["mid_price"]
    df["near_deviation"] = df["near_price"] - df["mid_price"]
    df["far_deviation"] = df["far_price"] - df["mid_price"]
    df["price_vs_ref"] = df["wap"] / df["reference_price"] - 1.0

    # --- Size features (log transform for heavy tails) ---
    df["log_ask_size"] = np.log1p(df["ask_size"].clip(lower=0))
    df["log_bid_size"] = np.log1p(df["bid_size"].clip(lower=0))
    df["log_matched_size"] = np.log1p(df["matched_size"].clip(lower=0))
    df["log_imbalance_size"] = np.log1p(df["imbalance_size"].clip(lower=0))

    # Size ratio
    total_size = (df["ask_size"] + df["bid_size"]).clip(lower=1)
    df["size_ratio"] = (df["ask_size"] - df["bid_size"]) / total_size

    # --- Imbalance features ---
    df["imbalance_weighted"] = df["imbalance_buy_sell_flag"] * np.log1p(
        df["imbalance_size"].clip(lower=0)
    )

    # --- Temporal features ---
    df["time_progress"] = df["seconds_in_bucket"] / 540.0
    df["time_sin"] = np.sin(2 * np.pi * df["seconds_in_bucket"] / 540.0)
    df["time_cos"] = np.cos(2 * np.pi * df["seconds_in_bucket"] / 540.0)
    df["near_close"] = (df["seconds_in_bucket"] > 450).astype(float)
    df["very_near_close"] = (df["seconds_in_bucket"] > 500).astype(float)

    # --- Group-level features (within date_id + stock_id + time_id) ---
    group_cols = ["date_id", "stock_id", "time_id"]

    for col in ["wap", "mid_price", "spread", "matched_size", "imbalance_size"]:
        if col in df.columns:
            grp = df.groupby(group_cols)[col]
            df[f"{col}_group_mean"] = grp.transform("mean")
            df[f"{col}_group_std"] = grp.transform("std").fillna(0)
            df[f"{col}_group_min"] = grp.transform("min")
            df[f"{col}_group_max"] = grp.transform("max")
            df[f"{col}_group_range"] = df[f"{col}_group_max"] - df[f"{col}_group_min"]
            df[f"{col}_group_rank"] = grp.rank(pct=True)

    # Within-group: deviation from group mean
    df["wap_dev_from_group"] = df["wap"] - df["wap_group_mean"]
    df["spread_dev_from_group"] = df["spread"] - df["spread_group_mean"]

    # --- Stock-level features (aggregated across all times for that stock on that day) ---
    stock_day_cols = ["date_id", "stock_id"]

    for col in ["wap", "spread", "matched_size"]:
        if col in df.columns:
            grp = df.groupby(stock_day_cols)[col]
            df[f"{col}_stockday_mean"] = grp.transform("mean")
            df[f"{col}_stockday_std"] = grp.transform("std").fillna(0)

    df["wap_vs_stockday"] = df["wap"] - df["wap_stockday_mean"]

    # --- Cross-sectional features (within same date_id and seconds_in_bucket) ---
    cross_cols = ["date_id", "seconds_in_bucket"]

    for col in ["wap", "spread"]:
        if col in df.columns:
            grp = df.groupby(cross_cols)[col]
            df[f"{col}_cross_mean"] = grp.transform("mean")
            df[f"{col}_cross_std"] = grp.transform("std").fillna(0)
            df[f"{col}_cross_zscore"] = (df[col] - df[f"{col}_cross_mean"]) / (
                df[f"{col}_cross_std"] + 1e-8
            )

    # --- Interaction features ---
    df["spread_x_imbalance"] = df["spread"] * df["imbalance_buy_sell_flag"]
    df["size_ratio_x_imbalance"] = df["size_ratio"] * df["imbalance_buy_sell_flag"]
    df["time_x_spread"] = df["time_progress"] * df["spread"]
    df["time_x_wap_dev"] = df["time_progress"] * df["wap_deviation"]

    # --- Price level features ---
    df["ask_vs_ref"] = df["ask_price"] / df["reference_price"] - 1.0
    df["bid_vs_ref"] = df["bid_price"] / df["reference_price"] - 1.0
    df["wap_vs_ref"] = df["wap"] / df["reference_price"] - 1.0

    # --- Handle remaining NaNs ---
    if "near_price" in df.columns:
        df["near_price"] = df["near_price"].fillna(df["mid_price"])
        df["near_deviation"] = df["near_price"] - df["mid_price"]
    if "far_price" in df.columns:
        df["far_price"] = df["far_price"].fillna(df["mid_price"])
        df["far_deviation"] = df["far_price"] - df["mid_price"]

    # Fill any remaining NaNs
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(0)

    return df


# Apply feature engineering
print("Engineering features for train...")
df_train = engineer_features(df_train)

print("Engineering features for validation...")
df_val = engineer_features(df_val)

print("Engineering features for test...")
df_test = engineer_features(test)

# ============================================================
# 5. DEFINE FEATURE COLUMNS
# ============================================================
exclude_cols = ["target", "row_id"]
common_cols = set(df_train.columns) & set(df_test.columns)
feature_cols = sorted([c for c in common_cols if c not in exclude_cols])

print(f"Number of features: {len(feature_cols)}")

# ============================================================
# 6. PREPARE MATRICES
# ============================================================
X_train = df_train[feature_cols].values.astype(np.float32)
y_train = df_train[target_col].values.astype(np.float32)

X_val = df_val[feature_cols].values.astype(np.float32)
y_val = df_val[target_col].values.astype(np.float32)

X_test = df_test[feature_cols].values.astype(np.float32)

print(f"\nFinal shapes:")
print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"  X_test: {X_test.shape}")

# ============================================================
# 7. MODEL PARAMETERS
# ============================================================
params = {
    "objective": "mae",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 200,
    "max_depth": -1,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "num_threads": -1,
    "verbose": -1,
    "seed": 42,
}

# ============================================================
# 8. TRAIN MODEL WITH EARLY STOPPING
# ============================================================
print("\nTraining LightGBM model...")

train_data = lgb.Dataset(
    X_train, label=y_train, feature_name=feature_cols, free_raw_data=False
)
val_data = lgb.Dataset(
    X_val,
    label=y_val,
    feature_name=feature_cols,
    reference=train_data,
    free_raw_data=False,
)

model = lgb.train(
    params,
    train_data,
    num_boost_round=3000,
    valid_sets=[train_data, val_data],
    valid_names=["train", "valid"],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)],
)

print(f"Best iteration: {model.best_iteration}")
print(f"Best valid MAE (from LGBM): {model.best_score['valid']['l1']:.4f}")

# ============================================================
# 9. VALIDATION PREDICTIONS
# ============================================================
val_pred = model.predict(X_val, num_iteration=model.best_iteration)
val_mae = mean_absolute_error(y_val, val_pred)
print(f"Validation MAE (sklearn): {val_mae:.4f}")

# ============================================================
# 10. TEST PREDICTIONS
# ============================================================
test_pred = model.predict(X_test, num_iteration=model.best_iteration)
print(f"Test predictions shape: {test_pred.shape}")
print(f"Test pred stats: mean={test_pred.mean():.4f}, std={test_pred.std():.4f}")

# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================
importance = model.feature_importance(importance_type="gain")
feat_imp = pd.DataFrame(
    {"feature": feature_cols, "importance": importance}
).sort_values("importance", ascending=False)
print("\nTop 15 features by gain:")
print(feat_imp.head(15).to_string(index=False))

# ============================================================
# 12. SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame({"id": np.arange(len(test_pred)), "target": test_pred})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(submission.head())

# ============================================================
# 13. SAVE MODEL
# ============================================================
model.save_model("./working/lgbm_model.txt")
print("Model saved to ./working/lgbm_model.txt")

# ============================================================
# FINAL SCORE
# ============================================================
score = val_mae
print(f"Final Validation Score: {score}")
