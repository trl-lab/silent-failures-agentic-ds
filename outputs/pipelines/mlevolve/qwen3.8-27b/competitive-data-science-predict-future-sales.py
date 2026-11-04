import pandas as pd
import numpy as np
import os
import json
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(
    f"Train date_block_num range: {train['date_block_num'].min()} - {train['date_block_num'].max()}"
)
print(
    f"Test date_block_num range: {test['date_block_num'].min()} - {test['date_block_num'].max()}"
)

# ============================================================
# 2. CLEANING
# ============================================================
train["item_cnt_day"] = train["item_cnt_day"].clip(lower=0)
train["item_price"] = train["item_price"].clip(lower=0, upper=10000)

# Create date features
train["year"] = 2013 + train["date_block_num"] // 12
train["month"] = train["date_block_num"] % 12
train["day_of_week_approx"] = train["date_block_num"] % 7

test["year"] = 2013 + test["date_block_num"] // 12
test["month"] = test["date_block_num"] % 12
test["day_of_week_approx"] = test["date_block_num"] % 7

# ============================================================
# 3. VALIDATION SPLIT (by time - last 2 date_block_num)
# ============================================================
max_block = train["date_block_num"].max()
val_blocks = [max_block - 1, max_block]  # blocks 32 and 33
train_mask = ~train["date_block_num"].isin(val_blocks)
val_mask = train["date_block_num"].isin(val_blocks)

train_split = train[train_mask].copy()
val_split = train[val_mask].copy()

print(
    f"Train split: {len(train_split)} rows, blocks {train_split['date_block_num'].min()}-{train_split['date_block_num'].max()}"
)
print(
    f"Val split: {len(val_split)} rows, blocks {val_split['date_block_num'].min()}-{val_split['date_block_num'].max()}"
)

# ============================================================
# 4. FEATURE ENGINEERING (fit only on train_split, time-aware)
# ============================================================


def compute_time_aware_features(df, full_train):
    """
    Compute time-series aware features for df.
    Rolling stats are computed using only data from blocks BEFORE the target block.
    full_train: the complete training data (used to compute rolling/lag features).
    """
    df = df.copy()

    # --- Item-level features (global aggregates from full_train) ---
    item_agg = (
        full_train.groupby("item_id")["item_cnt_day"]
        .agg(
            item_mean_sales="mean",
            item_std_sales="std",
            item_max_sales="max",
            item_min_sales="min",
            item_total_sales="sum",
            item_n_days="count",
            item_median_sales="median",
        )
        .reset_index()
    )
    item_agg["item_std_sales"] = item_agg["item_std_sales"].fillna(0)
    item_agg["item_avg_per_day"] = (
        item_agg["item_total_sales"] / item_agg["item_n_days"]
    )
    df = df.merge(item_agg, on="item_id", how="left")

    # --- Shop-level features ---
    shop_agg = (
        full_train.groupby("shop_id")["item_cnt_day"]
        .agg(
            shop_mean_sales="mean",
            shop_std_sales="std",
            shop_max_sales="max",
            shop_total_sales="sum",
            shop_n_items="nunique",
        )
        .reset_index()
    )
    shop_agg["shop_std_sales"] = shop_agg["shop_std_sales"].fillna(0)
    shop_agg["shop_avg_per_item"] = (
        shop_agg["shop_total_sales"] / shop_agg["shop_n_items"]
    )
    df = df.merge(shop_agg, on="shop_id", how="left")

    # --- Shop-Item interaction features (global) ---
    shop_item_agg = (
        full_train.groupby(["shop_id", "item_id"])["item_cnt_day"]
        .agg(
            si_mean_sales="mean",
            si_std_sales="std",
            si_max_sales="max",
            si_total_sales="sum",
            si_n_days="count",
            si_median_sales="median",
        )
        .reset_index()
    )
    shop_item_agg["si_std_sales"] = shop_item_agg["si_std_sales"].fillna(0)
    shop_item_agg["si_avg_per_day"] = (
        shop_item_agg["si_total_sales"] / shop_item_agg["si_n_days"]
    )
    df = df.merge(shop_item_agg, on=["shop_id", "item_id"], how="left")

    # Fill NaN for shop-item combinations not seen in training
    for col in [
        "si_mean_sales",
        "si_std_sales",
        "si_max_sales",
        "si_total_sales",
        "si_n_days",
        "si_median_sales",
        "si_avg_per_day",
    ]:
        df[col] = df[col].fillna(0)

    # --- Price features ---
    df["log_price"] = np.log1p(df["item_price"].clip(lower=0))
    df["price_zscore"] = (df["item_price"] - full_train["item_price"].mean()) / (
        full_train["item_price"].std() + 1e-8
    )

    # --- Temporal features ---
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["block_num"] = df["date_block_num"]

    # --- Interaction features ---
    df["shop_item_mean_ratio"] = df["si_mean_sales"] / (df["item_mean_sales"] + 1e-8)
    df["item_shop_mean_ratio"] = df["si_mean_sales"] / (df["shop_mean_sales"] + 1e-8)

    # --- TIME-SERIES AWARE FEATURES ---
    # Compute rolling stats and lags per shop-item, using only data before target block
    # For each row in df, we need stats from blocks < current block

    # Pre-compute per-block aggregates for shop-item pairs in full_train
    # This is efficient: group by (shop_id, item_id, date_block_num)
    si_block = (
        full_train.groupby(["shop_id", "item_id", "date_block_num"])["item_cnt_day"]
        .mean()
        .reset_index()
        .rename(columns={"item_cnt_day": "si_block_mean"})
    )

    # Pivot to wide format: rows = (shop_id, item_id), cols = block
    si_wide = si_block.pivot_table(
        index=["shop_id", "item_id"], columns="date_block_num", values="si_block_mean"
    )
    si_wide = si_wide.reset_index()

    # For each target block, compute rolling features
    # We'll process each unique block in df
    target_blocks = sorted(df["date_block_num"].unique())

    # Pre-compute rolling features for all possible target blocks
    # For target block B, we use blocks 0..B-1
    rolling_results = []
    max_available_block = full_train["date_block_num"].max()

    for target_block in target_blocks:
        # Get data from blocks before target_block
        prev_blocks = si_wide.copy()
        if target_block <= 0:
            # No previous data available
            prev_data = pd.DataFrame(columns=["shop_id", "item_id"])
            for c in range(0, max_available_block + 1):
                prev_data[c] = np.nan
        else:
            prev_data = si_wide.copy()
            # Zero out current and future blocks
            cols_to_zero = [c for c in range(target_block, max_available_block + 1) if c in prev_data.columns]
            for c in cols_to_zero:
                prev_data[c] = np.nan

        # Compute rolling means/stds for last 1, 2, 3, 6 months (blocks)
        # Lag features: value at t-1, t-2, t-3
        for lag in [1, 2, 3]:
            col_name = target_block - lag
            if col_name in prev_data.columns:
                prev_data[f"lag_{lag}"] = prev_data[col_name]
            else:
                prev_data[f"lag_{lag}"] = np.nan

        # Rolling mean/std over last 1, 2, 3, 6 blocks
        for window in [1, 2, 3, 6]:
            start_block = max(0, target_block - window)
            window_cols = [
                c
                for c in range(start_block, target_block)
                if c in prev_data.columns
            ]
            if window_cols:
                prev_data[f"roll_mean_{window}"] = prev_data[window_cols].mean(axis=1)
                prev_data[f"roll_std_{window}"] = prev_data[window_cols].std(axis=1)
            else:
                prev_data[f"roll_mean_{window}"] = np.nan
                prev_data[f"roll_std_{window}"] = np.nan

        # Time-decay weighted mean (exponential decay with half-life=3 blocks)
        if target_block > 0:
            decay_cols = [c for c in range(max(0, target_block - 12), target_block) if c in prev_data.columns]
            if decay_cols:
                weights = np.array([0.5 ** ((target_block - 1 - c) / 3.0) for c in decay_cols])
                vals = prev_data[decay_cols].values
                weighted_vals = vals * weights[np.newaxis, :]
                weighted_sums = np.nansum(weighted_vals, axis=1)
                weight_sums = np.nansum(np.where(np.isnan(vals), 0, weights[np.newaxis, :]), axis=1)
                prev_data["time_decay_mean"] = np.where(weight_sums > 0, weighted_sums / weight_sums, np.nan)
            else:
                prev_data["time_decay_mean"] = np.nan
        else:
            prev_data["time_decay_mean"] = np.nan

        # Item popularity rank within shop (from full_train up to target_block)
        shop_item_rank_data = full_train[full_train["date_block_num"] < target_block]
        if len(shop_item_rank_data) > 0:
            shop_item_sales = (
                shop_item_rank_data.groupby(["shop_id", "item_id"])["item_cnt_day"]
                .sum()
                .reset_index()
                .rename(columns={"item_cnt_day": "si_sales_rank"})
            )
            shop_item_sales["item_rank_in_shop"] = (
                shop_item_sales.groupby("shop_id")["si_sales_rank"]
                .rank(ascending=False, method="dense")
            )
            shop_item_sales = shop_item_sales[["shop_id", "item_id", "item_rank_in_shop"]]
            prev_data = prev_data.merge(shop_item_sales, on=["shop_id", "item_id"], how="left")
        else:
            prev_data["item_rank_in_shop"] = np.nan

        rolling_results.append(prev_data)

    # Merge rolling features back to df
    # For each block in df, merge the corresponding rolling features
    for i, target_block in enumerate(target_blocks):
        mask = df["date_block_num"] == target_block
        rolling_df = rolling_results[i][["shop_id", "item_id"] + [
            "lag_1", "lag_2", "lag_3",
            "roll_mean_1", "roll_mean_2", "roll_mean_3", "roll_mean_6",
            "roll_std_1", "roll_std_2", "roll_std_3", "roll_std_6",
            "time_decay_mean", "item_rank_in_shop"
        ]]
        # Merge into df for rows with this block
        df_idx = df[mask].index
        merged = df.loc[df_idx, ["shop_id", "item_id"]].merge(
            rolling_df, on=["shop_id", "item_id"], how="left"
        )
        for col in merged.columns:
            if col not in ["shop_id", "item_id"]:
                df.loc[df_idx, col] = merged[col].values

    # Fill NaNs in rolling features
    rolling_cols = [
        "lag_1", "lag_2", "lag_3",
        "roll_mean_1", "roll_mean_2", "roll_mean_3", "roll_mean_6",
        "roll_std_1", "roll_std_2", "roll_std_3", "roll_std_6",
        "time_decay_mean", "item_rank_in_shop",
    ]
    for col in rolling_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


# Apply time-aware features
# For train_split: use train_split itself as the reference (features computed from earlier blocks)
# For val_split: use train_split as reference (no leakage - val blocks are after train)
# For test: use train_split as reference

train_feat = compute_time_aware_features(train_split, train_split)
val_feat = compute_time_aware_features(val_split, train_split)
test_feat = compute_time_aware_features(test, train_split)

# ============================================================
# 5. DEFINE FEATURE COLUMNS
# ============================================================
feature_cols = [
    "date_block_num",
    "year",
    "month",
    "day_of_week_approx",
    "month_sin",
    "month_cos",
    "block_num",
    "item_price",
    "log_price",
    "price_zscore",
    "item_mean_sales",
    "item_std_sales",
    "item_max_sales",
    "item_min_sales",
    "item_total_sales",
    "item_n_days",
    "item_median_sales",
    "item_avg_per_day",
    "shop_mean_sales",
    "shop_std_sales",
    "shop_max_sales",
    "shop_total_sales",
    "shop_n_items",
    "shop_avg_per_item",
    "si_mean_sales",
    "si_std_sales",
    "si_max_sales",
    "si_total_sales",
    "si_n_days",
    "si_median_sales",
    "si_avg_per_day",
    "shop_item_mean_ratio",
    "item_shop_mean_ratio",
    # Time-series features
    "lag_1",
    "lag_2",
    "lag_3",
    "roll_mean_1",
    "roll_mean_2",
    "roll_mean_3",
    "roll_mean_6",
    "roll_std_1",
    "roll_std_2",
    "roll_std_3",
    "roll_std_6",
    "time_decay_mean",
    "item_rank_in_shop",
]

# Ensure all feature columns exist
for col in feature_cols:
    if col not in train_feat.columns:
        train_feat[col] = 0.0
        val_feat[col] = 0.0
        test_feat[col] = 0.0
    if col not in val_feat.columns:
        val_feat[col] = 0.0
    if col not in test_feat.columns:
        test_feat[col] = 0.0

# Replace any remaining NaN/inf
for df in [train_feat, val_feat, test_feat]:
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"\nFeature matrix shapes:")
print(f"  Train: {train_feat[feature_cols].shape}")
print(f"  Val:   {val_feat[feature_cols].shape}")
print(f"  Test:  {test_feat[feature_cols].shape}")

# ============================================================
# 6. PREPARE MATRICES FOR MODEL
# ============================================================
X_train = train_feat[feature_cols].values.astype(np.float32)
y_train = train_feat["item_cnt_day"].values.astype(np.float32)
X_val = val_feat[feature_cols].values.astype(np.float32)
y_val = val_feat["item_cnt_day"].values.astype(np.float32)
X_test = test_feat[feature_cols].values.astype(np.float32)

print(f"\nTraining data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Validation data: {X_val.shape[0]} samples")
print(f"Test data: {X_test.shape[0]} samples")
print(
    f"Target stats - train: mean={y_train.mean():.4f}, std={y_train.std():.4f}, max={y_train.max():.1f}"
)
print(
    f"Target stats - val:   mean={y_val.mean():.4f}, std={y_val.std():.4f}, max={y_val.max():.1f}"
)

# ============================================================
# 7. DEFINE MODELS: Two-stage ensemble
# ============================================================
# Model 1: LightGBM with MSE (regression_l2), low lr, more trees, reduced complexity
model1 = lgb.LGBMRegressor(
    objective="regression_l2",
    n_estimators=10000,
    learning_rate=0.02,
    num_leaves=127,
    max_depth=-1,
    min_child_samples=500,
    subsample=0.8,
    subsample_freq=5,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

# Model 2: LightGBM with Huber loss (alpha=2.0) for robustness to outliers
model2 = lgb.LGBMRegressor(
    objective="huber",
    huber_delta=2.0,
    n_estimators=10000,
    learning_rate=0.02,
    num_leaves=127,
    max_depth=-1,
    min_child_samples=500,
    subsample=0.8,
    subsample_freq=5,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

# ============================================================
# 8. TRAIN BOTH MODELS WITH EARLY STOPPING (patience=500)
# ============================================================
print("\n--- Training Model 1: LightGBM MSE (lr=0.02, leaves=127) ---")

callbacks1 = [
    lgb.early_stopping(stopping_rounds=500, verbose=False),
    lgb.log_evaluation(period=500),
]

model1.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    callbacks=callbacks1,
)

best_iter1 = model1.best_iteration_
best_val_rmse1 = model1.best_score_["valid_0"]["rmse"]
print(f"Model 1 - Best iteration: {best_iter1}, Best val RMSE: {best_val_rmse1:.4f}")

print("\n--- Training Model 2: LightGBM Huber (alpha=2.0, lr=0.02, leaves=127) ---")

callbacks2 = [
    lgb.early_stopping(stopping_rounds=500, verbose=False),
    lgb.log_evaluation(period=500),
]

model2.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    callbacks=callbacks2,
)

best_iter2 = model2.best_iteration_
best_val_rmse2 = model2.best_score_["valid_0"]["rmse"]
print(f"Model 2 - Best iteration: {best_iter2}, Best val RMSE: {best_val_rmse2:.4f}")

# ============================================================
# 9. SAVE MODELS
# ============================================================
os.makedirs("./working", exist_ok=True)
model1.booster_.save_model("./working/lgbm_model1_mse.txt")
model2.booster_.save_model("./working/lgbm_model2_huber.txt")
print("Models saved to ./working/")

# ============================================================
# 10. VALIDATION PREDICTIONS (ensemble with equal weights)
# ============================================================
val_pred1_raw = model1.predict(X_val, num_iteration=best_iter1)
val_pred2_raw = model2.predict(X_val, num_iteration=best_iter2)

# Ensemble: equal weights
val_pred_raw = 0.5 * val_pred1_raw + 0.5 * val_pred2_raw
val_pred = np.clip(val_pred_raw, 0, None)

val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"\nValidation RMSE (ensemble, clipped >= 0): {val_rmse:.4f}")

# Individual model validation scores
val_pred1 = np.clip(val_pred1_raw, 0, None)
val_pred2 = np.clip(val_pred2_raw, 0, None)
print(f"  Model 1 (MSE) val RMSE: {np.sqrt(mean_squared_error(y_val, val_pred1)):.4f}")
print(f"  Model 2 (Huber) val RMSE: {np.sqrt(mean_squared_error(y_val, val_pred2)):.4f}")

val_rmse_raw = np.sqrt(mean_squared_error(y_val, val_pred_raw))
print(f"Validation RMSE (raw, unclipped): {val_rmse_raw:.4f}")

# ============================================================
# 11. RETRAIN ON ALL DATA FOR FINAL SUBMISSION
# ============================================================
print("\n--- Retraining on ALL data (blocks 0-33) for final submission ---")

# Recompute features on full train data
full_train_feat = compute_time_aware_features(train, train)

X_full = full_train_feat[feature_cols].values.astype(np.float32)
y_full = full_train_feat["item_cnt_day"].values.astype(np.float32)

print(f"Full training data: {X_full.shape[0]} samples, {X_full.shape[1]} features")

# Retrain model 1 on all data with best iteration
model1_final = lgb.LGBMRegressor(
    objective="regression_l2",
    n_estimators=best_iter1,
    learning_rate=0.02,
    num_leaves=127,
    max_depth=-1,
    min_child_samples=500,
    subsample=0.8,
    subsample_freq=5,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
model1_final.fit(X_full, y_full)

# Retrain model 2 on all data with best iteration
model2_final = lgb.LGBMRegressor(
    objective="huber",
    huber_delta=2.0,
    n_estimators=best_iter2,
    learning_rate=0.02,
    num_leaves=127,
    max_depth=-1,
    min_child_samples=500,
    subsample=0.8,
    subsample_freq=5,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
model2_final.fit(X_full, y_full)

print("Final models trained on all data.")

# ============================================================
# 12. TEST PREDICTIONS (ensemble, same post-processing)
# ============================================================
test_pred1 = model1_final.predict(X_test)
test_pred2 = model2_final.predict(X_test)
test_pred_raw = 0.5 * test_pred1 + 0.5 * test_pred2
test_pred = np.clip(test_pred_raw, 0, None)

print(
    f"\nTest prediction stats: mean={test_pred.mean():.4f}, std={test_pred.std():.4f}, max={test_pred.max():.2f}"
)

# ============================================================
# 13. GENERATE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)

submission = pd.DataFrame({"id": range(len(test_pred)), "item_cnt_day": test_pred})
submission.to_csv("./submission/submission.csv", index=False)
print(
    f"Submission saved: {submission.shape[0]} rows, columns: {submission.columns.tolist()}"
)
print(f"Submission head:\n{submission.head()}")

# ============================================================
# 14. FEATURE IMPORTANCE (top 10) - from model 1
# ============================================================
fi = pd.Series(model1.feature_importances_, index=feature_cols).sort_values(
    ascending=False
)
print("\nTop 10 Feature Importances (Model 1 - MSE):")
for name, imp in fi.head(10).items():
    print(f"  {name}: {imp}")

# ============================================================
# 15. FINAL SCORE
# ============================================================
score = val_rmse
print(f"\n{'='*50}")
print(f"Model: Ensemble of LightGBM MSE + Huber (lr=0.02, leaves=127)")
print(f"Model 1 best iteration: {best_iter1}, Model 2 best iteration: {best_iter2}")
print(f"Validation split: last 2 date_block_num (blocks 32-33)")
print(f"Final models retrained on ALL data (blocks 0-33)")
print(f"Metric: RMSE (official)")
print(f"Post-processing: clip predictions to >= 0")
print(f"{'='*50}")
print(f"Final Validation Score: {score}")
