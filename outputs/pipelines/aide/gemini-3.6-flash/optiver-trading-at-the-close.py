import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Filter rows with missing target in train
train_df = train_df.dropna(subset=["target"]).reset_index(drop=True)

train_df["is_test"] = 0
test_df["is_test"] = 1
if "target" not in test_df.columns:
    test_df["target"] = np.nan

# Combine train and test for temporal and cross-sectional feature engineering
df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# Downcast float precision to conserve memory
float_cols = df.select_dtypes(include=["float64"]).columns
df[float_cols] = df[float_cols].astype(np.float32)

# Ensure correct sorting for feature calculations
df = df.sort_values(by=["date_id", "seconds_in_bucket", "stock_id"]).reset_index(
    drop=True
)

# 2. Microstructure Features
df["bid_ask_spread"] = df["ask_price"] - df["bid_price"]
df["bid_ask_size_imbalance"] = (df["ask_size"] - df["bid_size"]) / (
    df["ask_size"] + df["bid_size"] + 1e-8
)
df["matched_imbalance_ratio"] = df["imbalance_size"] / (df["matched_size"] + 1e-8)
df["reference_wap_spread"] = df["reference_price"] - df["wap"]
df["signed_imbalance"] = df["imbalance_size"] * df["imbalance_buy_sell_flag"]
df["far_near_spread"] = df["far_price"] - df["near_price"]
df["far_wap_spread"] = df["far_price"] - df["wap"]
df["near_wap_spread"] = df["near_price"] - df["wap"]

# 3. Stock-level Aggregates (computed only on training subset to avoid leakage)
stock_stats = (
    df[df["is_test"] == 0]
    .groupby("stock_id")
    .agg(
        stock_mean_matched=("matched_size", "mean"),
        stock_mean_bid_size=("bid_size", "mean"),
        stock_mean_ask_size=("ask_size", "mean"),
    )
    .reset_index()
)

df = df.merge(stock_stats, on="stock_id", how="left")
df["matched_size_norm"] = df["matched_size"] / (df["stock_mean_matched"] + 1e-8)
df["bid_size_norm"] = df["bid_size"] / (df["stock_mean_bid_size"] + 1e-8)
df["ask_size_norm"] = df["ask_size"] / (df["stock_mean_ask_size"] + 1e-8)
df.drop(
    columns=["stock_mean_matched", "stock_mean_bid_size", "stock_mean_ask_size"],
    inplace=True,
)

# 4. Market Cross-Sectional Aggregations per time bucket
market_stats = (
    df.groupby(["date_id", "seconds_in_bucket"])
    .agg(
        market_wap_mean=("wap", "mean"),
        market_wap_std=("wap", "std"),
        market_signed_imbalance=("signed_imbalance", "sum"),
        market_matched_size=("matched_size", "sum"),
    )
    .reset_index()
)

df = df.merge(market_stats, on=["date_id", "seconds_in_bucket"], how="left")
df["wap_market_diff"] = df["wap"] - df["market_wap_mean"]

# 5. Multi-Scale Temporal Lags, Differences, and Market-Relative Returns per (stock_id, date_id)
df = df.sort_values(by=["stock_id", "date_id", "seconds_in_bucket"]).reset_index(
    drop=True
)

group_cols = ["stock_id", "date_id"]
lags = [1, 3, 6]

for lag in lags:
    df[f"wap_lag_{lag}"] = df.groupby(group_cols)["wap"].shift(lag)
    df[f"wap_diff_{lag}"] = df["wap"] - df[f"wap_lag_{lag}"]
    df[f"wap_ret_{lag}"] = (df["wap"] - df[f"wap_lag_{lag}"]) / (
        df[f"wap_lag_{lag}"] + 1e-8
    )

    df[f"spread_lag_{lag}"] = df.groupby(group_cols)["bid_ask_spread"].shift(lag)
    df[f"spread_diff_{lag}"] = df["bid_ask_spread"] - df[f"spread_lag_{lag}"]

    df[f"signed_imbalance_lag_{lag}"] = df.groupby(group_cols)[
        "signed_imbalance"
    ].shift(lag)
    df[f"signed_imbalance_diff_{lag}"] = (
        df["signed_imbalance"] - df[f"signed_imbalance_lag_{lag}"]
    )

# Compute market-relative returns (cross-sectional mean de-meaned returns per timestamp)
time_group = ["date_id", "seconds_in_bucket"]
for lag in lags:
    market_ret = df.groupby(time_group)[f"wap_ret_{lag}"].transform("mean")
    df[f"wap_relative_ret_{lag}"] = df[f"wap_ret_{lag}"] - market_ret

# Separate back into train and test
train_processed = df[df["is_test"] == 0].copy()
test_processed = df[df["is_test"] == 1].copy()

# Sort test dataset to maintain consistent index mapping
test_processed = test_processed.sort_values(
    by=["date_id", "seconds_in_bucket", "stock_id"]
).reset_index(drop=True)

# Define feature set
drop_cols = ["target", "is_test", "row_id", "time_id"]
feature_cols = [c for c in train_processed.columns if c not in drop_cols]

X = train_processed[feature_cols]
y = train_processed["target"]
groups = train_processed["date_id"]

X_test = test_processed[feature_cols]

# 6. Model Training & Evaluation with GroupKFold
gkf = GroupKFold(n_splits=5)

oof_preds = np.zeros(len(train_processed))
test_preds = np.zeros(len(test_processed))

lgb_params = {
    "objective": "regression_l1",
    "metric": "mae",
    "boosting_type": "gbdt",
    "n_estimators": 1200,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
    )

    val_pred = model.predict(X_val)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(X_test) / gkf.n_splits

cv_mae = mean_absolute_error(y, oof_preds)
print(f"Validation Metric (MAE): {cv_mae:.5f}")

# 7. Save submission file
os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame({"id": np.arange(len(test_preds)), "target": test_preds})
sub.to_csv("./working/submission.csv", index=False)
