import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import os
import warnings

warnings.filterwarnings("ignore")

os.makedirs("./working", exist_ok=True)

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Drop rows with NaN target
train = train.dropna(subset=["target"]).reset_index(drop=True)


# Feature engineering
def add_features(df):
    df = df.copy()
    df["spread"] = df["ask_price"] - df["bid_price"]
    df["mid_price"] = (df["ask_price"] + df["bid_price"]) / 2
    df["log_ask_size"] = np.log1p(df["ask_size"])
    df["log_bid_size"] = np.log1p(df["bid_size"])
    df["log_total_size"] = np.log1p(df["ask_size"] + df["bid_size"])
    df["size_ratio"] = df["ask_size"] / (df["bid_size"] + 1)
    df["price_diff_ref"] = df["mid_price"] - df["reference_price"]
    df["price_diff_near"] = df["mid_price"] - df["near_price"]
    df["price_diff_far"] = df["mid_price"] - df["far_price"]
    df["imbalance_size_log"] = np.log1p(df["imbalance_size"])
    df["matched_size_log"] = np.log1p(df["matched_size"])
    df["wap_diff"] = df["wap"] - df["mid_price"]
    df["seconds_sin"] = np.sin(2 * np.pi * df["seconds_in_bucket"] / 540)
    df["seconds_cos"] = np.cos(2 * np.pi * df["seconds_in_bucket"] / 540)
    return df


train = add_features(train)
test = add_features(test)

# Define features
drop_cols = ["row_id", "target"]
feature_cols = [c for c in train.columns if c not in drop_cols]
print(f"Number of features: {len(feature_cols)}")

# Time-based validation split: use last 10% of unique dates
unique_dates = np.sort(train["date_id"].unique())
val_dates = unique_dates[int(0.9 * len(unique_dates)) :]
print(f"Validation dates: {val_dates[:5]}...{val_dates[-5:]} ({len(val_dates)} dates)")

train_mask = ~train["date_id"].isin(val_dates)
X_train, y_train = train[train_mask][feature_cols], train[train_mask]["target"]
X_val, y_val = train[~train_mask][feature_cols], train[~train_mask]["target"]

print(f"Train split: {len(X_train)}, Val split: {len(X_val)}")

# LightGBM parameters
params = {
    "objective": "mae",
    "metric": "mae",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 255,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "num_threads": -1,
    "seed": 42,
}

# Train with early stopping
dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=dtrain)

model = lgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    valid_sets=[dval],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)

val_pred = model.predict(X_val, num_iteration=model.best_iteration)
val_mae = mean_absolute_error(y_val, val_pred)
print(f"\nValidation MAE: {val_mae:.6f}")
print(f"Best iteration: {model.best_iteration}")

# Retrain on full training data with best iteration count
dtrain_full = lgb.Dataset(
    train[feature_cols], label=train["target"], feature_name=feature_cols
)
model_full = lgb.train(
    params,
    dtrain_full,
    num_boost_round=model.best_iteration,
    callbacks=[lgb.log_evaluation(-1)],
)

# Predict on test
test_pred = model_full.predict(
    test[feature_cols], num_iteration=model_full.best_iteration
)

# Create submission
submission = pd.DataFrame({"id": np.arange(len(test_pred)), "target": test_pred})
submission.to_csv("./working/submission.csv", index=False)
print(f"\nSubmission saved with {len(submission)} rows")
print(submission.head())
