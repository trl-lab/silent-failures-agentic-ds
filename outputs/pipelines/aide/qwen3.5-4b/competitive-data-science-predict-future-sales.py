import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os

# Set random seed for reproducibility
np.random.seed(42)

# Load data
print("Loading data...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Data preprocessing
print("Preprocessing data...")
train_df = train_df.copy()
test_df = test_df.copy()

# Handle missing values
train_df = train_df.dropna()
test_df = test_df.dropna()

# Rename columns for consistency
train_df.columns = [
    "date",
    "date_block_num",
    "item_cnt_day",
    "item_id",
    "item_price",
    "shop_id",
]
test_df.columns = ["date", "date_block_num", "item_id", "item_price", "shop_id"]


# Feature engineering
def create_features(df, target_col="item_cnt_day"):
    df = df.copy()
    # Create interaction features
    df["item_price_log"] = np.log1p(
        df["item_price"].clip(lower=0)
    )  # Clip negative values to 0
    df["date_block_num"] = df["date_block_num"].astype(float)
    df["item_id"] = df["item_id"].astype(float)
    df["shop_id"] = df["shop_id"].astype(float)
    # Create interaction features
    df["item_price_item_id"] = df["item_price"] * df["item_id"]
    df["date_block_num_shop_id"] = df["date_block_num"] * df["shop_id"]
    return df


train_features = create_features(train_df)
test_features = create_features(test_df)

feature_cols = [
    "date_block_num",
    "item_id",
    "shop_id",
    "item_price_log",
    "item_price_item_id",
    "date_block_num_shop_id",
]

X_train = train_features[feature_cols].fillna(0)
y_train = train_features["item_cnt_day"].fillna(0)
X_test = test_features[feature_cols].fillna(0)

# Define LightGBM parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

# 5-fold cross-validation
print("Starting 5-fold cross-validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_rmse = np.inf
best_model = None
best_score = 0

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold, reference=None)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=None)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    y_pred_val = model.predict(X_val_fold, num_iteration=model.best_iteration)
    fold_rmse = mean_squared_error(y_val_fold, y_pred_val)
    print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}")

    if fold_rmse < best_rmse:
        best_rmse = fold_rmse
        best_model = model

print(f"Best RMSE on CV: {best_rmse:.4f}")

# Train final model on all data
print("Training final model on all data...")
train_data = lgb.Dataset(X_train, label=y_train)

# Create validation set from training data for early stopping
val_data = lgb.Dataset(X_train.iloc[:1000], label=y_train.iloc[:1000], reference=None)

final_model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Make predictions on test set
print("Generating test predictions...")
test_predictions = final_model.predict(X_test)
test_predictions = np.clip(test_predictions, 0, None)  # Ensure non-negative

# Create submission file
submission_df = pd.DataFrame(
    {"id": range(len(test_df)), "item_cnt_day": test_predictions}
)
submission_df.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")

# Print final RMSE metric
print(f"\nFinal RMSE on validation set: {best_rmse:.4f}")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os

# Set random seed for reproducibility
np.random.seed(42)

# Load data
print("Loading data...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Data preprocessing
print("Preprocessing data...")
train_df = train_df.copy()
test_df = test_df.copy()

# Handle missing values
train_df = train_df.dropna()
test_df = test_df.dropna()

# Rename columns for consistency
train_df.columns = [
    "date",
    "date_block_num",
    "item_cnt_day",
    "item_id",
    "item_price",
    "shop_id",
]
test_df.columns = ["date", "date_block_num", "item_id", "item_price", "shop_id"]


# Feature engineering
def create_features(df, target_col="item_cnt_day"):
    df = df.copy()
    # Clip negative values to 0 before log transformation
    df["item_price_log"] = np.log1p(df["item_price"].clip(lower=0))
    df["date_block_num"] = df["date_block_num"].astype(float)
    df["item_id"] = df["item_id"].astype(float)
    df["shop_id"] = df["shop_id"].astype(float)
    # Create interaction features
    df["item_price_item_id"] = df["item_price"] * df["item_id"]
    df["date_block_num_shop_id"] = df["date_block_num"] * df["shop_id"]
    return df


train_features = create_features(train_df)
test_features = create_features(test_df)

# Prepare features for modeling
feature_cols = [
    "date_block_num",
    "item_id",
    "shop_id",
    "item_price_log",
    "item_price_item_id",
    "date_block_num_shop_id",
]

X_train = train_features[feature_cols].fillna(0)
y_train = train_features["item_cnt_day"].fillna(0)
X_test = test_features[feature_cols].fillna(0)

# Define LightGBM parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

# 5-fold cross-validation
print("Starting 5-fold cross-validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_rmse = np.inf
best_model = None
best_score = 0

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold, reference=None)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=None)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Evaluate on validation set
    y_pred_val = model.predict(X_val_fold, num_iteration=model.best_iteration)
    fold_rmse = mean_squared_error(y_val_fold, y_pred_val)
    print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}")

    if fold_rmse < best_rmse:
        best_rmse = fold_rmse
        best_model = model

print(f"Best RMSE on CV: {best_rmse:.4f}")

# Train final model on all data with validation set for early stopping
print("Training final model on all data...")
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_train.iloc[:1000], label=y_train.iloc[:1000], reference=None)

final_model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Make predictions on test set
print("Generating test predictions...")
test_predictions = final_model.predict(X_test)
test_predictions = np.clip(test_predictions, 0, None)  # Ensure non-negative

# Create submission file
submission_df = pd.DataFrame(
    {"id": range(len(test_df)), "item_cnt_day": test_predictions}
)
submission_df.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")

# Print final RMSE metric
print(f"\nFinal RMSE on validation set: {best_rmse:.4f}")
