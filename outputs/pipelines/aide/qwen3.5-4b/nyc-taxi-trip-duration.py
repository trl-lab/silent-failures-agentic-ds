import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Extract datetime features
train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
train_df["dropoff_datetime"] = pd.to_datetime(train_df["dropoff_datetime"])
train_df["pickup_hour"] = train_df["pickup_datetime"].dt.hour
train_df["dropoff_hour"] = train_df["dropoff_datetime"].dt.hour
train_df["pickup_day"] = train_df["pickup_datetime"].dt.day
train_df["dropoff_day"] = train_df["dropoff_datetime"].dt.day
train_df["pickup_date"] = train_df["pickup_datetime"].dt.date
train_df["dropoff_date"] = train_df["dropoff_datetime"].dt.date
train_df["pickup_date"] = train_df["pickup_datetime"].dt.date
train_df["dropoff_date"] = train_df["dropoff_datetime"].dt.date


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Extract datetime features
train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
train_df["dropoff_datetime"] = pd.to_datetime(train_df["dropoff_datetime"])
train_df["pickup_hour"] = train_df["pickup_datetime"].dt.hour
train_df["pickup_day"] = train_df["pickup_datetime"].dt.day
train_df["pickup_month"] = train_df["pickup_datetime"].dt.month
train_df["pickup_year"] = train_df["pickup_datetime"].dt.year
train_df["dropoff_hour"] = train_df["dropoff_datetime"].dt.hour
train_df["dropoff_day"] = train_df["dropoff_datetime"].dt.day
train_df["dropoff_month"] = train_df["dropoff_datetime"].dt.month
train_df["dropoff_year"] = train_df["dropoff_datetime"].dt.year
train_df["pickup_date"] = train_df["pickup_datetime"].dt.date
train_df["dropoff_date"] = train_df["dropoff_datetime"].dt.date
train_df["pickup_date"] = train_df["pickup_datetime"].dt.date
train_df["dropoff_date"] = train_df["dropoff_datetime"].dt.date
train_df["trip_length"] = (
    train_df["dropoff_datetime"] - train_df["pickup_datetime"]
).dt.total_seconds() / 3600
train_df["pickup_longitude"] = train_df["pickup_longitude"]
train_df["dropoff_longitude"] = train_df["dropoff_longitude"]
train_df["pickup_latitude"] = train_df["pickup_latitude"]
train_df["dropoff_latitude"] = train_df["dropoff_latitude"]
train_df["pickup_latitude"] = train_df["pickup_latitude"]
train_df["dropoff_latitude"] = train_df["dropoff_latitude"]
train_df["trip_distance"] = np.sqrt(
    (train_df["dropoff_longitude"] - train_df["pickup_longitude"]) ** 2
    + (train_df["dropoff_latitude"] - train_df["pickup_latitude"]) ** 2
)
train_df["trip_distance"] = train_df["trip_distance"]
train_df["trip_distance"] = train_df["trip_distance"]
train_df["trip_distance"] = train_df["trip_distance"]

# Extract test datetime features
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])
test_df["dropoff_datetime"] = pd.to_datetime(test_df["dropoff_datetime"])
test_df["pickup_hour"] = test_df["pickup_datetime"].dt.hour
test_df["pickup_day"] = test_df["pickup_datetime"].dt.day
test_df["pickup_month"] = test_df["pickup_datetime"].dt.month
test_df["pickup_year"] = test_df["pickup_datetime"].dt.year
test_df["dropoff_hour"] = test_df["dropoff_datetime"].dt.hour
test_df["dropoff_day"] = test_df["dropoff_datetime"].dt.day
test_df["dropoff_month"] = test_df["dropoff_datetime"].dt.month
test_df["dropoff_year"] = test_df["dropoff_datetime"].dt.year
test_df["pickup_date"] = test_df["pickup_datetime"].dt.date
test_df["dropoff_date"] = test_df["dropoff_datetime"].dt.date
test_df["pickup_date"] = test_df["pickup_datetime"].dt.date
test_df["dropoff_date"] = test_df["dropoff_datetime"].dt.date
test_df["trip_length"] = (
    test_df["dropoff_datetime"] - test_df["pickup_datetime"]
).dt.total_seconds() / 3600
test_df["pickup_longitude"] = test_df["pickup_longitude"]
test_df["dropoff_longitude"] = test_df["dropoff_longitude"]
test_df["pickup_latitude"] = test_df["pickup_latitude"]
test_df["dropoff_latitude"] = test_df["dropoff_latitude"]
test_df["trip_distance"] = np.sqrt(
    (test_df["dropoff_longitude"] - test_df["pickup_longitude"]) ** 2
    + (test_df["dropoff_latitude"] - test_df["pickup_latitude"]) ** 2
)
test_df["trip_distance"] = test_df["trip_distance"]
test_df["trip_distance"] = test_df["trip_distance"]
test_df["trip_distance"] = test_df["trip_distance"]

# Create feature matrix
feature_cols = [
    "pickup_hour",
    "pickup_day",
    "pickup_month",
    "pickup_year",
    "dropoff_hour",
    "dropoff_day",
    "dropoff_month",
    "dropoff_year",
    "pickup_longitude",
    "dropoff_longitude",
    "pickup_latitude",
    "dropoff_latitude",
    "trip_distance",
    "passenger_count",
    "store_and_fwd_flag",
    "vendor_id",
]
X_train = train_df[feature_cols].copy()
X_test = test_df[feature_cols].copy()
y_train = np.log1p(train_df["trip_duration"])

# Split data into train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Define LightGBM model
params = {
    "objective": "regression",
    "metric": "rmse",
    "verbose": -1,
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 20,
}

# Train model with callbacks
model = lgb.train(
    params,
    lgb.Dataset(X_train, label=y_train),
    num_boost_round=500,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Evaluate on validation set
val_pred = model.predict(X_val)
val_rmsle = mean_squared_log_error(y_val, val_pred)
print(f"RMSLE on validation set: {val_rmsle:.6f}")

# Predict on test set
test_pred = model.predict(X_test)
test_pred = np.clip(test_pred, 0, 1e6)
test_pred = np.round(test_pred)

# Create submission
submission = pd.DataFrame({"id": range(len(test_pred)), "trip_duration": test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
