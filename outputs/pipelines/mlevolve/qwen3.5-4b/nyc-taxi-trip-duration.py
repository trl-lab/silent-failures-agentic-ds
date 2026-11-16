import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import os
import pickle

# Set random seed for reproducibility
np.random.seed(42)

# Create working directories if they don't exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================================
# STEP 1: Data Loading and Preprocessing
# ============================================================================

print("Loading data...")

# Load training data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# ============================================================================
# Feature Engineering
# ============================================================================

print("Engineering features...")

# Extract time features from dropoff_datetime
train_df["dropoff_hour"] = pd.to_datetime(train_df["dropoff_datetime"]).dt.hour
train_df["dropoff_day"] = pd.to_datetime(train_df["dropoff_datetime"]).dt.dayofweek
train_df["dropoff_month"] = pd.to_datetime(train_df["dropoff_datetime"]).dt.month
train_df["dropoff_hour_of_day"] = pd.to_datetime(train_df["dropoff_datetime"]).dt.hour
train_df["dropoff_day_of_week"] = pd.to_datetime(train_df["dropoff_datetime"]).dt.dayofweek
train_df["dropoff_month"] = pd.to_datetime(train_df["dropoff_datetime"]).dt.month

# Extract time features from pickup_datetime
train_df["pickup_hour"] = pd.to_datetime(train_df["pickup_datetime"]).dt.hour
train_df["pickup_day"] = pd.to_datetime(train_df["pickup_datetime"]).dt.dayofweek
train_df["pickup_month"] = pd.to_datetime(train_df["pickup_datetime"]).dt.month
train_df["pickup_hour_of_day"] = pd.to_datetime(train_df["pickup_datetime"]).dt.hour
train_df["pickup_day_of_week"] = pd.to_datetime(train_df["pickup_datetime"]).dt.dayofweek
train_df["pickup_month"] = pd.to_datetime(train_df["pickup_datetime"]).dt.month

# Calculate time differences
train_df["pickup_dropoff_hour"] = (
    pd.to_datetime(train_df["dropoff_datetime"])
    - pd.to_datetime(train_df["pickup_datetime"])
).dt.total_seconds() / 3600  # Convert to hours

# Calculate distance (simplified using latitude/longitude difference)
train_df["pickup_dropoff_lat_diff"] = (
    train_df["dropoff_latitude"] - train_df["pickup_latitude"]
)
train_df["pickup_dropoff_lon_diff"] = (
    train_df["dropoff_longitude"] - train_df["pickup_longitude"]
)
train_df["pickup_dropoff_dist"] = np.sqrt(
    train_df["pickup_dropoff_lat_diff"] ** 2 + train_df["pickup_dropoff_lon_diff"] ** 2
)

# Add combined time features
train_df["pickup_hour_day_interaction"] = train_df["pickup_hour"] * train_df["pickup_day"]
train_df["dropoff_hour_day_interaction"] = train_df["dropoff_hour"] * train_df["dropoff_day"]
train_df["pickup_hour_dropoff_hour_diff"] = train_df["pickup_hour"] - train_df["dropoff_hour"]
train_df["pickup_day_dropoff_day_diff"] = train_df["pickup_day"] - train_df["dropoff_day"]

# Add vendor_id interactions with time features
train_df["vendor_id_pickup_hour"] = train_df["vendor_id"] * train_df["pickup_hour"]
train_df["vendor_id_dropoff_hour"] = train_df["vendor_id"] * train_df["dropoff_hour"]
train_df["vendor_id_pickup_day"] = train_df["vendor_id"] * train_df["pickup_day"]
train_df["vendor_id_dropoff_day"] = train_df["vendor_id"] * train_df["dropoff_day"]

# Fill missing values
train_df["passenger_count"] = train_df["passenger_count"].fillna(
    train_df["passenger_count"].median()
)
train_df["vendor_id"] = train_df["vendor_id"].fillna(train_df["vendor_id"].mode()[0])

# Store original passenger_count for later use
train_df["original_passenger_count"] = train_df["passenger_count"]

# Encode day of week as categorical (0-6)
train_df["pickup_day"] = train_df["pickup_day"].astype(int)
train_df["dropoff_day"] = train_df["dropoff_day"].astype(int)
train_df["vendor_id"] = train_df["vendor_id"].astype(int)

# ============================================================================
# Feature Engineering for Test Data
# ============================================================================

print("Engineering features for test data...")

# Extract time features from dropoff_datetime
test_df["dropoff_hour"] = pd.to_datetime(test_df["dropoff_datetime"]).dt.hour
test_df["dropoff_day"] = pd.to_datetime(test_df["dropoff_datetime"]).dt.dayofweek
test_df["dropoff_month"] = pd.to_datetime(test_df["dropoff_datetime"]).dt.month
test_df["dropoff_hour_of_day"] = pd.to_datetime(test_df["dropoff_datetime"]).dt.hour
test_df["dropoff_day_of_week"] = pd.to_datetime(test_df["dropoff_datetime"]).dt.dayofweek
test_df["dropoff_month"] = pd.to_datetime(test_df["dropoff_datetime"]).dt.month

# Extract time features from pickup_datetime
test_df["pickup_hour"] = pd.to_datetime(test_df["pickup_datetime"]).dt.hour
test_df["pickup_day"] = pd.to_datetime(test_df["pickup_datetime"]).dt.dayofweek
test_df["pickup_month"] = pd.to_datetime(test_df["pickup_datetime"]).dt.month
test_df["pickup_hour_of_day"] = pd.to_datetime(test_df["pickup_datetime"]).dt.hour
test_df["pickup_day_of_week"] = pd.to_datetime(test_df["pickup_datetime"]).dt.dayofweek
test_df["pickup_month"] = pd.to_datetime(test_df["pickup_datetime"]).dt.month

# Calculate time differences
test_df["pickup_dropoff_hour"] = (
    pd.to_datetime(test_df["dropoff_datetime"])
    - pd.to_datetime(test_df["pickup_datetime"])
).dt.total_seconds() / 3600  # Convert to hours

# Calculate distance (simplified using latitude/longitude difference)
test_df["pickup_dropoff_lat_diff"] = (
    test_df["dropoff_latitude"] - test_df["pickup_latitude"]
)
test_df["pickup_dropoff_lon_diff"] = (
    test_df["dropoff_longitude"] - test_df["pickup_longitude"]
)
test_df["pickup_dropoff_dist"] = np.sqrt(
    test_df["pickup_dropoff_lat_diff"] ** 2 + test_df["pickup_dropoff_lon_diff"] ** 2
)

# Add combined time features
test_df["pickup_hour_day_interaction"] = test_df["pickup_hour"] * test_df["pickup_day"]
test_df["dropoff_hour_day_interaction"] = test_df["dropoff_hour"] * test_df["dropoff_day"]
test_df["pickup_hour_dropoff_hour_diff"] = test_df["pickup_hour"] - test_df["dropoff_hour"]
test_df["pickup_day_dropoff_day_diff"] = test_df["pickup_day"] - test_df["dropoff_day"]

# Add vendor_id interactions with time features
test_df["vendor_id_pickup_hour"] = test_df["vendor_id"] * test_df["pickup_hour"]
test_df["vendor_id_dropoff_hour"] = test_df["vendor_id"] * test_df["dropoff_hour"]
test_df["vendor_id_pickup_day"] = test_df["vendor_id"] * test_df["pickup_day"]
test_df["vendor_id_dropoff_day"] = test_df["vendor_id"] * test_df["dropoff_day"]

# Fill missing values
test_df["passenger_count"] = test_df["passenger_count"].fillna(
    test_df["passenger_count"].median()
)
test_df["vendor_id"] = test_df["vendor_id"].fillna(test_df["vendor_id"].mode()[0])

# Store original passenger_count for later use
test_df["original_passenger_count"] = test_df["passenger_count"]

# Encode day of week as categorical (0-6)
test_df["pickup_day"] = test_df["pickup_day"].astype(int)
test_df["dropoff_day"] = test_df["dropoff_day"].astype(int)
test_df["vendor_id"] = test_df["vendor_id"].astype(int)

# Store original passenger_count for later use
test_df["original_passenger_count"] = test_df["passenger_count"]

# ============================================================================
# Prepare Features and Target
# ============================================================================

print("Preparing features...")

# Define feature columns
feature_columns = [
    "dropoff_hour",
    "dropoff_day",
    "dropoff_month",
    "pickup_hour",
    "pickup_day",
    "pickup_month",
    "pickup_dropoff_hour",
    "pickup_dropoff_lat_diff",
    "pickup_dropoff_lon_diff",
    "pickup_dropoff_dist",
    "passenger_count",
    "vendor_id",
    "original_passenger_count",
    "pickup_hour_of_day",
    "dropoff_hour_of_day",
    "pickup_day_of_week",
    "dropoff_day_of_week",
    "pickup_hour_day_interaction",
    "dropoff_hour_day_interaction",
    "pickup_hour_dropoff_hour_diff",
    "pickup_day_dropoff_day_diff",
    "vendor_id_pickup_hour",
    "vendor_id_dropoff_hour",
    "vendor_id_pickup_day",
    "vendor_id_dropoff_day",
]

# Encode categorical variables
train_df["vendor_id"] = train_df["vendor_id"].astype(int)
test_df["vendor_id"] = test_df["vendor_id"].astype(int)

# ============================================================================
# Train-Validation Split
# ============================================================================

print("Splitting data...")

# Use 70-30 split for train-validation
train_features = train_df[feature_columns]
train_target = train_df["trip_duration"]

# Split into train and validation sets FIRST before any feature engineering
train_features, val_features, train_target, val_target = train_test_split(
    train_features, train_target, test_size=0.3, random_state=42
)

print(f"Train set size: {len(train_target)}")
print(f"Validation set size: {len(val_target)}")

# ============================================================================
# Create LightGBM Model
# ============================================================================

print("Creating LightGBM model...")

# Define callbacks
callbacks = [lgb.early_stopping(100), lgb.log_evaluation(-1)]

# Create LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=-1,
    num_leaves=57,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)

# ============================================================================
# Train Model
# ============================================================================

print("Training model...")

model.fit(
    train_features,
    train_target,
    eval_set=[(val_features, val_target)],
    callbacks=callbacks,
)

# ============================================================================
# Validate Model
# ============================================================================

print("Validating model...")

# Make predictions on validation set
val_predictions = model.predict(val_features)

# Calculate RMSE (Root Mean Squared Error)
val_rmse = np.sqrt(mean_squared_error(val_target, val_predictions))

print(f"Final Validation Score: {val_rmse:.4f}")

# ============================================================================
# Generate Submission
# ============================================================================

print("Generating submission...")

# Make predictions on test set
test_predictions = model.predict(test_df[feature_columns])

# Create submission dataframe
submission_df = pd.DataFrame({"id": test_df["id"], "trip_duration": test_predictions})

# Save submission file
submission_df.to_csv("./submission/submission.csv", index=False)

print("Submission saved to ./submission/submission.csv")
print("Done!")