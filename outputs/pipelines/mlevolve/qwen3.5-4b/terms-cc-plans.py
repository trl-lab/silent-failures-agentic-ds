import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# 1. Data Loading and Preprocessing
# ============================================================================

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Identify feature columns (exclude target 'APR' and non-numeric columns)
all_numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
target_col = "APR"
feature_cols = [col for col in all_numeric_cols if col != target_col]

# Create feature matrix and target vector
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()

X_test = test_df[feature_cols].copy()

# Handle any missing values
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

print(f"Feature columns: {len(feature_cols)}")

# ============================================================================
# 2. Model Training with Validation Split
# ============================================================================

# Split data into train and validation sets
X_train_full, X_val, y_train_full, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Create LightGBM dataset
train_data = lgb.Dataset(X_train_full, label=y_train_full)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Define LightGBM parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

# Train model with early stopping
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(100, verbose=False)],
)

# ============================================================================
# 3. Validation Metric Calculation
# ============================================================================

# Predict on validation set
y_val_pred = model.predict(X_val)

# Calculate validation score (rmse)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

print(f"Final Validation Score: {val_rmse:.6f}")

# ============================================================================
# 4. Test Set Prediction and Submission
# ============================================================================

# Predict on test set
test_predictions = model.predict(X_test)

# Create submission dataframe with correct format
submission_df = pd.DataFrame({"id": range(len(test_df)), "APR": test_predictions})

# Save submission
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")
print(f"Submission columns: {submission_df.columns.tolist()}")
