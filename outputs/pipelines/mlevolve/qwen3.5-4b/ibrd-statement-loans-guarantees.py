import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# Set random seed for reproducibility
np.random.seed(42)

# Load data
print("Loading data...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Check columns
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Train columns: {train_df.columns.tolist()}")
print(f"Test columns: {test_df.columns.tolist()}")

# Define target column
target_col = "Original Principal Amount (US$)"

# Prepare features - keep only numeric columns for regression
numeric_cols = [col for col in train_df.columns if pd.api.types.is_numeric_dtype(train_df[col])]
feature_cols = [col for col in numeric_cols if col != target_col]

# Select only the feature columns we identified
X_train = train_df[feature_cols].astype(float)
y_train = train_df[target_col].astype(float)
X_test = test_df[feature_cols].astype(float)

# Split data for validation FIRST (before any feature engineering/scaling)
X_train_val, X_test_val, y_train_val, y_train = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# LightGBM parameters
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

# Train model
print("Training model...")
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

model.fit(
    X_train_val,
    y_train_val,
    eval_set=[(X_train_val, y_train_val), (X_test_val, y_train)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(50)],
)

# Calculate validation score using model predictions
y_val_pred = model.predict(X_train_val)
rmse_score = np.sqrt(mean_squared_error(y_train_val, y_val_pred))
print(f"Final Validation Score: {rmse_score:.4f}")

# Generate predictions for test set
print("Generating test predictions...")
test_predictions = model.predict(X_test)

# Create submission dataframe
submission = pd.DataFrame(
    {"id": range(len(test_df)), "Original Principal Amount (US$)": test_predictions}
)

# Save submission
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission.shape}")
print(f"Submission columns: {submission.columns.tolist()}")