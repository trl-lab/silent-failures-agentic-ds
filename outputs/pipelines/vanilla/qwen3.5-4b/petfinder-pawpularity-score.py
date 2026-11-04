import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# Define data paths
train_path = '/path/to/user/workspace/ds_failures/datasets/mlebench/petfinder-pawpularity-score/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/mlebench/petfinder-pawpularity-score/data/test.parquet'

# Load datasets
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Identify target column
target_col = "Pawpularity"

if target_col not in train_df.columns:
    raise ValueError(f"Target column '{target_col}' not found in training data.")

# Identify feature columns (all columns except target)
train_feature_cols = [col for col in train_df.columns if col != target_col]

# Find intersection to ensure test data has matching feature columns
common_feature_cols = [col for col in train_feature_cols if col in test_df.columns]

if not common_feature_cols:
    raise ValueError("No common feature columns found between train and test sets after filtering.")

# Prepare feature matrices
X_train = train_df[common_feature_cols]
X_test = test_df[common_feature_cols]
y_train = train_df[target_col]

# Initialize model using CatBoost for automatic handling of missing values and categorical columns
model = CatBoostRegressor(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="RMSE",
    silent=True,
    verbose=False,
    random_state=42,
    eval_metric="RMSE"
)

# Train the model
model.fit(X_train, y_train)

# Generate predictions
predictions = model.predict(X_test)

# Create output dataframe with required columns
# 'id' is the integer row index starting from 0
output_df = pd.DataFrame({
    'id': range(len(predictions)),
    'Pawpularity': predictions
})

# Save to predictions.csv
output_df.to_csv('predictions.csv', index=False)