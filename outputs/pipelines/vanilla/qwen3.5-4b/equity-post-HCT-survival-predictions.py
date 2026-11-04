import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

# Define paths
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/kaggle/equity-post-HCT-survival-predictions/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/kaggle/equity-post-HCT-survival-predictions/data/test.parquet'
OUTPUT_PATH = 'predictions.csv'

# Load Data
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

# Define target and feature columns
target_column = 'efs'
feature_columns = [col for col in train_df.columns if col != target_column and col not in ['id']]

X_train = train_df[feature_columns].copy()
y_train = train_df[target_column].values

X_test = test_df[feature_columns].copy()

# Handle missing values and categorical columns automatically using CatBoost settings
# CatBoost handles missing values and categorical variables internally, so we pass raw data.
# We ensure the target is not included in the prediction.

# Initialize Model
model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.03,
    depth=6,
    loss_function='RMSE',
    random_state=42,
    verbose=0
)

# Train Model
model.fit(X_train, y_train)

# Predict on Test Set
# CatBoostRegressor returns the prediction directly.
# Ensure we don't accidentally predict 'efs' if it exists in test df columns (should not)
predictions = model.predict(X_test)

# Prepare Output DataFrame
output_df = pd.DataFrame({
    'id': range(len(predictions)),
    'efs': predictions
})

# Save Predictions
output_df.to_csv(OUTPUT_PATH, index=False)