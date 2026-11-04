import pandas as pd
import numpy as np
import catboost as ctb

# File Paths
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/mlebench/stanford-covid-vaccine/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/mlebench/stanford-covid-vaccine/data/test.parquet"

# Load Data
try:
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load data: {e}")

# Identify columns
TARGET_COL = "reactivity"
ID_COL = "id"

# Get feature columns
# Exclude target and id from both train and test
feature_columns = [col for col in train_df.columns if col != TARGET_COL and col != ID_COL]

# Prepare Training Data
X_train = train_df[feature_columns].copy()
y_train = train_df[TARGET_COL].copy()

# Handle Missing Values and Categories automatically using CatBoost
# CatBoost handles categorical and missing value imputation efficiently
# Convert to float where possible for safety, though CB handles strings as categorical
# Ensure column order is consistent
X_train = X_train.reindex(columns=feature_columns)
X_train = X_train.fillna(-1)  # Default miss_value for CatBoost

# Prepare Test Data
test_features = test_df[[col for col in feature_columns if col in test_df.columns]].copy()
test_features = test_features.fillna(-1)

# Model Training
model = ctb.CatBoostRegressor(
    iterations=1000,
    depth=6,
    learning_rate=0.05,
    losses=['RMSE'],
    metrics=['RMSE'],
    verbose=0,
    random_state=42
)

model.fit(X_train, y_train.values)

# Predictions
y_pred = model.predict(test_features)

# Prepare Output CSV
output_id = list(range(len(test_features)))
output_reactivity = y_pred.flatten()

result_df = pd.DataFrame({
    "id": output_id,
    "reactivity": output_reactivity
})

result_df.to_csv("predictions.csv", index=False)