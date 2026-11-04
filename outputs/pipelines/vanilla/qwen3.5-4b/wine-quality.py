import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# Define file paths
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/uci/wine-quality/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/uci/wine-quality/data/test.parquet'

# Load data
df_train = pd.read_parquet(TRAIN_PATH)
df_test = pd.read_parquet(TEST_PATH)

# Separate features and target
target_column = 'quality'
X_train = df_train.drop(columns=[target_column])
y_train = df_train[target_column]
X_test = df_test.drop(columns=[target_column])

# Initialize model
# CatBoost automatically handles categorical features and missing values
model = CatBoostRegressor(iterations=200, depth=6, learning_rate=0.05, verbose=False, random_seed=42)

# Train model
model.fit(X_train, y_train)

# Predict on test set
preds = model.predict(X_test)

# Create output dataframe
# id starts from 0 corresponding to test set row index
output_df = pd.DataFrame({
    'id': range(len(preds)),
    'quality': preds
})

# Save predictions
output_df.to_csv('predictions.csv', index=False)