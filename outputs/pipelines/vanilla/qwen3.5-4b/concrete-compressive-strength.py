import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import os

# Define paths based on the data description
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/concrete-compressive-strength/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/concrete-compressive-strength/data/test.parquet"

# Load data
df_train = pd.read_parquet(TRAIN_PATH)
df_test = pd.read_parquet(TEST_PATH)

# Identify target feature
target_column = 'strength'

# Separate features and target
X_train = df_train.drop(columns=[target_column])
y_train = df_train[target_column]
X_test = df_test.copy()

# Initialize the CatBoost Regressor which automatically handles missing values and categorical columns
# n_estimators: number of boosting rounds
# depth: tree depth
# learning_rate: step size shrinkage
# verbose: print information
model = CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.1, verbose=0, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Prepare output dataframe
output_df = pd.DataFrame({
    'id': range(len(df_test)),
    'strength': predictions
})

# Save predictions to csv in the current working directory
output_df.to_csv('predictions.csv', index=False)