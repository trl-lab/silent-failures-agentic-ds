import os
import sys
import pandas as pd
import numpy as np

# Ensure paths are handled correctly
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/appliances-energy-prediction/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/appliances-energy-prediction/data/test.parquet"
OUTPUT_PATH = "predictions.csv"

# Load data
print("Loading training data...")
df_train = pd.read_parquet(TRAIN_PATH)
df_test = pd.read_parquet(TEST_PATH)

# Define target column
target_col = 'Appliances'

# Prepare Features
# Exclude target and 'date' (often not predictive without lagging/binning and may cause category issues)
exclude_cols = [target_col, 'date']
feature_columns = [col for col in df_train.columns if col not in exclude_cols]

print("Preprocessing and training...")
if len(feature_columns) == 0:
    raise ValueError("No valid feature columns found.")

# Select columns
X_train = df_train[feature_columns].copy()
X_test = df_test[feature_columns].copy()
y_train = df_train[target_col].copy()

# Handle missing values automatically by filling with median
# CatBoost handles missing values, but explicit filling ensures consistency with sklearn defaults if switched
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

# Handle potential NaNs again (Parquet reading sometimes leaves None)
# Using median fill again to ensure all are numeric
for col in feature_columns:
    if not pd.api.types.is_numeric_dtype(X_train[col]):
        try:
            # Attempt to convert to numeric if it failed to read properly
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(X_train[col].median())
        except:
            pass
    if not pd.api.types.is_numeric_dtype(X_test[col]):
        try:
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(X_test[col].median())
        except:
            pass

# Select numeric columns to force behavior
X_train = X_train.select_dtypes(include=[np.number])
X_test = X_test.select_dtypes(include=[np.number])

# Ensure no remaining NaNs
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

# Train model using CatBoost which handles various data types robustly
try:
    from catboost import CatBoostRegressor
except ImportError:
    from lightgbm import LGBMRegressor
    
    # Fallback to LightGBM
    model = LGBMRegressor(objective='regression', random_state=42, verbosity=0,
                          max_depth=10, num_leaves=31, learning_rate=0.1, n_estimators=1000)
else:
    model = CatBoostRegressor(iterations=1000, eval_metric='RMSE', verbose=False, random_state=42)

model.fit(X_train, y_train)

# Predictions
print("Generating predictions...")
preds = model.predict(X_test)

# Create output DataFrame with 'id' starting from 0
output_df = pd.DataFrame({'id': range(len(X_test)), 'Appliances': preds})

# Save to csv
output_df.to_csv(OUTPUT_PATH, index=False)

print(f"Predictions saved to {OUTPUT_PATH}")