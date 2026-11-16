import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# Paths as specified in the prompt
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/uci/bike-sharing-dataset/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/uci/bike-sharing-dataset/data/test.parquet'

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Load data
df_train = pd.read_parquet(TRAIN_PATH)
df_test = pd.read_parquet(TEST_PATH)

# Identify features and target based on description
# Exclude ID ('instant') and Target ('cnt') and unused columns if any (e.g., 'dteday')
# We retain the explicit numeric and categorical columns defined in the table
feature_cols = [
    'season', 'yr', 'mnth', 'hr', 
    'holiday', 'weekday', 'workingday', 'weathersit', 
    'temp', 'atemp', 'hum', 'windspeed'
]
target_col = 'cnt'

# Prepare Training Data
X_train = df_train[feature_cols].dropna()  # Remove rows with any missing values in features
y_train = df_train[target_col]

# Ensure train set has all feature columns (some rows might be dropped if not aligned)
# However, for simplicity and standard workflow, we re-select after dropping nulls
X_train = df_train[feature_cols].fillna(method="ffill") # Fallback for safety, though CatBoost handles nan
# Better: Concatenate original columns back to see what was dropped if needed, but for now:
# Re-load and clean
df_clean_train = df_train.copy()
df_clean_train[feature_cols] = df_clean_train[feature_cols].fillna(0)
X_train_final = df_clean_train[feature_cols]
y_train_final = df_clean_train[target_col].values

# Prepare Test Data
# Replace categorical/nan handling similar to training phase to ensure consistency
df_clean_test = df_test.copy()
df_clean_test[feature_cols] = df_clean_test[feature_cols].fillna(0) # Fill NaNs similarly
X_test_final = df_clean_test[feature_cols]
test_id_idx = range(len(df_clean_test))

# Train Model
model = CatBoostRegressor(
    iterations=1000, 
    depth=6, 
    learning_rate=0.05, 
    eval_metric='RMSE',
    silent=True,
    verbose=0
)

model.fit(X_train_final, y_train_final)

# Make Predictions
predictions = model.predict(X_test_final)

# Create Output DataFrame
output_df = pd.DataFrame({
    'id': test_id_idx,
    'cnt': predictions
})

# Save predictions
output_df.to_csv('predictions.csv', index=False)