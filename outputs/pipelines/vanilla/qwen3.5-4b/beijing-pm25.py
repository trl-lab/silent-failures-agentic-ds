import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler

# Define file paths
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/uci/beijing-pm25/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/uci/beijing-pm25/data/test.parquet'

# Load data
df_train = pd.read_parquet(TRAIN_PATH)
df_test = pd.read_parquet(TEST_PATH)

# Define target and ID columns to exclude
TARGET_COL = 'pm2.5'
EXCLUDE_COLS = {'pm2.5', 'No'}

# Identify feature columns present in train
train_feature_cols = [col for col in df_train.columns if col not in EXCLUDE_COLS]

# Ensure feature columns exist in test
available_features_in_test = [col for col in train_feature_cols if col in df_test.columns]

# Prepare Training Data
X_train = df_train[available_features_in_test]
y_train = df_train[TARGET_COL]

# Prepare Test Data
X_test = df_test[available_features_in_test].copy()

# Categorical handling: Identify categorical columns from description
CATEGORICAL_FEATURES = ['cbwd']

# Convert specific categorical columns to category dtype for LightGBM
for col in CATEGORICAL_FEATURES:
    if col in available_features_in_test:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

# Handle Missing Values in Numerical Features
numerical_features = [col for col in available_features_in_test if col not in CATEGORICAL_FEATURES]
missing_indices = {}

for col in numerical_features:
    # Find indices of missing values
    missing_mask = X_train[col].isna()
    if missing_mask.any():
        train_fill_val = X_train[col][~missing_mask].median()
        test_fill_val = X_test[col][~X_test[col].index.isna()][~X_test[col].isna()].median() if X_test[col].notna().sum() > 0 else np.nan
        
        X_train[col] = X_train[col].fillna(train_fill_val)
        X_test[col] = X_test[col].fillna(train_fill_val)
    
    # Also handle missing in test specifically if different logic
    if missing_mask.sum() == len(X_train):
        pass

# Additional safety for infinity
for col in numerical_features:
    X_train[col] = X_train[col].replace([np.inf, -np.inf], np.nan)
    X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan)

# Initialize LightGBM Model
model = LGBMRegressor(random_state=42, verbose=-1, objective='regression')

# Train Model
model.fit(X_train, y_train)

# Predict on Test Set
predictions = model.predict(X_test)

# Create Output DataFrame
result_df = pd.DataFrame({
    'id': range(len(X_test)),
    'pm2.5': predictions
})

# Save predictions
result_df.to_csv('predictions.csv', index=False)