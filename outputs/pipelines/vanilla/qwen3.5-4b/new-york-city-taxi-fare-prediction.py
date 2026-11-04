import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Define paths
data_root = "/path/to/user/workspace/ds_failures/datasets/mlebench/new-york-city-taxi-fare-prediction/data/"
train_file = os.path.join(data_root, "train.parquet")
test_file = os.path.join(data_root, "test.parquet")

# Load data
train_df = pd.read_parquet(train_file)
test_df = pd.read_parquet(test_file)

target_col = "fare_amount"

# Get feature columns (exclude target)
feature_cols = [col for col in train_df.columns if col != target_col]

# Initialize feature matrices
X_train = train_df[feature_cols].copy()
X_test = test_df[feature_cols].copy()

# Preprocessing: Handle missing values and categorical columns
for col in feature_cols:
    if pd.api.types.is_numeric_dtype(X_train[col]):
        # Numeric: Fill missing with median
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)
    else:
        # Categorical/Object: Fill missing strings and Label Encode
        # Convert to string to ensure uniformity
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)
        
        # Replace nulls in object columns
        X_train[col] = X_train[col].fillna('MISSING_CAT')
        X_test[col] = X_test[col].fillna('MISSING_CAT')
        
        # Label Encode
        le = LabelEncoder()
        le.fit(X_train[col])
        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])

# Train LightGBM model
model = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=42, n_jobs=-1)
model.fit(X_train, train_df[target_col])

# Make predictions
predictions = model.predict(X_test)

# Create output dataframe with required columns: 'id' and 'fare_amount'
result_df = pd.DataFrame({
    "id": range(len(test_df)),
    "fare_amount": predictions
})

# Save to CSV
result_df.to_csv("predictions.csv", index=False)