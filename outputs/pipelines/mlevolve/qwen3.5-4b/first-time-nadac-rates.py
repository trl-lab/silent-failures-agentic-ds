

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import os

# Read data
train_df = pd.read_parquet('./input/train.parquet')
test_df = pd.read_parquet('./input/test.parquet')

# Identify target column
target_col = 'NADAC Rate'

# Check available columns
print(f"Train columns: {train_df.columns.tolist()}")
print(f"Test columns: {test_df.columns.tolist()}")

# Preprocess data
# Use only numeric columns for modeling
numeric_cols = [col for col in train_df.columns if train_df[col].dtype in ['int64', 'float64']]
print(f"Using numeric columns: {numeric_cols}")

X_train = train_df[numeric_cols].copy()
y_train = train_df[target_col].copy()
X_test = test_df[numeric_cols].copy()

# Create validation split
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_split)
X_val_scaled = scaler.transform(X_val_split)
X_test_scaled = scaler.transform(X_test)

# Create LightGBM datasets
train_data = lgb.Dataset(X_train_scaled, label=y_train_split)
val_data = lgb.Dataset(X_val_scaled, label=y_val_split, reference=train_data)

# Train model with early stopping
model = lgb.train(
    {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'verbose': -1,
        'seed': 42,
    },
    train_set=train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(50),
        lgb.log_evaluation(-1)
    ]
)

# Make predictions on test data
test_predictions = model.predict(X_test_scaled)

# Create submission dataframe
submission_df = pd.DataFrame({
    'id': range(len(test_df)),
    'NADAC Rate': test_predictions
})

# Save submission
submission_df.to_csv('./submission/submission.csv', index=False)

print("Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")

# Calculate and print validation score
val_predictions = model.predict(X_val_scaled)
val_rmse = mean_squared_error(y_val_split, val_predictions, squared=False)
print(f'Final Validation Score: {val_rmse:.6f}')