import pandas as pd
import numpy as np
import catboost as cb
import os

# Define paths for training and testing data
train_path = '/path/to/user/workspace/ds_failures/datasets/uci/istanbul-stock-exchange/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/istanbul-stock-exchange/data/test.parquet'

# Load data
try:
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
except Exception as e:
    raise RuntimeError(f"Failed to load parquet files. Check paths or file availability: {e}")

# Define target variable
target_col = 'ISE'

# Verify target exists in train_df
if target_col not in train_df.columns:
    raise ValueError(f"Target column '{target_col}' not found in training data.")

# Select features (exclude target)
# The dataset description lists 'ISE' potentially twice, so we drop any instance of 'ISE' from features to ensure only predictors remain
feature_columns = [col for col in train_df.columns if col != target_col]

# Filter out any empty columns if present (though unlikely with parquet)
X_train = train_df[feature_columns].copy()
y_train = train_df[target_col].copy()
X_test = test_df[feature_columns].copy()

# Initialize CatBoost Regressor with default parameters for automatic handling of categorical/missing values
model = cb.CatBoostRegressor(
    objective='RMSE', 
    iterations=500,
    learning_rate=0.1,
    verbose=False,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions on test set
preds = model.predict(X_test)

# Prepare output dataframe
ids = list(range(len(preds)))
output_df = pd.DataFrame({'id': ids, 'ISE': preds})

# Save predictions to CSV
output_file = 'predictions.csv'
output_df.to_csv(output_file, index=False)

print(f"Predictions saved to {output_file}")