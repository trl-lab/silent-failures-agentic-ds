import pandas as pd
import numpy as np
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

# Define file paths
train_path = "/path/to/user/workspace/ds_failures/datasets/strable/prepaid-financial-product/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/prepaid-financial-product/data/test.parquet"

# Load data
df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

# Select target variable
target_col = "current_status"

# Remove target column to keep it hidden from features, ensuring model trains on features
X_train = df_train.drop(columns=[target_col])
y_train = df_train[target_col]

# Prepare test features
X_test = df_test.drop(columns=[target_col])

# Initialize model with automatic handling for categoricals and missing values
model = cb.CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, verbose=False, random_state=42)

# Train the model
model.fit(X_train, y_train, eval_set=(X_test, y_train), early_stopping_rounds=10)

# Generate predictions for test set
pred_values = model.predict(X_test)

# Create predictions DataFrame with required columns
result_df = pd.DataFrame({
    "id": range(len(pred_values)),
    "current_status": pred_values
})

# Save predictions to csv
result_df.to_csv("predictions.csv", index=False)