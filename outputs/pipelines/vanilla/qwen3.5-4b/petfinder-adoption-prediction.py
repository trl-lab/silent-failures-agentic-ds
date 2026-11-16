import pandas as pd
import numpy as np
import catboost as cb

# Define paths
train_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/petfinder-adoption-prediction/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/petfinder-adoption-prediction/data/test.parquet"

# Load data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Target column
target_col = "AdoptionSpeed"

# Separate features from train/test
# Find common columns excluding target
common_cols = [col for col in train_df.columns if col != target_col]

# Prepare Train
X_train = train_df[common_cols].copy()
y_train = train_df[target_col].copy()

# Prepare Test
X_test = test_df[common_cols].copy()

# Handle missing values automatically (median fill for numeric, leave as is if categorical handling in model)
# CatBoost handles string columns as categorical automatically and has its own logic for missing values, 
# but filling with median is a good baseline to reduce noise if the model doesn't prefer explicit handling.
# We apply fillna to all columns to ensure numerical continuity if CatBoost needs it.
# Note: CatBoost handles nan internally if allow_missing=True (default). To be safe and robust:
# Fill with median.
def fill_na(series):
    return series.fillna(series.median() if pd.api.types.is_float_dtype(series) else series.mode()[0] if len(series) > 0 else 0)

# Apply to columns that were originally float/int (assumed numeric for training safety)
for col in common_cols:
    if pd.api.types.is_numeric_dtype(X_train[col]):
        X_train[col] = X_train[col].fillna(X_train[col].median())
        X_test[col] = X_test[col].fillna(X_test[col].median())
    elif pd.api.types.is_object_dtype(X_train[col]):
        # For categorical strings, ensure consistent representation
        pass

# Initialize Model
model = cb.CatBoostRegressor(iterations=1000, learning_rate=0.05, verbose=0, random_state=42)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Generate Output DataFrame
results_df = pd.DataFrame({
    "id": range(len(predictions)),
    "AdoptionSpeed": predictions
})

# Save to CSV
results_df.to_csv("predictions.csv", index=False)