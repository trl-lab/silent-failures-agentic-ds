import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

# Define file paths
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/fts-requirement-and-funding/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/fts-requirement-and-funding/data/test.parquet"

# Load data
print("Loading data...")
df_train = pd.read_parquet(TRAIN_PATH)
df_test = pd.read_parquet(TEST_PATH)

# Identify target and feature columns
target_col = "funding"
feature_cols = [col for col in df_train.columns if col != target_col]

if len(feature_cols) == 0:
    raise ValueError("No feature columns found.")

# Initialize preprocessing
# Handle numerical missing values
numerical_transformer = SimpleImputer(strategy="median")

# Handle categorical missing values and encoding
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine transformers
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, feature_cols)
], remainder="passthrough")

# Create the full pipeline
model_pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

# Fit the model on training data
print("Training model...")
model_pipeline.fit(df_train[feature_cols], df_train[target_col])

# Apply preprocessing and prediction on test data
# Reset index to ensure sequential IDs starting from 0
test_df_clean = df_test.copy()
test_df_clean = test_df_clean[feature_cols]

# We need to match the original test dataframe's row count to the ID requirement
# The requirement says "id (integer row index starting from 0)" implying a sequential index 0..N-1
test_indices = np.arange(len(df_test))

# Make predictions
pred_values = model_pipeline.predict(test_df_clean)

# Create output dataframe
output_df = pd.DataFrame({
    "id": test_indices,
    "funding": pred_values
})

# Save predictions
output_path = "predictions.csv"
output_df.to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")