import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor

# File paths
TRAIN_DATA_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/power-plants/data/train.parquet'
TEST_DATA_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/power-plants/data/test.parquet'
OUTPUT_PATH = './predictions.csv'
TARGET_COLUMN = 'NAMPLT_CAP'

# Load data
train_df = pd.read_parquet(TRAIN_DATA_PATH)
test_df = pd.read_parquet(TEST_DATA_PATH)

# Identify feature columns (all except target)
feature_columns = [col for col in train_df.columns if col != TARGET_COLUMN]

# Separate columns into numeric and categorical
numeric_cols = []
categorical_cols = []
for col in feature_columns:
    if pd.api.types.is_numeric_dtype(train_df[col]):
        numeric_cols.append(col)
    else:
        categorical_cols.append(col)

# Define preprocessing steps
numeric_preprocessor = SimpleImputer(strategy='median')
categorical_preprocessor = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

# Create ColumnTransformer to apply appropriate transformer based on data type
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_preprocessor, numeric_cols),
        ('cat', categorical_preprocessor, categorical_cols)
    ],
    remainder='drop'
)

# Define the regression model
model = LGBMRegressor(objective='regression', n_estimators=500, random_state=42)

# Create pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Fit the pipeline on training data
pipeline.fit(train_df[feature_columns], train_df[TARGET_COLUMN])

# Make predictions on test data
predictions = pipeline.predict(test_df[feature_columns])

# Create submission dataframe
submission_df = pd.DataFrame({
    'id': range(len(test_df)),
    'NAMPLT_CAP': predictions
})

# Save predictions to CSV
submission_df.to_csv(OUTPUT_PATH, index=False)