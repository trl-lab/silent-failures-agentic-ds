import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import catboost as cb

# Define paths
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/awarded-grants/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/awarded-grants/data/test.parquet"
OUTPUT_PATH = "predictions.csv"

# Load data
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

# Identify Target and Feature Columns
target_col = "Financial Assistance"
# Ensure target column exists
assert target_col in train_df.columns, f"Target column {target_col} not found in train data."

# Get candidate feature columns
feature_columns = [col for col in train_df.columns if col != target_col and col.lower() != "id"]

# Remove columns that might exist in both (ID matching issues) or duplicates
df_train = train_df.copy()
df_test = test_df.copy()

# Separate IDs if present in the test set to be consistent with output requirements
# We need to output 'id' starting from 0 regardless of test df contents
n_samples = len(test_df)

# Prepare Preprocessing Pipeline
# 1. Identify categorical and numeric columns in training data
categorical_cols = df_train[feature_columns].select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = df_train[feature_columns].select_dtypes(include=['int', 'float', 'bool']).columns.tolist()

print(f"Categorical Columns Found: {len(categorical_cols)}")
print(f"Numerical Columns Found: {len(numerical_cols)}")

# Construct ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('imputer_num', SimpleImputer(strategy='median'), numerical_cols),
        ('imputer_cat', SimpleImputer(strategy='most_frequent'), categorical_cols),
        ('one_hot', OneHotEncoder(sparse_output=False, handle_unknown='ignore', categories='auto'), categorical_cols)
    ]
)

# Wrap target in a simple pipeline to apply preprocessing only to features
# We create a dummy pipeline for consistency
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', cb.CatBoostRegressor(iterations=1000, verbose=False, random_state=42, silent=True, learning_rate=0.05, depth=4, loss_function="RMSE"))
])

# Fit the pipeline
full_pipeline.fit(df_train[feature_columns], train_df[target_col])

# Transform and predict
test_features = test_df[feature_columns]
preds = full_pipeline.predict(test_features)

# Create output dataframe
results = pd.DataFrame({
    "id": range(n_samples),
    "Financial Assistance": preds
})

# Save predictions
results.to_csv(OUTPUT_PATH, index=False)