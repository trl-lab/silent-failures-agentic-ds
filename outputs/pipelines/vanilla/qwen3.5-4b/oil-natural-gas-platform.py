import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, Imputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

# Paths
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/oil-natural-gas-platform/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/oil-natural-gas-platform/data/test.parquet'
output_path = 'predictions.csv'

# Load Data
print("Loading data...")
df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

# Identify features and target
target_col = 'WATERDEPTH'
feature_cols = [col for col in df_train.columns if col != target_col]
X_train = df_train[feature_cols].copy()
y_train = df_train[target_col].copy()
X_test = df_test[feature_cols].copy()

# Initialize Preprocessor
# Using a ColumnTransformer to handle numeric (Imputation) and categorical (Encoding) automatically
ct = ColumnTransformer(transformers=[
    ('num', Pipeline(steps=[('imp', Imputer(strategy='mean'))]), 
     ['continuous']), # Will filter dynamic
])

# Since we need to detect types dynamically for the preprocessor
numeric_cols = []
categorical_cols = []

for col in feature_cols:
    if pd.api.types.is_numeric_dtype(df_train[col]):
        numeric_cols.append(col)
    else:
        # Handle categorical
        if df_train[col].isna().any():
            # Treat categorical with missing values; fillna for safety before encoding if needed
            # Actually, OneHotEncoder handles unknown via default_strategy='impute' or 'error'
            # We need to ensure strings are treated correctly.
            categorical_cols.append(col)
        else:
            categorical_cols.append(col)

# Clean categorical columns for OneHotEncoder (replace NaN with empty string or None handled by encoder)
# Ensure Object columns are encoded, Numeric are imputed
# To make it fully automatic as requested without hardcoding column types which might be tricky due to floating point vs int:
# Strategy: Detect dtype at runtime.

preproc = ColumnTransformer(
    transformers=[
        ('num', Imputer(strategy='median'), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='drop'
)

# Preprocess Training Data
X_train_processed = preproc.fit_transform(X_train)

# Preprocess Test Data
X_test_processed = preproc.transform(X_test)

# Build and Train Model
model = LGBMRegressor(
    objective='regression',
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbose=-1,
    device='cpu'
)

model.fit(X_train_processed, y_train)

# Make Predictions
predictions = model.predict(X_test_processed)

# Generate output DataFrame
test_indices = list(range(len(df_test)))
output_df = pd.DataFrame({
    'id': test_indices,
    'WATERDEPTH': predictions
})

# Save to CSV
output_df.to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")