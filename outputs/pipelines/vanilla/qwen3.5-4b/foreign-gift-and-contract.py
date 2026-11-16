import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Paths defined in requirements
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/foreign-gift-and-contract/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/foreign-gift-and-contract/data/test.parquet'
OUTPUT_PATH = 'predictions.csv'
TARGET_COL = 'Amount'

# Load data
df_train = pd.read_parquet(TRAIN_PATH)
df_test = pd.read_parquet(TEST_PATH)

# Identify columns to predict and exclude target
features = [col for col in df_train.columns if col != TARGET_COL]

# Separate numeric and categorical columns for appropriate preprocessing
numeric_cols = [col for col in features if pd.api.types.is_numeric_dtype(df_train[col]) or pd.api.types.is_bool_dtype(df_train[col])]
categorical_cols = [col for col in features if pd.api.types.is_object_dtype(df_train[col]) or pd.api.types.is_categorical_dtype(df_train[col])]

# Define preprocessing pipelines for each type
num_imputer = SimpleImputer(strategy='median')
cat_enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Preprocess categorical columns: Fill missing values with 'UNKNOWN' string to handle in OneHotEncoder
cat_cols_data = []
for col in categorical_cols:
    # Ensure missing values in object categories are replaced to avoid NaN errors in OneHotEncoder
    # Simple strategy: drop or map. Map is safer for prediction.
    # Since we want to preserve existing rows, fill NA with a unique string
    temp_series = df_train[col]
    # Check if NA exists
    if temp_series.isna().any():
        # Fill NaN with 'UNKNOWN' string. 
        # Note: This is a manual replacement step to simulate automated handling in a pipeline context
        # Without this, OneHotEncoder would throw error on NaN in object columns
        # However, for object columns, we need to convert NA to a known category before encoding.
        # Using 'UNKNOWN' as a placeholder category is safe if we ensure consistency.
        # We must be careful: if we fill 'NaN' in train, we must fill 'NaN' in test.
        df_train.loc[df_train[col].isna(), col] = 'UNKNOWN'
        df_test.loc[df_test[col].isna(), col] = 'UNKNOWN'
        cat_cols_data.append(('nan_fill', col))
        # Re-process to include 'UNKNOWN'
        # Since we modified in place, proceed to next logic below.
        
cat_cols_prepared = categorical_cols.copy()

# Construct ColumnTransformer
ct = ColumnTransformer(transformers=[
    ('imputer', num_imputer, numeric_cols),
    ('onehot', cat_enc, categorical_cols)
])

# Construct Pipeline
pipe = Pipeline(steps=[
    ('preprocess', ct),
    ('model', lgb.LGBMRegressor(objective='regression', random_state=42, n_jobs=-1))
])

# Fit model on Training Data
X_train = pipe.preprocess.fit_transform(df_train[features])
y_train = df_train[TARGET_COL]
pipe.fit(X_train, y_train)

# Predict on Test Data
X_test = pipe.preprocess.transform(df_test[features])
y_pred = pipe.predict(X_test)

# Prepare Output DataFrame with 'id' starting from 0
output_df = pd.DataFrame({'id': range(len(df_test)), 'Amount': y_pred})

# Save to CSV
output_df.to_csv(OUTPUT_PATH, index=False)