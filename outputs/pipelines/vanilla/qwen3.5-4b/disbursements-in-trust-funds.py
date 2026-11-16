import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb

# Define file paths
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/disbursements-in-trust-funds/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/disbursements-in-trust-funds/data/test.parquet'

# Load data
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

# Target column
TARGET_COL = 'Cash Disbursements (US$)'

# Feature columns (all except target)
FEATURE_COLS = [col for col in train_df.columns if col != TARGET_COL]

# Identify numeric and categorical columns
# Treat datetime as numeric for scaling purposes, though treating as categorical might also work,
# standard treatment is numeric/timestamp. Here we separate numeric/object.
numeric_cols = []
categorical_cols = []

for col in FEATURE_COLS:
    col_type = train_df[col].dtype
    if pd.api.types.is_numeric_dtype(col_type) or pd.api.types.is_datetime64_any_dtype(col_type):
        numeric_cols.append(col)
    else:
        categorical_cols.append(col)

# Define preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Set up ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='drop'
)

# Define Model Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', lgb.LGBMRegressor(objective='regression', random_state=42, verbose=-1))
])

# Fit the model on training data
X_train = train_df[FEATURE_COLS]
y_train = train_df[TARGET_COL]
model.fit(X_train, y_train)

# Predict on test data
X_test = test_df[FEATURE_COLS]
predictions = model.predict(X_test)

# Prepare output dataframe with required columns
output_df = pd.DataFrame({
    'id': range(len(test_df)),
    'Cash Disbursements (US$)': predictions
})

# Save predictions
output_df.to_csv('predictions.csv', index=False)