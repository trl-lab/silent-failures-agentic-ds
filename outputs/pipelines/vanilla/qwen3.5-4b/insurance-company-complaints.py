import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/insurance-company-complaints/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/insurance-company-complaints/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Recovery'

# Select feature columns excluding target
feature_cols = [col for col in train_df.columns if col != target_col]

# Split features into numeric and categorical based on train data
num_cols = []
cat_cols = []
for col in feature_cols:
    dtype = train_df[col].dtype
    if dtype == 'object' or dtype.name == 'string':
        cat_cols.append(col)
    elif pd.api.types.is_numeric_dtype(dtype):
        num_cols.append(col)

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ],
    remainder='passthrough'
)

# Define pipeline
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', lgb.LGBMRegressor(objective='regression', verbose=-1, random_state=42, n_estimators=500))
])

# Handle missing target values in training data (filter rows)
valid_mask = ~train_df[target_col].isna()
train_df_clean = train_df[valid_mask][feature_cols]
y_train = train_df_clean[target_col]

# Clean test data to match feature columns
X_test = test_df[feature_cols]

# Fit and Predict
pipeline.fit(train_df_clean, y_train)
predictions = pipeline.predict(X_test)

# Create output dataframe with required columns
id_col = range(len(test_df))
result_df = pd.DataFrame({'id': id_col, 'Recovery': predictions})

# Save to CSV
result_df.to_csv('predictions.csv', index=False)