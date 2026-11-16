import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Define paths
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/drug-enforcement/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/drug-enforcement/data/test.parquet'

# Load data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Define target column
target_col = 'classification'

# Extract features and target
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
X_test = test_df.drop(columns=[target_col])

# Determine column types
numeric_cols = X_train.select_dtypes(include=['int64', 'float64', 'bool']).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Handle potential empty columns if any
numeric_cols = [c for c in numeric_cols if c in X_train.columns]
cat_cols = [c for c in cat_cols if c in X_train.columns]

# Define transformers
num_transf = SimpleImputer(strategy='median')
cat_transf = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Create ColumnTransformer
ct = ColumnTransformer(transformers=[
    ('num', num_transf, numeric_cols),
    ('cat', cat_transf, cat_cols)
], remainder='drop')

# Fit preprocessor on training data
X_train_processed = ct.fit_transform(X_train)
X_test_processed = ct.transform(X_test)

# Determine objective based on number of classes
unique_classes = set(y_train.values)
num_classes = len(unique_classes)

if num_classes == 2:
    model = lgb.LGBMClassifier(objective='binary', random_state=42, verbose=-1)
else:
    model = lgb.LGBMClassifier(objective='multi', random_state=42, verbose=-1)

model.fit(X_train_processed, y_train)

# Predict
predictions = model.predict(X_test_processed)

# Create output dataframe with id and classification
ids = np.arange(len(predictions))
output_df = pd.DataFrame({'id': ids, 'classification': predictions})

# Save predictions
output_df.to_csv('predictions.csv', index=False)