import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb

# Configuration
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/michelin-ratings/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/michelin-ratings/data/test.parquet'

# Load Data
train_data = pd.read_parquet(train_path)
test_data = pd.read_parquet(test_path)

target_column = 'awardFor'

# Separate Features and Labels
train_y = train_data[target_column]
train_X = train_data.drop(columns=[target_column])

test_target_removed = False
if target_column in test_data.columns:
    test_X = test_data.drop(columns=[target_column])
    test_target_removed = True
else:
    test_X = test_data

# Identify column types
numeric_cols = []
categorical_cols = []

for col in train_X.columns:
    if pd.api.types.is_numeric_dtype(train_X[col]):
        numeric_cols.append(col)
    elif pd.api.types.is_object_dtype(train_X[col]) or pd.api.types.is_string_dtype(train_X[col]):
        categorical_cols.append(col)

# Define Preprocessing
numeric_transformer = SimpleImputer(strategy='median')
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Build Model Pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', lgb.LGBMClassifier(objective='multiclass', class_weight=None, max_depth=None, n_estimators=1000,
                                      learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                                      random_state=42, verbosity=-1))
])

# Fit Model
model.fit(train_X, train_y)

# Generate Predictions
test_processed = model.named_steps['preprocessor'].transform(test_X)
predicted_indices = model.named_steps['classifier'].predict(test_processed)

# Create Results DataFrame with required IDs (0-based integer row index starting from 0)
results = pd.DataFrame({
    'id': range(len(predicted_indices)),
    'awardFor': predicted_indices
})

# Save to CSV
results.to_csv('predictions.csv', index=False)