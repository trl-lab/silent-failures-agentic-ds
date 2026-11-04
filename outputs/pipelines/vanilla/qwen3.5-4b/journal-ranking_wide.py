import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack

# Configuration
train_path = "/path/to/user/workspace/ds_failures/datasets/strable/journal-ranking_wide/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/journal-ranking_wide/data/test.parquet"
target_col = "H index"

# Load Data
train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

# Identify Target and Features
y_train = train[target_col].values
train_features = train.drop(columns=target_col)

# Prepare Features
# Determine column types based on training data
all_feature_cols = [col for col in test.columns if col != target_col]
numerical_cols = []
categorical_cols = []

for col in all_feature_cols:
    if train[col].dtype == 'object' or train[col].dtype.name == 'category' or \
       pd.api.types.is_object_dtype(train[col]):
        categorical_cols.append(col)
    else:
        numerical_cols.append(col)

# Handle Missing Values and Categorical Encoding
numeric_imputer = SimpleImputer(strategy='median')
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Define Preprocessing Pipeline
ct = ColumnTransformer(transformers=[
    ('num', numeric_imputer, numerical_cols),
    ('cat', one_hot_encoder, categorical_cols)
])

# Define Model
model = Pipeline(steps=[
    ('preprocessor', ct),
    ('regressor', lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=-1, verbose=-1, random_state=42))
])

# Fit Model on Training Data
model.fit(train_features, y_train)

# Generate Predictions
pred_values = model.predict(test)

# Create Output DataFrame with required 'id' and 'H index' columns
results = pd.DataFrame({
    'id': range(len(test)),
    'H index': pred_values
})

# Save to predictions.csv
results.to_csv('predictions.csv', index=False)