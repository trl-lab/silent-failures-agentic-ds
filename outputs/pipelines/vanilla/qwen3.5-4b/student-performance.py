import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/uci/student-performance/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/student-performance/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Define columns
exclude_targets = ['G1', 'G2', 'G3']
feature_cols = [col for col in train_df.columns if col not in exclude_targets]

# Split features into categorical and numerical for preprocessing
categorical_cols = feature_cols[~feature_cols.apply(lambda x: isinstance(train_df[x].iloc[0], str))]
numerical_cols = feature_cols[~feature_cols.apply(lambda x: isinstance(train_df[x].iloc[0], str))]

# Create Transformer for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
])

# Create Transformer for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine transformers
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Prepare training data
X_train = train_df[feature_cols].copy()
y_train = train_df['G3'].values

# Prepare test data
X_test = test_df[feature_cols].copy()

# Apply preprocessing
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Define and train model
model = lgb.LGBRegressor(objective='regression', metric='rmse', 
                         random_state=42, learning_rate=0.1, max_depth=-1, num_leaves=31, verbose=-1)
model.fit(X_train_processed, y_train)

# Predict
predictions = model.predict(X_test_processed)

# Prepare output DataFrame
output_df = pd.DataFrame({'id': range(len(predictions)), 'G3': predictions})
output_df.to_csv('predictions.csv', index=False)