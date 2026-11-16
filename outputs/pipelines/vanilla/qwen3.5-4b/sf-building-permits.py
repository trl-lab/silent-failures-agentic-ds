import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import lightgbm as lgb

# Define paths
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/sf-building-permits/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/sf-building-permits/data/test.parquet'

# Load data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Target column
target_column = 'time_to_approve'

# Separate features and target
feature_cols = [col for col in train_df.columns if col != target_column]
X_train = train_df[feature_cols].copy()
y_train = train_df[target_column].values
X_test = test_df[feature_cols].copy()

# Identify numeric and categorical features for preprocessing
numeric_features = []
categorical_features = []

for col in feature_cols:
    if pd.api.types.is_numeric_dtype(X_train[col]):
        numeric_features.append(col)
    else:
        categorical_features.append(col)

# Define preprocessor
numeric_transformer = SimpleImputer(strategy='median')
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Define model
model = lgb.LGBMRegressor(random_state=42, n_estimators=100, learning_rate=0.1, verbose=-1)

# Create pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# Train model
pipeline.fit(X_train, y_train)

# Predict on test set
predictions = pipeline.predict(X_test)

# Prepare output dataframe
output_df = pd.DataFrame({
    'id': range(len(predictions)),
    'time_to_approve': predictions
})

# Save to CSV
output_df.to_csv('predictions.csv', index=False)