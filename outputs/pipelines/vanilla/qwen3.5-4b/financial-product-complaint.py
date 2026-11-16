import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import lightgbm as lgb

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/financial-product-complaint/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/financial-product-complaint/data/test.parquet'
target_col = 'Company response to consumer'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Prepare features
X_train = train_df.drop(columns=[target_col]).copy()
y_train = train_df[target_col]
X_test = test_df.drop(columns=[target_col]).copy()

# Separate numeric and categorical features for automatic handling
# 'number' includes int, float, complex, datetime in pandas selection
numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=['number']).columns.tolist()

# Create Preprocessor Pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ], resume='none'), numeric_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ], resume='none'), categorical_features)
])

# Create and fit Model
model_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('classifier', lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, random_state=42, verbose=-1, max_depth=-1))
])

model_pipeline.fit(X_train, y_train)

# Generate Predictions
y_pred = model_pipeline.predict(X_test)

# Save Output
result = pd.DataFrame({
    'id': range(len(test_df)),
    target_col: y_pred
})
result.to_csv('predictions.csv', index=False)