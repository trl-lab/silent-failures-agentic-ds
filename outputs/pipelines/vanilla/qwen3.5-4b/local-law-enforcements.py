import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import lightgbm as lgb

# File paths
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/local-law-enforcements/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/local-law-enforcements/data/test.parquet'

# Load data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Target column
target_col = 'number_of_officers'

# Prepare features
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
X_test = test_df.drop(columns=[target_col])

# Automatically identify numerical and categorical columns based on training data
numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median'))
        ])), numerical_columns],
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])), categorical_columns
    ])

# Initialize model
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train model
model.fit(preprocessor.fit_transform(X_train), y_train)

# Transform and predict
processed_test = preprocessor.transform(X_test)
predictions = model.predict(processed_test)

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(predictions)),
    'number_of_officers': predictions
})
output_df.to_csv('predictions.csv', index=False)