import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

# Define file paths
train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/avito-demand-prediction/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/avito-demand-prediction/data/test.parquet'
output_path = './predictions.csv'

try:
    # Load data
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    # Separate features and target
    target_col = 'deal_probability'
    train_features = train_df.drop(columns=[target_col])
    test_features = test_df.copy()
    
    # Get column names
    feature_columns = train_features.columns.tolist()
    test_features = test_features[feature_columns]
    
    # Initialize transformers for handling categories and missing values
    # Numeric imputation
    num_transformer = Pipeline(steps=[
        ('imputer', pd.Series.fillna(method='ffill') if len(train_features.columns > 0) else None)
    ])
    
    # We need a strategy to handle missing values and categoricals
    # 1. Identify column types
    dtypes = {col: train_features[col].dtype for col in feature_columns}
    numeric_cols = []
    categorical_cols = []
    
    for col in feature_columns:
        dtype = dtypes[col]
        if pd.api.types.is_numeric_dtype(dtype):
            # Handle NaN for numeric
            numeric_cols.append(col)
        elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            # Treat as categorical
            categorical_cols.append(col)
        else:
            # Fallback to treat unknown dtypes as categorical to be safe
            categorical_cols.append(col)
            
    # Preprocess Train
    # Replace NaN in numeric with median
    for col in numeric_cols:
        median_val = train_features[col].median()
        train_features[col] = train_features[col].fillna(median_val)
        
    # Preprocess Test
    for col in numeric_cols:
        median_val = test_features[col].median()
        test_features[col] = test_features[col].fillna(median_val)
        
    # Encode Categorical Columns
    # Using LabelEncoder for automatic handling
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        train_series = train_features[col].astype(str)
        test_series = test_features[col].astype(str)
        le.fit(pd.concat([train_series, test_series]))
        train_features[col] = le.transform(train_series)
        test_features[col] = le.transform(test_series)
        encoders[col] = le
        
    # Convert everything to float to ensure matrix consistency
    for col in categorical_cols:
        train_features[col] = train_features[col].astype(float)
        test_features[col] = test_features[col].astype(float)
    
    # Prepare matrices
    X_train = train_features.values
    y_train = train_df[target_col].values
    X_test = test_features.values
    
    # Normalize Target (optional but helps with RMSE stability)
    # Since trees are invariant to target scaling, but we need probability-like output
    # We keep target as is (probability 0-1)
    
    # Model: LightGBM is fast and handles missing values/numerical data well
    model = LGBMRegressor(
        objective='regression',
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        random_state=42,
        verbose=-1
    )
    
    # Fit model
    model.fit(X_train, y_train, eval_set=(X_train, y_train), callbacks=[_print_fit])
    
    # Predict
    predictions = model.predict(X_test)
    
    # Prepare output dataframe
    # Requirement: id (integer row index starting from 0)
    test_ids = test_df.reset_index(drop=True)
    
    output_df = pd.DataFrame({
        'id': range(len(test_ids)),
        'deal_probability': predictions
    })
    
    output_df.to_csv(output_path, index=False)
    
    print(f"Predictions saved to {output_path}")
    
except Exception as e:
    print(f"Error occurred: {e}")
    raise e

# Suppressing the callback warning in the actual run
def _print_fit(self):
    pass