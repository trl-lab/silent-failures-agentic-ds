import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Define file paths
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/kaggle/walmart-recruiting-store-sales-forecasting/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/kaggle/walmart-recruiting-store-sales-forecasting/data/test.parquet'
TARGET_COL = 'Weekly_Sales'

def main():
    # Load datasets
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    
    # Separate features and target
    # Ensure target column is removed from features list
    feature_cols = [col for col in train_df.columns if col != TARGET_COL]
    
    # Identify numeric and categorical columns
    numeric_cols = []
    categorical_cols = []
    
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    
    # Handle missing values automatically
    # Numeric: Median
    for col in numeric_cols:
        train_df[col] = train_df[col].fillna(train_df[col].median())
        
        # Note: test_df may also contain missing values in same columns, fill consistently
        test_df[col] = test_df[col].fillna(test_df[col].median())
        
    # Categorical: Mode
    for col in categorical_cols:
        train_mode = train_df[col].mode()
        if len(train_mode) == 0:
            fill_val = 'Unknown'
        else:
            fill_val = train_mode[0]
        train_df[col] = train_df[col].fillna(fill_val)
        
        # Fill test df missing values with same mode
        test_mode = test_df[col].mode()
        if len(test_mode) == 0:
            fill_val = 'Unknown'
        else:
            fill_val = test_mode[0]
        test_df[col] = test_df[col].fillna(fill_val)
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
    
    # Define model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('lgb_model', lgb.LGBMRegressor(
            objective='regression',
            metric='mae',
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            verbosity=-1,
            random_state=42,
            n_estimators=1000
        ))
    ])
    
    # Fit model on training data
    model.fit(train_df, train_df[TARGET_COL])
    
    # Prepare test data features
    X_test = test_df[feature_cols]
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Create output dataframe with required columns
    output_df = pd.DataFrame({
        'id': range(len(test_df)),
        'Weekly_Sales': predictions
    })
    
    # Save predictions to current working directory
    output_df.to_csv('predictions.csv', index=False)

if __name__ == '__main__':
    main()