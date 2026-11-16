import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import OneHotEncoder, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Configuration
PATH_TRAIN = '/path/to/user/workspace/ds_failures/datasets/strable/grant/data/train.parquet'
PATH_TEST = '/path/to/user/workspace/ds_failures/datasets/strable/grant/data/test.parquet'
TARGET_COL = 'Federal Funds'
OUTPUT_FILE = 'predictions.csv'

def main():
    # Load Data
    print("Loading data...")
    train_df = pd.read_parquet(PATH_TRAIN)
    test_df = pd.read_parquet(PATH_TEST)
    
    # Prepare Data
    print("Processing data...")
    
    # Drop target column to get features
    train_y = train_df[TARGET_COL]
    train_df = train_df.drop(columns=[TARGET_COL])
    
    test_target_col = test_df.columns.tolist()
    if TARGET_COL in test_df.columns:
        test_y = test_df[TARGET_COL]
        test_df = test_df.drop(columns=[TARGET_COL])
    else:
        test_y = None
    
    # Determine feature columns (excluding target)
    feature_cols = [col for col in train_df.columns if col not in ['id']]
    # Ensure consistent feature columns between train and test
    common_features = sorted(train_df.columns.tolist()) # Assuming test has same structure but maybe without target
    if TARGET_COL in test_df.columns:
        common_features = sorted([col for col in train_df.columns if col != TARGET_COL])
    else:
        common_features = [col for col in train_df.columns]
    
    train_df_clean = train_df[common_features]
    test_df_clean = test_df[common_features]
    
    # Preprocessing Strategy: Automatic handling of missing values and categoricals
    # Identify numeric vs categorical columns
    numeric_cols = []
    categorical_cols = []
    for col in common_features:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            numeric_cols.append(col)
        elif pd.api.types.is_object_dtype(train_df[col]):
            categorical_cols.append(col)
        elif pd.api.types.is_bool_dtype(train_df[col]):
            categorical_cols.append(col)
    
    # Pipeline Definition
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='drop'
    )
    
    scaler = None # Optional scaling, XGBoost handles unscaled data well but scaling helps convergence
    # For simplicity and adherence to "standard libraries" without extra dependencies on scaling strategies, 
    # we rely on the pipeline above for categorical/missing handling.
    
    # Assemble Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBRegressor(objective='regression', eval_metric='rmse', n_estimators=500, verbosity=0))
    ])
    
    # Fit
    print("Training model...")
    try:
        pipeline.fit(train_df_clean, train_y)
    except Exception as e:
        print(f"Error fitting model: {e}")
        return
    
    # Transform and Predict
    train_pred = pipeline.predict(train_df_clean)
    test_pred = pipeline.predict(test_df_clean)
    
    # Prepare Output
    print(f"Saving predictions to {OUTPUT_FILE}...")
    output_data = pd.DataFrame({
        'id': list(range(len(test_df))), # Row index starting from 0
        'Federal Funds': test_pred
    })
    
    output_data.to_csv(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == '__main__':
    main()