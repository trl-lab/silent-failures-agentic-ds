import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.impute import SimpleImputer, MostFrequentImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import os

def main():
    # Load data
    train_path = '/path/to/user/workspace/ds_failures/datasets/strable/us-school-bus-fleet/data/train.parquet'
    test_path = '/path/to/user/workspace/ds_failures/datasets/strable/us-school-bus-fleet/data/test.parquet'
    
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    # Configuration
    target_col = 'Bus age'
    
    # Feature extraction
    feature_cols = [col for col in test_df.columns if col != target_col]
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[target_col]
    
    # Preprocessing: Identify numeric and categorical columns for automatic handling
    # We use a pipeline to handle imputation and encoding automatically
    
    # Check dtypes to prepare pipeline
    num_cols = []
    cat_cols = []
    # Assume float/int/bool/negative are numeric, others are categorical (strings)
    # Handling missing values automatically within sklearn preprocessing
    
    # Strategy: Fill missing values, encode categorical columns
    processor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('encoder', None) # Keep as float
            ]), ['dummy']) # Placeholder for dynamic selection
        ], remainder='passthrough'
    )
    
    # Simpler robust approach for this specific environment without deep inspection:
    # Use LightGBM which handles categorical features if we pass strings and define 'auto_encoder=True'
    # However, to ensure 'automatic' handling of missing values, we will preprocess.
    
    # Convert categorical objects to strings and fill missing with mode for robustness
    # Ensure all features are processed consistently
    
    # Create a simple pipeline for handling
    transformers = []
    feature_names = []
    for col in feature_cols:
        val_counts = X_train[col].value_counts()
        # Identify if mostly categorical
        # For safety in this script without complex heuristics: 
        # We treat everything as numeric for lightgbm if possible, 
        # but handle string columns by filling mode.
        
        col_dtype = X_train[col].dtype
        
        if str(col_dtype) == 'object' or str(col_dtype) == 'string':
            cat_col = True
        else:
            cat_col = False
            
        if cat_col:
            # Fill with mode for categorical
            imputer_obj = SimpleImputer(strategy='constant', fill_value='mode')
            # For string fill_value, sklearn handles 'fill_value' for imputer as object
            transformers.append(('cat', imputer_obj))
        else:
            # Fill with median
            transformers.append(('num', SimpleImputer(strategy='median')))
            
        feature_names.append(col)
        
    # Construct pipeline with LightGBM
    pipe = Pipeline(steps=[
        ('preprocessor', ColumnTransformer(transformers=transformers,remainder=None)),
        ('model', lgb.LGBMRegressor(objective='regression', learning_rate=0.05, n_estimators=200, subsample=0.8, colsample_bytree=0.8))
    ])
    
    # Train
    pipe.fit(X_train, y_train)
    
    # Predict
    y_pred = pipe.predict(X_test)
    
    # Prepare output
    results_df = pd.DataFrame({
        'id': range(len(test_df)),
        'Bus age': y_pred
    })
    
    # Save predictions
    output_path = 'predictions.csv'
    results_df.to_csv(output_path, index=False)
    
    print(f"Predictions saved to {output_path}")

if __name__ == '__main__':
    main()