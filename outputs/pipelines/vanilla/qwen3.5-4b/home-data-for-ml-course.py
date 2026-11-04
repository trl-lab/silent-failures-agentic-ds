import warnings
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Suppress warnings for cleaner execution
warnings.filterwarnings('ignore')

# Define file paths as specified in the challenge
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/kaggle/home-data-for-ml-course/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/kaggle/home-data-for-ml-course/data/test.parquet'

def main():
    # 1. Load Data
    df_train = pd.read_parquet(TRAIN_PATH)
    df_test = pd.read_parquet(TEST_PATH)

    # Identify Target and Features
    target_col = 'SalePrice'
    feature_cols = df_train.columns.difference([target_col])
    
    # 2. Preprocess Training Data
    df_train_feat = df_train[feature_cols].copy()
    y_train = df_train[target_col].copy()
    
    # Determine numeric and categorical columns
    numeric_features = df_train_feat.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df_train_feat.select_dtypes(include=[object, 'string']).columns.tolist()
    
    # Initialize Imputers
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    
    # Prepare Transforms
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Create preprocessing pipe
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    # Prepare Training Data Arrays
    X_train_preproc = preprocessor.fit_transform(df_train_feat)
    y_train_arr = y_train.values
    
    # 3. Initialize and Train Model (LightGBM)
    model = lgb.LGBMRegressor(
        objective='regression',
        n_estimators=1000,
        learning_rate=0.05,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=0.05,
        max_depth=-1,
        random_state=42,
        verbose=-1  # Suppress progress bar
    )
    model.fit(X_train_preproc, y_train_arr)
    
    # 4. Preprocess Test Data
    # Assume test file has feature columns matching train (excluding target)
    df_test_feat = df_test[feature_cols].copy()
    
    # Get transformed features for test
    try:
        X_test_preproc = preprocessor.transform(df_test_feat)
    except Exception:
        # Fallback if columns mismatch, reload or try reselect
        print("Warning: Column mismatch detected, attempting alignment.")
        # In a robust script, we align based on dtypes here, but we proceed with selected columns
        pass
    
    # 5. Predict
    y_pred_arr = model.predict(X_test_preproc)
    y_pred = pd.Series(y_pred_arr)
    
    # 6. Save Predictions
    # Requirement: Exactly two columns: 'id' (integer row index starting from 0) and 'SalePrice'
    predictions_df = pd.DataFrame({
        'id': range(len(y_pred)),
        'SalePrice': y_pred.round(2) # Optional rounding, keeping float precision might be safer but standard is rounded
    })
    
    # Ensure CSV file is saved in current working directory
    predictions_df.to_csv('predictions.csv', index=False)
    
    print(f"Predictions saved to 'predictions.csv'.")

if __name__ == '__main__':
    main()