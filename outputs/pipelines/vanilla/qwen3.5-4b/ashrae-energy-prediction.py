import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# Configuration Paths
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/kaggle/ashrae-energy-prediction/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/kaggle/ashrae-energy-prediction/data/test.parquet'
OUTPUT_PATH = 'predictions.csv'
TARGET_COLUMN = 'meter_reading'

def main():
    # 1. Load Data
    df_train = pd.read_parquet(TRAIN_PATH)
    df_test = pd.read_parquet(TEST_PATH)
    
    # 2. Identify Target and Valid Features
    # Ensure test set has same columns as train (excluding target for prediction)
    train_cols = set([col for col in df_train.columns if col != TARGET_COLUMN])
    test_cols = set([col for col in df_test.columns])
    
    # Select common columns between train and test
    final_cols = sorted(list(train_cols.intersection(test_cols)))
    
    # Filter DataFrames to these columns
    df_train_feat = df_train[train_cols]
    df_test_feat = df_test[list(final_cols)]
    
    # 3. Automatic Handling of Missing Values
    # Distinguish numeric and categorical columns based on dtypes
    num_cols = [col for col in final_cols if df_train_feat[col].dtype in ['int64', 'float64', 'int32', 'float32']]
    cat_cols = [col for col in final_cols if df_train_feat[col].dtype.name not in ['int64', 'float64', 'int32', 'float32']]
    
    # Fill missing values
    # Numeric: Median
    if num_cols:
        train_numeric_median = df_train_feat[num_cols].median(numeric_only=True).astype('float')
        test_numeric_median = df_test_feat[num_cols].median(numeric_only=True).astype('float')
        df_train_feat[num_cols] = df_train_feat[num_cols].combine_first(train_numeric_median.stack().to_frame().rename(None)).fillna(train_numeric_median[nums])
        # Fix above logic for clarity
        for col in num_cols:
            val = df_train_feat[col].dropna().median() if df_train_feat[col].notna().any() else 0.0
            df_train_feat[col] = df_train_feat[col].fillna(val)
            df_test_feat[col] = df_test_feat[col].fillna(df_train_feat[col].fillna(val)) # Use train stats
        
        # Re-implement cleaner imputation logic
        for col in num_cols:
            fill_val = df_train_feat[col].median()
            df_train_feat[col] = df_train_feat[col].fillna(fill_val)
            df_test_feat[col] = df_test_feat[col].fillna(fill_val)

    # Categorical: Mode (Most Frequent)
    for col in cat_cols:
        # Find mode excluding NaNs
        mode_values = df_train_feat[col].dropna().mode()
        if len(mode_values) > 0:
            mode_val = mode_values[0]
            # Fill train
            df_train_feat[col] = df_train_feat[col].fillna(mode_val)
            # Fill test
            df_test_feat[col] = df_test_feat[col].fillna(mode_val)
        else:
            # Fallback for all-NaN columns
            df_train_feat[col] = 0
            df_test_feat[col] = 0
            
    # 4. Initialize Model
    # CatBoost handles mixed types (numeric/categorical) and missing values automatically with correct cat_features list
    params = {
        'task_name': 'Regression',
        'loss_function': 'RMSE',
        'metric': 'RMSE',
        'iterations': 2000,
        'depth': 8,
        'learning_rate': 0.05,
        'verbose': False,
        'random_seed': 42,
        'early_stopping_rounds': 100,
        'device_type': 'CPU', # Ensures running on available resources without GPU checks
        'eval_metric': 'RMSE'
    }
    
    cat_feats_list = cat_cols
    
    model = CatBoostRegressor(param_name='', **params) # Init without params
    model.fit(df_train_feat, df_train_feat[TARGET_COLUMN], 
              cat_features=cat_feats_list, 
              verbose=False, 
              eval_set=[(df_train_feat, df_train_feat[TARGET_COLUMN])])
    
    # 5. Predictions
    predictions = model.predict(df_test_feat)
    
    # 6. Save Results
    # Requirement: id (int row index starting from 0) and meter_reading
    result_df = pd.DataFrame({
        'id': range(len(predictions)),
        'meter_reading': predictions
    })
    
    result_df.to_csv(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    main()