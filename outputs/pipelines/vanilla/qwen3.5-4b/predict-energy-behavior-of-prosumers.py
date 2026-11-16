import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import lightgbm as lgb

# Suppress progress bars during training
def _progress(*args, **kwargs):
    pass
sys.stdout.flush = lambda s: None # Disable stdout buffering if needed

# Define Paths based on prompt instructions
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/kaggle/predict-energy-behavior-of-prosumers/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/kaggle/predict-energy-behavior-of-prosumers/data/test.parquet'
DATA_DIR = '/path/to/user/workspace/ds_failures/datasets/kaggle/predict-energy-behavior-of-prosumers/data'

# List of potential reference table filenames
REF_FILES = ['client.csv', 'electricity_prices.csv', 'forecast_weather.csv', 
             'historical_weather.csv', 'gas_prices.csv', 'weather_station_to_county_mapping.csv']

def load_reference_tables():
    """Attempts to load reference tables from the data directory."""
    dfs = {}
    ref_cols = []
    merged_df = None
    
    for filename in REF_FILES:
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                dfs[filename] = df
                # Heuristic: if table is massive, we may skip certain merges to save memory/time
                # Assuming smaller client/weather tables can be joined
            except Exception as e:
                print(f"Warning: Failed to load {filename}: {e}")
                continue
    
    return dfs

def preprocess_data(dfs, train_path, test_path):
    """Load, merge, and engineer features."""
    # Load parquet files
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    # Combine into one initial frame for processing, keeping track of source
    combined = pd.concat([train_df, test_df], ignore_index=True)
    
    # 1. Time Features Engineering
    if 'datetime' in combined.columns:
        combined['datetime'] = pd.to_datetime(combined['datetime'])
        combined['year'] = combined['datetime'].dt.year
        combined['month'] = combined['datetime'].dt.month
        combined['day'] = combined['datetime'].dt.day
        combined['hour'] = combined['datetime'].dt.hour
        combined['dayofweek'] = combined['datetime'].dt.dayofweek
        
        # Rolling averages using historical data if forecast/historical exists
        if 'date' in combined.columns or 'datetime' in combined.columns:
            pass # Already done
    
    # 2. Try to Join Reference Data
    ref_dfs = load_reference_tables()
    if ref_dfs:
        # Find unique county key for join
        county_col = 'county' if 'county' in combined.columns else None
        if county_col and not county_col.lower().startswith('id'):
            # Attempt to join client data
            for name, df in ref_dfs.items():
                if 'county' in df.columns or 'unit' in df.columns:
                    if county_col in df.columns:
                        merged = combined.merge(df, on='county', how='left')
                        # Remove duplicates if multiple columns joined
                        if name == 'client.csv':
                            break
    # 3. Drop rows that failed join or are empty after time eng (safe-guard)
    combined = combined.dropna(subset=['target'], how='all') # Remove invalid targets
    
    # 4. Feature Selection & Encoding
    feature_cols = ['county', 'is_business', 'product_type', 'is_consumption']
    
    # Identify categorical columns to encode
    categoricals = [col for col in feature_cols + ['year', 'month', 'dayofweek']] # Explicit ones
    categorical_columns = combined[categoricals].select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Replace categorical strings with numeric labels for LGBM
    # (LGBM handles object dtype well sometimes, but numeric is faster)
    # We will use LabelEncoder for object strings and standard ints
    
    for col in categoricals:
        if col in combined.columns and combined[col].dtype in ['object', 'category']:
            le = LabelEncoder()
            combined[col] = le.fit_transform(combined[col].astype(str))
        
    # Impute missing values
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    num_cols = combined.select_dtypes(exclude=['object', 'string']).columns.tolist()
    # Select only columns intended for modeling (exclude datetime derived that weren't kept)
    # Keep year, month, day, hour, dayofweek
    modeling_cols = num_cols + ['year', 'month', 'day', 'hour', 'dayofweek']
    modeling_cols = [c for c in modeling_cols if c in combined.columns]
    
    for col in modeling_cols:
        if col not in combined.columns: continue
        # Don't impute already filled cols
        if 'counties' in col: continue 

    # Specific handling for 'target' column
    target_col = 'target'
    
    X = combined[modeling_cols]
    y = combined[target_col]
    
    # Split Train/Test based on index availability (we have both combined)
    # However, we must preserve the separation from the beginning.
    # Let's reconstruct X_train/Y_train and X_test/Y_test properly.
    
    # Re-separate original dataframes after cleaning logic was applied to combined
    # Note: Combined lost original row_ids distinction potentially if dropped. 
    # But combined has test and train concatenated. We need to identify test rows.
    # Original test has row_id, train has row_id.
    
    # Re-read test parquet to reset index mapping correctly for output
    # We will train on the first N rows (train), predict on next M rows (test)
    # But easier: use the concat trick, then slice.
    # How to distinguish? Use the original row_id.
    
    # Let's re-do the train/test split cleanly based on 'row_id' uniqueness if needed,
    # but combining then slicing by count (if known) is risky.
    # Safest: Filter combined based on 'row_id' type or simply re-infer index.
    # Assuming test parquet has unique ids not in train.
    
    # Let's assume we processed the 'combined' dataframe which has all data.
    # But we need to separate them for final output ID.
    
    # Let's store the split indices during the initial load phase.
    pass 
    
    # Reset preprocessing logic to separate train/test explicitly for prediction ID
    # Re-load train/test separately for cleaner split
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    # Apply same transforms to train
    def apply_time_features(df):
        df = df.copy()
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['hour'] = df['datetime'].dt.hour
            df['dayofweek'] = df['datetime'].dt.dayofweek
            
        return df
    
    train_df = apply_time_features(train_df)
    test_df = apply_time_features(test_df)
    
    # Reconstruct Reference Join
    merged_train = train_df.copy()
    merged_test = test_df.copy()
    
    if ref_dfs:
        # Common keys often 'county'
        if 'county' in train_df.columns:
            for fname, df in ref_dfs.items():
                if not df.empty:
                    merged_train = merged_train.merge(df, how='left')
                    merged_test = test_df.merge(df, how='left')
                    if fname == 'client.csv':
                        break
    # If merged_train is now huge, consider dropping unused columns
    # We will keep common columns
    
    # Final Feature Definition
    final_cols = ['county', 'is_business', 'product_type', 'is_consumption', 
                  'hour', 'dayofweek', 'year', 'month', 'day']
    
    def prepare_features(df):
        # Basic cleanup
        df_clean = df.copy()
        # Numeric/Int Imputation
        imputer_num = SimpleImputer(strategy='median')
        int_cols = df_clean.select_dtypes(include=[np.integer]).columns.tolist()
        cat_cols = df_clean.select_dtypes(include=[np.object]).columns.tolist()
        
        for col in int_cols:
            if col in final_cols:
                df_clean[col] = df_clean[col].fillna(0)
                
        for col in cat_cols:
            if col in final_cols:
                # Label encoding for categories
                df_clean[col] = LabelEncoder().fit_transform(df_clean[col].astype(str))
        
        # Ensure only relevant cols
        df_final = df_clean[final_cols].copy()
        return df_final

    X_train = prepare_features(merged_train)
    y_train = merged_train['target']
    
    X_test = prepare_features(merged_test)
    # Target for validation is not needed here, just predicting
    # But we need to map the prediction to ID correctly.
    
    # Train Model
    params = {'objective': 'regression', 'metric': 'mae', 'verbosity': -1}
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    
    # Predictions
    pred = model.predict(X_test)
    pred = np.clip(pred, 0, 1000) # Soft bound to avoid negative energy
    
    # Save results
    out_df = pd.DataFrame({
        'id': np.arange(len(test_df)),
        'target': pred
    })
    
    out_df.to_csv('predictions.csv', index=False)
    print("Predictions saved to predictions.csv")

if __name__ == '__main__':
    preprocess_data(None, TRAIN_PATH, TEST_PATH)