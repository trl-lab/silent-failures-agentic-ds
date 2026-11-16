import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Configuration
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/hypertension-control-wide/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/hypertension-control-wide/data/test.parquet"
TARGET_NAME = "Estimated Percentage of Patients with Controlled Blood Pressure"
OUTPUT_FILE = "predictions.csv"

def main():
    # Load Data
    print("Loading data...")
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    
    # Identify Columns
    # Assuming target exists, otherwise raise error
    if TARGET_NAME not in train_df.columns:
        print(f"Warning: Target column '{TARGET_NAME}' not found.")
        return
    
    # Exclude target from features
    feature_cols = [col for col in train_df.columns if col != TARGET_NAME]
    
    # Identify if 'id' exists in data to handle consistency
    # We will generate ids manually from 0 to N-1 for output as per requirements
    # But we should remove existing id columns from feature set if present to prevent leakage
    columns_to_remove = ['id'] if 'id' in train_df.columns else []
    feature_cols = [col for col in feature_cols if col not in columns_to_remove]
    
    if len(feature_cols) == 0:
        print("Error: No features found after excluding target and reserved columns.")
        return
    
    # Extract Features
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    y_train = train_df[TARGET_NAME].values
    
    # Handle Missing Values and Types
    # Convert object/categorical columns appropriately for LightGBM
    # LightGBM handles string columns automatically if they are treated as categorical, 
    # but explicit conversion to 'category' ensures stability across versions.
    
    def prepare_dataframe(df, feature_cols):
        result = df.copy()
        for col in feature_cols:
            if result[col].dtype == 'object' or str(result[col].dtype).startswith('float'):
                # Handle string columns
                if result[col].dtype == 'object':
                    result[col] = result[col].astype('category')
                elif pd.api.types.is_numeric_dtype(result[col]):
                    result[col] = result[col].fillna(np.mean(result[col].dropna()))
            else:
                # General fill for other dtypes
                result[col] = result[col].fillna(np.nan)
        return result
    
    X_train_processed = prepare_dataframe(X_train, feature_cols)
    X_test_processed = prepare_dataframe(X_test, feature_cols)
    
    # Handle remaining NaNs in numeric features
    # LightGBM handles NaN natively for tree building, so we keep them as float/NaN
    # but ensure consistent dtype
    X_train_processed = X_train_processed.apply(lambda x: pd.to_numeric(x, errors='coerce') if pd.api.types.is_numeric_dtype(original_column) else x).astype(object) # Simplified logic for mixed types is complex, relying on LGBM default
        
    # Re-process cleanly using pd.Categorical
    # Simplest robust way:
    X_train_clean = X_train.copy()
    X_test_clean = X_test.copy()
    
    for col in feature_cols:
        if X_train_clean[col].dtype == 'object':
            X_train_clean[col] = X_train_clean[col].astype('category')
            X_test_clean[col] = X_test_clean[col].astype('category')
    
    # Train Model
    print("Training model...")
    params = {
        'objective': 'binary', # Assumed binary classification based on metric
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 20,
        'seed': 42,
        'verbose': -1,
        'early_stopping_rounds': 100
    }
    
    train_data = lgb.Dataset(X_train_clean, label=y_train)
    val_data = None # No validation split in simple case, or split if large
    if len(y_train) > 1000:
        _, X_val, _, y_val = train_test_split(X_train_clean, y_train, test_size=0.2, random_state=42)
        params['validation_set'] = lgb.Dataset(X_val, label=y_val)
        train_data, val_data = lgb.Dataset(train_data), val_data
        
    clf = lgb.LGBMClassifier(**params)
    clf.fit(train_data, y_train, eval_set=val_data, early_stopping=True)
    
    # Predictions
    print("Generating predictions...")
    preds = clf.predict_proba(X_test_clean)[:, 1]
    
    # Create Output DataFrame
    output_df = pd.DataFrame({
        'id': list(range(len(test_df))),
        TARGET_NAME: preds
    })
    
    # Save
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()