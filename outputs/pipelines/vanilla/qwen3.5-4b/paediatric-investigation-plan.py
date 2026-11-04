import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict

def load_and_prepare_data():
    train_path = "/path/to/user/workspace/ds_failures/datasets/strable/paediatric-investigation-plan/data/train.parquet"
    test_path = "/path/to/user/workspace/ds_failures/datasets/strable/paediatric-investigation-plan/data/test.parquet"
    
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    
    target_col = 'Decision type'
    feature_cols = [col for col in df_train.columns if col != target_col]
    
    X_train_raw = df_train[feature_cols].copy()
    X_test_raw = df_test[feature_cols].copy()
    y_train = df_train[target_col].copy()
    
    return df_train, df_test, X_train_raw, X_test_raw, y_train, target_col, feature_cols

def main():
    try:
        _, _, X_train_raw, X_test_raw, y_train, target_col, feature_cols = load_and_prepare_data()
        
        # Define Preprocessors Dictionary
        precs = OrderedDict()
        for col in feature_cols:
            # Check for string/object type which indicates categorical
            if pd.api.types.is_object_dtype(X_train_raw[col].dtype) or \
               (not pd.api.types.is_numeric_dtype(X_train_raw[col].dtype) and X_train_raw[col].dtype.kind != 'f'):
                # Categorical
                precs[col] = {'type': 'encoder'}
            else:
                # Numerical
                precs[col] = {'type': 'imputer'}
        
        # Initialize Objects
        encoders = {}
        imputers = {}
        
        for col in feature_cols:
            if precs[col]['type'] == 'encoder':
                encoder = LabelEncoder()
                # Replace NaN with placeholder for fitting
                temp_train = X_train_raw[col].fillna('_MISSING_').astype(str)
                encoder.fit(temp_train.values)
                encoders[col] = encoder
            else:
                imputer = SimpleImputer(strategy='median')
                imputers[col] = imputer
                # Fit on training data frame for consistency
                imputers[col].fit(X_train_raw[[col]])
        
        # Transform Training Data
        X_train_processed = X_train_raw.copy()
        for col in feature_cols:
            if col in encoders:
                # String Processing
                val = X_train_processed[col].fillna('_MISSING_').astype(str)
                X_train_processed[col] = encoders[col].transform(val)
            else:
                # Numeric Processing
                # SimpleImputer returns DataFrame, extract column
                X_train_processed[col] = imputers[col].transform(X_train_raw[[col]]).flatten()
        
        # Encode Target Variable
        y_train_str = y_train.fillna('__UNKNOWN__').astype(str)
        target_encoder = LabelEncoder()
        target_encoder.fit(y_train_str.values)
        y_train_labels = target_encoder.transform(y_train_str)
        
        # Train Model
        X_np_train = X_train_processed.values
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_np_train, y_train_labels)
        
        # Transform Test Data
        X_test_processed = X_test_raw.copy()
        for col in feature_cols:
            if col in encoders:
                val = X_test_processed[col].fillna('_MISSING_').astype(str)
                X_test_processed[col] = encoders[col].transform(val)
            else:
                # Ensure imputer is applied to test data using parameters learned on train
                X_test_processed[col] = imputers[col].transform(X_test_raw[[col]]).flatten()
        
        # Make Predictions
        preds_int = lgb_model.predict(X_test_processed.values)
        preds_str = target_encoder.inverse_transform(preds_int)
        
        # Prepare Output
        results = pd.DataFrame({
            'id': range(len(df_test)),
            'Decision type': preds_str
        })
        
        # Save to CSV
        results.to_csv("predictions.csv", index=False)
        
        print(f"Predictions saved to predictions.csv")
        
    except Exception as e:
        # Catch exception for debugging purposes (should not happen in clean env)
        print(f"Error: {e}")
        raise

if __name__ == '__main__':
    main()