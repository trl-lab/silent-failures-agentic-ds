import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

def preprocess_data(df):
    # Identify target column
    target_col = 'scalar_coupling_constant'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in df. Available columns: {list(df.columns)}")
    
    # Identify ID column (commonly named 'id' or 'row_id')
    id_cols = [c for c in df.columns if 'id' in c.lower()]
    
    if len(id_cols) > 1:
        # If multiple potential IDs, take the last one (usually sample index)
        id_col = id_cols[-1]
    elif len(id_cols) == 1:
        id_col = id_cols[0]
    else:
        id_col = None
    
    # Prepare features
    features = df.drop(columns=[target_col])
    if id_col is not None:
        features = features.drop(columns=[id_col])
    
    return features.reset_index(drop=True), target_col, id_col

def main():
    # Define paths
    train_path = '/path/to/user/workspace/ds_failures/datasets/mlebench/champs-scalar-coupling/data/train.parquet'
    test_path = '/path/to/user/workspace/ds_failures/datasets/mlebench/champs-scalar-coupling/data/test.parquet'
    predictions_file = 'predictions.csv'
    
    # Load Data
    try:
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
    except FileNotFoundError as e:
        raise RuntimeError(f"Failed to load data files. Error: {e}")
    
    # Preprocess Train
    X_train, y_train, train_id_col = preprocess_data(train_df)
    
    # Preprocess Test
    X_test, _, test_id_col = preprocess_data(test_df)
    
    # Identify categorical and numeric columns for imputation/encoding
    numeric_cols = X_train.select_dtypes(include=['float64', 'int64', 'uint8'])
    cat_cols = X_train.select_dtypes(include=['object'])
    
    # Handle Missing Values
    if len(numeric_cols.columns) > 0:
        num_imputer = SimpleImputer(strategy='median')
        X_train[numin_cols] = num_imputer.fit_transform(X_train[numin_cols].values)
    if len(cat_cols.columns) > 0:
        cat_encoders = {}
        for col in cat_cols.columns:
            enc = LabelEncoder()
            # Store fit result
            # Note: For multi-column categorical, we transform individually
            # But LabelEncoder converts to ints, preserving order roughly
            # This works for high-level tabular
            X_train[col] = enc.fit_transform(X_train[col])
            test_X_encoded[col] = enc.transform(X_test[col])
        
        # Combine transformed categorical data back into X_train if it was an object type
        # Rebuild X_train and X_test
        # Actually, we can just use a combined list of cols to avoid duplication
        all_feature_cols = list(numeric_cols.columns) + list(cat_cols.columns)
        
        # Reset indices to handle dropped IDs properly
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        
        # Ensure X_test has same columns as X_train
        # We rebuild X_test here based on the preprocessing done on train
        # Re-implementing clean logic for safety
        
        # Re-load logic simplified:
        # 1. Extract columns
        # 2. Fit encoders
        # 3. Transform both
        
        # Clean up columns again based on analysis
        feature_cols = list(set(X_train.columns) | set(X_test.columns)) 
        # This ensures consistency
        
        # Re-do clean transform loop for final model usage
        X_final_train = X_train.copy()
        X_final_test = X_test.copy()
        
        for col in X_train.columns:
            if pd.api.types.is_numeric_dtype(X_train[col]):
                continue # handled separately
            else:
                # Object/Unknown dtype
                if pd.api.types.is_object_dtype(X_train[col]):
                    le = LabelEncoder()
                    X_final_train[col] = le.fit_transform(X_train[col])
                    X_final_test[col] = le.transform(X_test[col])
                else:
                    X_final_train[col] = X_train[col].fillna(X_test[col].fillna(pd.na_value))
                    X_final_test[col] = X_test[col].fillna(pd.na_value)
        
        # Numeric imputation
        if len(feature_cols) > 0:
            for col in feature_cols:
                X_final_train[col] = X_final_train[col].fillna(X_final_train[col].median())
                X_final_test[col] = X_final_test[col].fillna(X_final_test[col].median())
            
            # Construct final arrays for model
            X_train_arr = X_final_train.values
            X_test_arr = X_final_test.values
            
            # Model Definition
            model = lgb.LGBMRegressor(objective='regression', 
                                     n_estimators=500, 
                                     max_depth=-1, 
                                     num_leaves=31, 
                                     learning_rate=0.05,
                                     random_state=42, 
                                     subsample=0.8, 
                                     colsample_bytree=0.8,
                     )
            
            model.fit(X_train_arr, y_train.values)
            
            preds = model.predict(X_test_arr)
            
            # Prepare output dataframe
            # Ensure 'id' column starts from 0
            n_samples = len(X_test)
            output_df = pd.DataFrame({
                'id': range(n_samples),
                'scalar_coupling_constant': preds
            })
            
            output_df.to_csv(predictions_file, index=False)
            
    except Exception as e:
        print(f"Error during execution: {e}")
        raise

if __name__ == '__main__':
    main()