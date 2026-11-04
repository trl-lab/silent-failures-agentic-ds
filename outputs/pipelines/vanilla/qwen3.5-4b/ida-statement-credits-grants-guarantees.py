import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Suppress output to avoid clutter during execution
lgb_verbosity = 0
lgb.log_mode(lgb_verbosity)

# Constants
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/ida-statement-credits-grants-guarantees/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/ida-statement-credits-grants-guarantees/data/test.parquet"
TARGET_NAME = "Original Principal Amount (US$)"
OUTPUT_CSV = "predictions.csv"

def preprocess_features(df, feature_columns, label_encoders, scalers, transformer_params=None):
    """
    Apply pre-processing: Impute numeric, Encode categorical, Transform.
    Returns processed DataFrame and updates fitted transformers.
    """
    df_processed = df.copy()
    
    # Numeric columns
    num_imputers = {}
    for col in feature_columns:
        # Check if column is numeric-like or needs imputation
        if np.isnan(df[col].sum()) < 100 or pd.api.types.is_numeric_dtype(df[col]):
            col_imp = SimpleImputer(strategy='median')
            num_imputers[col] = col_imp.fit_transform(df[col].values.reshape(-1, 1)).ravel()
            df_processed[col] = num_imputers[col]
        
    # Categorical columns
    cat_encoders = {}
    obj_encoders = {}
    # Ensure we fit encoders on train data for consistency
    
    for col in feature_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            if col not in label_encoders:
                le = LabelEncoder()
                # Only encode existing non-null values for fitting
                temp_mask = ~np.isnan(df[col])
                cats_val = df.loc[temp_mask, col].dropna()
                
                # Handle potential NaNs in categorical before encoding
                if len(cats_val) > 0:
                    le.fit(cats_val)
                else:
                    le.classes_ = ['0'] # Placeholder if empty
                cat_encoders[col] = le
                
                # Map to int
                mask = ~np.isnan(df[col])
                df_processed.loc[mask, col] = le.transform(df.loc[mask, col].astype(str).tolist()).astype(float)
                df_processed.loc[~mask, col] = le.transform(['nan'].tolist())[0]
    
    return df_processed

def main():
    try:
        # Load Data
        print("Loading data...")
        train_df = pd.read_parquet(TRAIN_PATH)
        test_df = pd.read_parquet(TEST_PATH)
        
        # Target Check
        if TARGET_NAME not in train_df.columns:
            raise ValueError(f"Target column {TARGET_NAME} not found.")
            
        y_train = train_df[TARGET_NAME].values
        X_train_raw = train_df.drop(columns=[TARGET_NAME]).copy()
        y_test = None
        
        # Feature Columns Identification
        feature_cols = [col for col in X_train_raw.columns if col != TARGET_NAME]
        
        # Initial Preprocessing Setup (Fit on Train)
        # We will use a simplified transformation:
        # 1. Impute NaN numeric columns with median
        # 2. Label encode object columns
        
        # Numerical Imputation
        numeric_mask = []
        object_cols = []
        
        for col in feature_cols:
            if X_train_raw[col].isnull().any():
                numeric_mask.append(col)
            if pd.api.types.is_object_dtype(X_train_raw[col]):
                object_cols.append(col)
        
        # Prepare Encoders and Scalers
        encoders = {}
        imputers = {}
        
        # Fit Imputers
        for col in numeric_mask:
            imp = SimpleImputer(strategy='median')
            imp.fit(X_train_raw[col].values.reshape(-1, 1))
            imputers[col] = imp
            
        # Fit Encoders
        for col in object_cols:
            le = LabelEncoder()
            # Safe fit handling
            vals = X_train_raw[col].dropna()
            if len(vals) > 0:
                le.fit(vals)
            else:
                le.classes_ = [np.nan]
            encoders[col] = le
            
        # Transform Data
        def transform(df, f_cols, e_dict, i_dict):
            res = df.copy()
            for col in f_cols:
                if col in i_dict:
                    # Reshape for fit, flat for predict (sklearn compat)
                    transformed_vals = i_dict[col].transform(res[col].values.reshape(-1, 1)).ravel()
                    res[col] = transformed_vals
                elif col in e_dict:
                    # Label Encode Object
                    vals = res[col].dropna()
                    if len(vals) > 0:
                        encoded = e_dict[col].transform(vals.astype(str))
                        mask = ~res[col].isna()
                        res.loc[mask, col] = encoded
                        res.loc[res[col].isna(), col] = encoders[col].classes_[0] if len(encoders[col].classes_) > 0 else 0.0
                    else:
                        res[col] = 0.0 
            return res
        
        # Apply Train Transform (re-calculated for clarity)
        X_train_processed = transform(X_train_raw, feature_cols, encoders, imputers)
        
        # Ensure Test Transformation is consistent
        X_test_processed = transform(test_df, feature_cols, encoders, imputers)
        
        # Reset Index for ID generation
        test_processed_id = X_test_processed.index.tolist()
        
        # Train Model
        print("Training model...")
        model = LGBMRegressor(
            objective='regression',
            random_state=42,
            verbose=lgb_verbosity,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=10
        )
        model.fit(X_train_processed, y_train)
        
        # Predict
        print("Generating predictions...")
        preds = model.predict(X_test_processed)
        
        # Prepare Output CSV
        results_df = pd.DataFrame({
            'id': range(len(preds)),
            TARGET_NAME: preds
        })
        
        # Save
        results_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved predictions to {OUTPUT_CSV}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()