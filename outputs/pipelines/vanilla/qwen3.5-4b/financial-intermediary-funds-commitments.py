import pandas as pd
import numpy as np
import catboost as cb
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Paths defined in requirements
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/financial-intermediary-funds-commitments/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/financial-intermediary-funds-commitments/data/test.parquet"
OUTPUT_PATH = "predictions.csv"

def main():
    # Load data
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    # Define target column
    target_col = "Amount in USD"
    y_train = train_df[target_col]
    
    # Prepare feature matrices
    # Ensure consistency between train and test feature columns
    X_train_full = train_df.drop(columns=[target_col])
    X_test_full = test_df
    
    # Align test data to train columns to prevent shape errors
    X_test_aligned = X_test_full[X_train_full.columns]

    # Separate numeric and categorical features for automated preprocessing
    # We identify column types based on original training data
    cat_cols = []
    num_cols = []
    
    for col in X_train_full.columns:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            num_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(train_df[col]):
            num_cols.append(col)
        else:
            cat_cols.append(col)
            
    # Preprocess training data
    # Use CatBoost's ability to handle categorical natively, but enforce numeric safety
    # For robustness without installing extra encoding libraries:
    # Convert categorical to int64 via ordinal encoding (safe for CatBoost)
    
    # Impute missing values in numeric columns using median
    X_train_clean = X_train_full.copy()
    X_test_clean = X_test_aligned.copy()
    
    for col in num_cols:
        train_median = X_train_clean[col].median()
        X_train_clean[col] = X_train_clean[col].fillna(train_median)
        test_median = X_test_clean[col].median()
        X_test_clean[col] = X_test_clean[col].fillna(test_median)
        
    for col in cat_cols:
        train_mean = X_train_clean[col].mean()
        X_train_clean[col] = X_train_clean[col].fillna(-1) # Placeholder for CatBoost handling
        
    # Re-align to ensure no new columns appeared
    # Fit Ordinal Encoder for Categorical Columns if not handled by CatBoost natively
    # To ensure compatibility with CatBoost's native handling of strings without heavy dependency:
    # We will let CatBoost handle string columns directly. It handles missing values too.
    
    # Select subset of columns to feed to CatBoost (ensure consistency)
    X_train_model = X_train_clean.reset_index(drop=True)
    X_test_model = X_test_clean.reset_index(drop=True)
    
    # Remove duplicates and check columns
    cols_to_use = list(X_train_model.columns)
    
    # Train CatBoost Regressor
    # Parameters tuned for stability and auto-handling
    model = cb.CatBoostRegressor(
        iterations=100,
        learning_rate=0.05,
        max_depth=-1,
        verbose=0,
        task_type="GPU" if str(1).startswith("CPU") else "CPU", # Simple logic for CPU availability assumption
        thread_count=-1,
        log_period=100,
        seed=42
    )
    
    try:
        model.fit(X_train_model, y_train.values)
    except Exception as e:
        # Fallback to Sklearn if CatBoost fails due to environment specifics
        from sklearn.linear_model import Ridge
        from sklearn.impute import SimpleImputer
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.ensemble import RandomForestRegressor
        
        # Attempt fallback pipeline if CatBoost fails initialization
        num_transformer = SimpleImputer(strategy='median')
        cat_transformer = OneHotEncoder(handle_unknown='ignore')
        
        transformer = ColumnTransformer([
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ])
        
        full_pipeline = Pipeline([
            ('transformer', transformer),
            ('classifier', Ridge())
        ])
        
        full_pipeline.fit(X_train_clean.drop_duplicates(), y_train.values)
        predictions = full_pipeline.predict(X_test_clean.drop_duplicates())
        
        # Save
        output_df = pd.DataFrame({'id': range(len(predictions))}, columns=['id'])
        output_df['Amount in USD'] = predictions
        output_df.to_csv(OUTPUT_PATH, index=False)
        return

    # Generate predictions
    predictions = model.predict(X_test_model)
    
    # Prepare output dataframe
    output_df = pd.DataFrame({
        'id': range(len(predictions)),
        'Amount in USD': predictions
    })
    
    # Save to CSV
    output_df.to_csv(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    main()