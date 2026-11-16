import pandas as pd
import numpy as np
import catboost as cb

# Define paths
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/kaggle/sberbank-russian-housing-market/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_fit/datasets/kaggle/sberbank-russian-housing-market/data/test.parquet'

def main():
    # Load data
    try:
        train_df = pd.read_parquet(TRAIN_PATH)
    except Exception:
        raise FileNotFoundError(f"Train file not found at {TRAIN_PATH}")
    
    try:
        test_df = pd.read_parquet(TEST_PATH)
    except Exception:
        raise FileNotFoundError(f"Test file not found at {TEST_PATH}")

    # Identify columns
    target_col = 'price_doc'
    id_col = 'id'
    
    # Remove duplicates if any, and ensure we have consistent dtypes
    if target_col not in train_df.columns:
        raise ValueError("Target column 'price_doc' not found in train data")
    
    # Get feature columns for training
    feature_cols = [col for col in train_df.columns if col not in [target_col]]
    
    # Preprocess Train Data
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    
    # Handle missing values
    for col in X_train.columns:
        if X_train[col].isnull().any():
            if pd.api.types.is_numeric_dtype(X_train[col]):
                X_train[col] = X_train[col].fillna(X_train[col].median())
            else:
                # For categorical/object columns
                X_train[col] = X_train[col].fillna(X_train[col].mode().iloc[0] if len(X_train[col].mode()) > 0 else '-')
    
    # Preprocess Test Data
    if id_col in test_df.columns:
        # Drop id if present to avoid confusion, we will regenerate 0..N
        test_df = test_df.drop([id_col], axis=1)
    elif 'id' not in test_df.columns:
        # Assuming test has an implicit index or we should create it
        pass
        
    X_test = test_df[feature_cols]
    
    # Handle missing values in Test
    for col in X_test.columns:
        if X_test[col].isnull().any():
            if pd.api.types.is_numeric_dtype(X_test[col]):
                X_test[col] = X_test[col].fillna(X_test[col].median())
            else:
                X_test[col] = X_test[col].fillna(X_test[col].mode().iloc[0] if len(X_test[col].mode()) > 0 else '-')

    # Ensure types are compatible for CatBoost (Object/String handled natively)
    # We ensure numeric features are float
    numeric_cols = X_test.select_dtypes(include=['float64', 'int64', 'int32', 'int16', 'uint32']).columns.tolist()
    for col in numeric_cols:
        X_test[col] = X_test[col].astype('float32')
        
    # Create target log for RMSLE optimization (avoid inf)
    y_log_train = np.log1p(y_train.values)
    
    # Initialize CatBoost Regressor
    # CatBoost handles categories in strings well natively
    model = cb.CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        verbose=False,
        eval_metric='RMSE', # Optimizing RMSE on log scale works well for RMSLE
        random_state=42,
        silent=True
    )
    
    # Fit model
    model.fit(
        X_train, 
        y_log_train,
        eval_set=None  # Skip eval_set to prevent overhead
    )
    
    # Predict on Test (Log Scale)
    y_log_pred = model.predict(X_test)
    
    # Inverse Transform to get price_doc
    y_pred = np.exp(y_log_pred - 1) # ln(a+1) -> exp(x)-1 approx ln(a). Correctly: np.exp(y_log)
    # Actually np.log1p(y) = log(y+1). So y+1 = e^log1p(y). So y = e^x - 1.
    # If I used np.log(y), then y = np.exp(y_pred).
    # I used np.log1p to avoid log(0).
    # Let's stick to log(y) assuming no zeros for simplicity and standard interpretation
    # Re-do:
    y_log_train = np.log(y_train.values + 1e-5) # Safety epsilon
    
    model.fit(
        X_train, 
        y_log_train,
        eval_set=(X_train, np.array(y_log_train)) if len(X_train) > 0 else None
    )
    
    y_log_pred = model.predict(X_test)
    y_pred_expanded = np.exp(y_log_pred) - 1e-5
    
    # Generate Output DataFrame
    # Requirement: id (integer row index starting from 0)
    ids = list(range(len(test_df)))
    
    result_df = pd.DataFrame({
        'id': ids,
        'price_doc': y_pred_expanded
    })
    
    # Sort by id just in case
    result_df = result_df.sort_values(by='id').reset_index(drop=True)
    
    # Save to predictions.csv
    output_path = './predictions.csv'
    result_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == '__main__':
    main()