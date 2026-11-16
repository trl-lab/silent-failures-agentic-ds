import pandas as pd
import numpy as np
import lightgbm as lgb

def main():
    # Hardcoded file paths as specified in requirements
    train_path = "/path/to/user/workspace/ds_failures/datasets/openml/california_housing/data/train.parquet"
    test_path = "/path/to/user/workspace/ds_failures/datasets/openml/california_housing/data/test.parquet"
    
    # Load datasets
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    
    # Target column and feature columns identification
    target_column = "medianHouseValue"
    feature_columns = [col for col in df_train.columns if col != target_column]
    
    # Ensure feature columns alignment between train and test
    # Assuming consistent schema, verify intersection
    feature_columns_test = [col for col in df_test.columns if col in feature_columns]
    
    # Step 1: Handle missing values
    # Iterate over feature columns in train to get median, then fill both sets
    for col in feature_columns:
        if df_train[col].isna().any():
            median_val = df_train[col].median()
            if pd.isna(median_val):
                # Fallback to mean if median fails or all missing
                median_val = df_train[col].mean()
            else:
                # Fill train and test
                df_train[col].fillna(median_val, inplace=True)
                df_test[col].fillna(median_val, inplace=True)
    
    # Step 2: Handle categorical columns automatically (String/Object types)
    # LightGBM handles string columns as categorical features directly, ensuring we keep dtypes consistent
    categorical_features = [col for col in feature_columns if df_train[col].dtype == 'object']
    
    for col in categorical_features:
        # Convert object columns to string to ensure consistency for tree models
        if col in feature_columns_test:
            df_train[col] = df_train[col].astype(str)
            df_test[col] = df_test[col].astype(str)
        else:
            # Ensure consistency if column exists only in test for some reason
            df_test[col] = df_test[col].astype(str) if col in df_test.columns else None

    # Step 3: Logarithmic Transform on Target Variable
    # Apply log transform as suggested in dataset description to stabilize regression
    # Using small constant to prevent log(0) even if zeros are theoretically excluded
    train_target_log = np.log(df_train[target_column] + 1e-6)
    test_target_log = np.log(df_test[target_column] + 1e-6)
    
    df_train_copy = df_train.copy()
    df_train_copy[target_column] = train_target_log
    df_test_copy = df_test.copy()
    df_test_copy[target_column] = test_target_log
    
    # Prepare Training and Testing Data Matrices
    X_train = df_train_copy[feature_columns].values
    y_train = train_target_log.values
    X_test = df_test_copy[feature_columns].values
    
    # Step 4: Train Model
    # Parameters optimized for tabular data performance
    model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        n_estimators=5000,
        learning_rate=0.05,
        num_leaves=31,
        verbose=-1,
        random_state=42,
        force_col_wise=True
    )
    
    model.fit(X_train, y_train)
    
    # Step 5: Generate Predictions
    # Predict in log space, then inverse transform back to original scale
    log_predictions = model.predict(X_test)
    predictions_original = np.exp(log_predictions - np.mean(np.log(df_test_copy[target_column] + 1e-6)) + 1e-6) 
    # Correction for bias: Simply exp is safer for unbiased log-prediction reconstruction, 
    # but standard approach for log-target is exp(pred). 
    # Since we added 1e-6 offset during fitting, we should ideally invert strictly.
    # Given the exponential nature, direct exp(prediction_on_log_target_with_offset) is stable.
    # However, to minimize bias introduced by the epsilon, we can subtract mean or just exp.
    # Standard practice for log-transformed targets: y_pred = exp(model.predict(X))
    
    # Refine prediction reconstruction:
    # The model trained on log(y+eps). Prediction is log(y_pred+eps).
    # We want y_pred.
    # So y_pred = exp(prediction_log_y - eps_approximation).
    # Let's keep it simple: exp(log_prediction)
    predictions_original = np.exp(log_predictions)
    
    # Create Output DataFrame
    output_df = pd.DataFrame({
        'id': range(len(X_test)),
        'medianHouseValue': predictions_original
    })
    
    # Save to CSV
    output_df.to_csv('predictions.csv', index=False)

if __name__ == "__main__":
    main()