import pandas as pd
import numpy as np
import catboost as cb

def main():
    # Define paths as per instructions
    train_path = '/path/to/user/workspace/ds_failures/datasets/openml/steel-plates-fault/data/train.parquet'
    test_path = '/path/to/user/workspace/ds_failures/datasets/openml/steel-plates-fault/data/test.parquet'
    output_path = 'predictions.csv'
    
    # Load data from parquet files
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    # Identify columns
    target_column = 'target'
    feature_columns = [col for col in test_df.columns if col != target_column]
    
    # Extract features and label
    X_train = train_df[feature_columns].copy()
    y_train = train_df[target_column].astype(str) # Ensure consistent format
    
    # Impute missing values in training data
    # Calculate median for numerical columns, mode for categorical/object columns
    # Using a general imputation strategy that fills NA with column statistics
    medians_and_modes = {}
    for col in feature_columns:
        # Try to get median, fallback to mode if NA exists or numeric issues
        if X_train[col].dtype == 'float64' or X_train[col].dtype == 'int64':
            val = X_train[col].median()
        elif X_train[col].isna().any():
            val = X_train[col].mode()[0] if len(X_train[col].mode()) > 0 else np.nan
        else:
            val = X_train[col].iloc[0]
        medians_and_modes[col] = val
    
    # Function to fill missing values based on calculated stats
    def fill_na(series):
        return series.fillna(series.median(skipna=True))

    # Apply imputation to training set
    X_train_imputed = X_train.fillna(0)
    for col in feature_columns:
        median_val = X_train[col].median(skipna=True)
        if pd.isna(median_val):
            X_train[col] = X_train[col].fillna(X_train[col].mode()[0] if len(X_train[col].mode()) > 0 else 0)
        else:
            X_train[col] = X_train[col].fillna(median_val)

    # Apply same imputation logic to test set for consistency
    X_test = test_df[feature_columns].copy()
    for col in feature_columns:
        median_val = X_train[col].median(skipna=True)
        if pd.isna(median_val):
            X_test[col] = X_test[col].fillna(X_train[col].mode()[0] if len(X_train[col].mode()) > 0 else 0)
        else:
            X_test[col] = X_test[col].fillna(median_val)
            
    # Convert classes to integers for CatBoost if they aren't already
    y_train_unique = sorted(list(set(y_train)))
    class_map = {cls: i for i, cls in enumerate(y_train_unique)}
    y_train_int = y_train.map(class_map)
    y_train_int = y_train_int.astype(int)
    
    # Initialize and train CatBoost model
    # Parameters optimized for tabular data with multi-class classification
    model = cb.CatBoostClassifier(
        depth=6, 
        iter_num=1000, 
        learning_rate=0.05, 
        loss_function='Multi-Class_CrossEntropy', 
        n_jobs=-1,
        verbosity=0,
        random_state=42
    )
    model.fit(X_train_imputed.values, y_train_int.values)
    
    # Make predictions on test set
    y_pred_int = model.predict(X_test.values)
    
    # Inverse mapping to original class labels
    # Note: We need to map back to original labels for the output
    y_pred_orig = np.array([class_map[int(p)] for p in y_pred_int])
    y_pred_orig = list(map(str, y_pred_orig)) # Convert back to strings for safety with original classes
    
    # Prepare output dataframe
    results = []
    for i, pred in enumerate(y_pred_orig):
        results.append({'id': i, 'target': pred})
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

if __name__ == '__main__':
    main()