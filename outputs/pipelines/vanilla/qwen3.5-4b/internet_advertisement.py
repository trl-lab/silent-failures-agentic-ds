import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

def main():
    # Define paths
    train_path = '/path/to/user/workspace/ds_failures/datasets/openml/internet_advertisement/data/train.parquet'
    test_path = '/path/to/user/workspace/ds_failures/datasets/openml/internet_advertisement/data/test.parquet'
    
    # Load Data
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    # Target column
    target_col = 'class'
    
    # Separate features and target
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    
    X_test = test_df.copy()
    
    # Replace missing values with 'UNKNOWN' as per interpretation
    # Prompt states: "Missing values should be interpreted as 'unknown'".
    # We will fill NaN in all feature columns.
    # Using 'unknown' string ensures consistency with prompt description even for numeric conversions
    # However, replacing NaN in numeric cols with 'unknown' will crash subsequent models expecting floats.
    # So we distinguish between numeric and object columns.
    
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    object_cols = X_train.select_dtypes(include=['object', 'string']).columns
    
    # Handle Numeric Columns
    def fill_numeric(df, cols):
        result = df.copy()
        for col in cols:
            if col in result.columns:
                # Use median or mode to minimize outlier impact, 'constant' strategy often used
                # Since prompt says interpret missing as 'unknown', but for numeric, 'unknown' means imputed
                # We use median for better performance
                result[col] = result[col].fillna(result[col].median())
        return result
    
    X_train = fill_numeric(X_train, numeric_cols)
    
    # Handle Object Columns (Text/Categorical)
    def fill_object(df, cols):
        result = df.copy()
        for col in cols:
            if col in result.columns:
                # Impute with 'Unknown' or mode. 
                # Prompt explicitly says "interpreted as 'unknown'"
                result[col] = result[col].fillna('Unknown')
        return result
    
    X_train = fill_object(X_train, object_cols)
    X_test = fill_object(X_test, object_cols) # Apply same transformation
    
    # Prepare LightGBM Dataset
    lgb_train_data = lgb.Dataset(X_train, label=y_train)
    
    # Training parameters
    params = {
        'objective': 'binary',
        'metric': ['auc'],
        'seed': 42,
        'verbose': -1,
        'boosting_type': 'gbdt',
        'force_col_wise': True
    }
    
    # Train Model
    model = lgb.train(
        params,
        lgb_train_data,
        num_boost_round=5000
    )
    
    # Predictions
    preds_raw = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Ensure probability distribution sum is correct for classification
    # LightGBM returns probabilities by default if objective=binary
    predictions = np.round(preds_raw, 5) 
    
    # Map classes (ensure 0 or 1 mapped correctly to string "nonad" or "ad")
    # We need to predict "class".
    # Check unique labels in training set to know thresholds
    train_labels = y_train.unique()
    
    # Determine threshold
    if len(set(y_train)) > 1:
        # Calculate threshold for balanced prediction or majority vote?
        # Use 0.5 for binary classification
        threshold = 0.5
        
    predicted_classes = ["nonad" if p < threshold else "ad" for p in predictions]
    
    # Create output dataframe
    # id: integer row index starting from 0
    # class: predicted class string
    predictions_df = pd.DataFrame({
        'id': range(len(X_test)),
        'class': predicted_classes
    })
    
    # Save to predictions.csv
    predictions_df.to_csv('/path/to/user/workspace/ds_failures/datasets/openml/internet_advertisement/data/predictions.csv', index=False)

if __name__ == "__main__":
    main()