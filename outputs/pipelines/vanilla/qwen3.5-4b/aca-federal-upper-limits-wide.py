import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

def main():
    # Data paths
    train_path = '/path/to/user/workspace/ds_failures/datasets/strable/aca-federal-upper-limits-wide/data/train.parquet'
    test_path = '/path/to/user/workspace/ds_failures/datasets/strable/aca-federal-upper-limits-wide/data/test.parquet'
    
    # Load data
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    target_col = 'ACA FUL'
    
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in train data.")
    
    # Feature matrix and labels
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].values
    
    # Test features
    X_test = test_df.drop(columns=[])
    test_ids = range(len(X_test))
    
    # Model initialization and training
    model = CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.05, verbose=0, loss_function='RMSE')
    
    model.fit(X_train, y_train)
    
    # Predictions
    preds = model.predict(X_test)
    
    # Prepare output dataframe
    output_df = pd.DataFrame({
        'id': test_ids,
        'ACA FUL': preds
    })
    
    # Save predictions
    output_df.to_csv('predictions.csv', index=False)

if __name__ == '__main__':
    main()