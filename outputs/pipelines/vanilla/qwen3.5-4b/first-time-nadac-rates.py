import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from catboost import CatBoostRegressor

TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/first-time-nadac-rates/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/first-time-nadac-rates/data/test.parquet'

RANDOM_STATE = 42

def main():
    # Load datasets
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    
    # Define target column
    target_name = 'NADAC Rate'
    assert target_name in train_df.columns, f"Target column {target_name} not found."
    
    y = train_df[target_name]
    X_train = train_df.drop(columns=[target_name])
    X_test = test_df
    
    # Initialize Model
    # CatBoost handles missing values and categorical columns automatically
    model = CatBoostRegressor(
        iter_500=False,
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        objective='RMSE',
        loss_func='RMSE',
        verbose=False,
        random_seed=RANDOM_STATE,
        threading_count=-1
    )
    
    # Fit model
    model.fit(X_train, y)
    
    # Generate predictions
    preds = model.predict(X_test)
    
    # Create output dataframe with id starting from 0
    results = pd.DataFrame({
        'id': range(len(X_test)),
        'NADAC Rate': preds
    })
    
    # Save predictions
    results.to_csv('predictions.csv', index=False)
    print("Saved predictions.csv successfully.")

if __name__ == '__main__':
    main()