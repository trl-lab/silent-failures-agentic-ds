import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/college-deposit-product-marketing/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/college-deposit-product-marketing/data/test.parquet'
TARGET_NAME = 'TOTAL_ACTIVE_ACCOUNTS_'

def main():
    # Load data from parquet files
    train_data = pd.read_parquet(TRAIN_PATH)
    test_data = pd.read_parquet(TEST_PATH)

    # Extract features (excluding target column)
    # We use errors='ignore' to safely handle cases where target column might be missing in test set
    X_train = train_data.drop(columns=[TARGET_NAME], errors='ignore')
    y_train = train_data[TARGET_NAME]
    X_test = test_data.drop(columns=[TARGET_NAME], errors='ignore')

    # Initialize CatBoostRegressor
    # Handles categorical and missing values automatically
    model = CatBoostRegressor(
        objective='RMSE',
        learning_rate=0.05,
        iterations=1000,
        depth=6,
        verbose=0,
        random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Create output dataframe
    # Column 'id' must be integer row index starting from 0
    output_df = pd.DataFrame({
        'id': range(len(predictions)),
        TARGET_NAME: predictions
    })

    # Save to predictions.csv
    output_df.to_csv('predictions.csv', index=False)

if __name__ == '__main__':
    main()