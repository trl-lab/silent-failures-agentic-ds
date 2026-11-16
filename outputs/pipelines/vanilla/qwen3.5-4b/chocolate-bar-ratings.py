import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
import os

# Paths defined in the problem statement
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/chocolate-bar-ratings/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/chocolate-bar-ratings/data/test.parquet'
OUTPUT_CSV = 'predictions.csv'
TARGET_COL = 'Rating'

def main():
    # Load data
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    
    # Separate features and target
    train_features = train_df.drop(columns=[TARGET_COL])
    test_features = test_df.drop(columns=[TARGET_COL])
    
    # Get column order from training set to ensure consistency
    feature_columns = train_features.columns.tolist()
    
    # Initialize the model
    # CatBoost handles missing values and categorical columns automatically
    model = CatBoostClassifier(iterations=1000, verbose=False, silent=True)
    
    # Train the model
    model.fit(train_features, train_df[TARGET_COL], early_stopping_rounds=100)
    
    # Predict on test set
    predictions = model.predict(test_features)
    
    # Create result dataframe with ID and Rating
    result_df = pd.DataFrame({
        'id': range(len(predictions)),
        'Rating': predictions
    })
    
    # Save to CSV
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Predictions saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()