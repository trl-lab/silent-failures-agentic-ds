import pandas as pd
import numpy as np
import os
import catboost

def main():
    # File paths
    train_path = "/path/to/user/workspace/ds_failures/datasets/strable/vehicles/data/train.parquet"
    test_path = "/path/to/user/workspace/ds_failures/datasets/strable/vehicles/data/test.parquet"
    output_path = "predictions.csv"

    # Load data
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Define target column
    target_col = "youSaveSpend"

    # Ensure index starts from 0 for both
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # Identify features (exclude target)
    feature_cols = [col for col in train_df.columns if col != target_col]
    
    # Prepare DataFrames for CatBoost
    train_cat = train_df[feature_cols]
    train_y = train_df[target_col]
    
    test_cat = test_df[feature_cols]
    
    # Initialize CatBoost model
    # catboost allows automatic handling of categorical and missing values
    model = catboost.CatBoostRegressor(iterations=100, verbose_level=0)
    
    # Train the model
    # catbgotst handles missing and categorical columns automatically based on types
    model.fit(train_cat, train_y)

    # Make predictions
    y_pred = model.predict(test_cat)

    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        "id": test_df.index.astype(int),
        "youSaveSpend": y_pred
    })

    # Save to CSV
    predictions_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()