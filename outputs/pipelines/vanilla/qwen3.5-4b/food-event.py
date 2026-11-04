import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import os

# Set random seed for reproducibility
np.random.seed(42)

def main():
    # File paths
    train_path = "/path/to/user/workspace/ds_failures/datasets/strable/food-event/data/train.parquet"
    test_path = "/path/to/user/workspace/ds_failures/datasets/strable/food-event/data/test.parquet"
    
    # Load data
    print("Loading data...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    # Define target column
    target_col = "outcomes"
    
    # Check if target exists
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in train data.")
    
    # Separate features and target
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    # Initialize Model
    # CatBoostClassifier handles missing values and categorical columns automatically by design
    model = CatBoostClassifier(iterations=1000, verbose=0, max_depth=6, learning_rate=0.05)
    
    # Train the model
    print("Training model...")
    model.fit(X_train, y_train, eval_set=(X_test, y_test))
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_test)
    
    # Prepare output dataframe
    # Columns: id (integer row index starting from 0) and outcomes
    output_df = pd.DataFrame({
        "id": np.arange(len(predictions)),
        "outcomes": predictions
    })
    
    # Save predictions
    output_path = "predictions.csv"
    print(f"Saving predictions to {output_path}...")
    output_df.to_csv(output_path, index=False)
    
    print("Done.")

if __name__ == "__main__":
    main()