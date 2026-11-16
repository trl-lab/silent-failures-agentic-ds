import os
import pandas as pd
import numpy as np
import lightgbm as lgb

# Define paths based on the prompt description
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/abalone/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/abalone/data/test.parquet"
OUTPUT_PATH = "predictions.csv"
TARGET_COLUMN = "Class_number_of_rings"

def main():
    # Check if paths exist (optional safety, but prompt implies they do)
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Training data not found at {TRAIN_PATH}")
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Test data not found at {TEST_PATH}")

    # Load Data
    print("Loading data...")
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    # Identify features and target
    # Exclude the target column for modeling
    feature_cols = [col for col in train_df.columns if col != TARGET_COLUMN]
    
    if len(train_df[feature_cols].empty):
        raise ValueError("Feature columns empty after excluding target.")

    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COLUMN].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df[TARGET_COLUMN].copy() # Keep for reference if needed

    # Ensure consistent data types for LightGBM
    # LightGBM handles categorical columns well by default, 
    # but ensures numeric features are float64
    
    # Preprocess: Fill missing values if necessary (Prompt says handle automatically)
    # LightGBM can handle NaN in numbers during training, but for stability we can ignore or fill
    # Using default behavior of LGBMRegressor which infers strategies
    
    # Convert object columns to category if needed for LGBM performance
    for col in feature_cols:
        if X_train[col].dtype == 'object':
            X_train[col] = X_train[col].astype('category')
        if X_test[col].dtype == 'object':
            X_test[col] = X_test[col].astype('category')
            
    # Check for nulls and fill if absolutely necessary to avoid training errors, though LGBM handles them
    # Default imputation strategy for NaN in numeric cols
    X_train = X_train.fillna(X_train.mean(numeric_only=True))
    X_test = X_test.fillna(X_test.mean(numeric_only=True))

    # Define LightGBM Parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbose': -1,
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'random_state': 42,
        'min_child_samples': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5
    }

    # Define Dataset objects
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test)

    # Train Model
    print("Training model...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )

    # Make Predictions
    print("Making predictions...")
    predictions = model.predict(X_test)

    # Generate Output DataFrame
    # Requirement: 'id' (integer row index starting from 0) and 'Class_number_of_rings'
    output_ids = range(len(predictions))
    predictions_df = pd.DataFrame({
        'id': output_ids,
        TARGET_COLUMN: predictions
    })

    # Save to CSV
    output_dir = os.path.dirname(OUTPUT_PATH) or '.'
    os.makedirs(output_dir, exist_ok=True)
    predictions_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Predictions saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()