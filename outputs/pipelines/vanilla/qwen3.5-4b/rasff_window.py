import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

def main():
    # Define paths as specified
    train_path = '/path/to/user/workspace/ds_failures/datasets/strable/rasff_window/data/train.parquet'
    test_path = '/path/to/user/workspace/ds_failures/datasets/strable/rasff_window/data/test.parquet'

    # Load data
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Define target
    target_col = 'risk_decision'
    
    # Features are all columns except target and 'id' (to avoid leakage during training if present)
    feature_cols = [col for col in train_df.columns if col != target_col and col != 'id']
    
    # Prepare training data
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    # Prepare test data (use same features)
    # Ensure categorical and numeric columns are handled correctly by passing the whole slice
    # CatBoost handles NaNs and categoricals automatically without preprocessing
    X_test = test_df[feature_cols].copy()

    # Initialize model with settings to optimize speed and automatic handling
    model = CatBoostClassifier(
        verbose=0,
        iterations=500,
        depth=5,
        loss_function='Logloss',
        random_state=42,
        thread_count=-1,
        enable_categorical=True # Required for handling string categories if converted or object dtype
    )

    # Train model
    model.fit(X_train, y_train, eval_set=(X_test, {'logloss': lambda y_true, y_pred: -np.log(y_pred)} if y_pred.size > 0 else None))

    # Make predictions
    predictions = model.predict(X_test)

    # Prepare output dataframe
    # Requirement: columns "id" (integer row index starting from 0) and "risk_decision"
    result_df = pd.DataFrame({
        'id': range(len(predictions)),
        'risk_decision': predictions
    })

    # Save to current working directory
    result_df.to_csv('predictions.csv', index=False)

if __name__ == '__main__':
    main()