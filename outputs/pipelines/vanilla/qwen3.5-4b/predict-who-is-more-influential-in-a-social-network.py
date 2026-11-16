import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

def main():
    train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/predict-who-is-more-influential-in-a-social-network/data/train.parquet'
    test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/predict-who-is-more-influential-in-a-social-network/data/test.parquet'

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = 'Choice'

    # Identify feature columns excluding target and identifier-like columns
    excluded_cols = set([target_col, 'id', 'user_id', 'index'])
    train_feature_cols = set(train_df.columns) - excluded_cols
    test_feature_cols = set(test_df.columns) - excluded_cols

    common_features = sorted(train_feature_cols.intersection(test_feature_cols))
    
    X_train = train_df[common_features]
    y_train = train_df[target_col]
    X_test = test_df[common_features]

    # Train model with automatic handling of missing values and categorical columns
    model = CatBoostClassifier(
        iterations=1000,
        depth=6,
        loss_function='LogLoss',
        random_seed=42,
        verbose=False
    )

    model.fit(X_train, y_train)

    # Predict probabilities for the test set
    proba = model.predict_proba(X_test)
    
    # Derive binary choice based on probability threshold
    preds = (proba[:, 1] > 0.5).astype(int)

    # Create output dataframe with required columns
    output_df = pd.DataFrame({
        'id': list(range(len(preds))),
        'Choice': preds
    })

    # Save predictions
    output_df.to_csv('predictions.csv', index=False)

if __name__ == '__main__':
    main()