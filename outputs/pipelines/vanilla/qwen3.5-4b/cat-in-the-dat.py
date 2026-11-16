import pandas as pd
import numpy as np
import catboost as cb

def main():
    train_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/cat-in-the-dat/data/train.parquet"
    test_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/cat-in-the-dat/data/test.parquet"
    output_path = "predictions.csv"

    # Load training and test data
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Identify feature columns (excluding target)
    feature_cols = [col for col in train_df.columns if col != 'target']

    # Initialize CatBoost classifier (handles categorical/missing values automatically)
    model = cb.CatBoostClassifier(verbose=False, random_state=42)

    # Fit model on training data
    model.fit(train_df[feature_cols], train_df['target'])

    # Predict probabilities for test data
    predictions = model.predict_proba(test_df[feature_cols])[:, 1]

    # Create submission dataframe with required columns
    submission_df = pd.DataFrame({
        "id": np.arange(len(predictions)),
        "target": predictions
    })

    # Save predictions to CSV
    submission_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()