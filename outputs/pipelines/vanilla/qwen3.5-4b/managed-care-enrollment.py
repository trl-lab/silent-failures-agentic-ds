import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostRegressor

def main():
    # Define paths
    train_path = '/path/to/user/workspace/ds_failures/datasets/strable/managed-care-enrollment/data/train.parquet'
    test_path = '/path/to/user/workspace/ds_failures/datasets/strable/managed-care-enrollment/data/test.parquet'
    target_col = 'Total Enrollment'

    # Load data
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Extract features and target
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])

    # Identify column types automatically
    numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=['number']).columns.tolist()

    # Define transformers
    numeric_imputer = SimpleImputer(strategy='median')
    cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_imputer, numeric_cols),
        ('cat', cat_encoder, categorical_cols)
    ])

    # Initialize model
    model = CatBoostRegressor(
        iterations=200,
        learning_rate=0.05,
        depth=6,
        verbose=False,
        random_state=42,
        thread_count=1
    )

    # Fit preprocessor and model
    X_transformed_train = preprocessor.fit_transform(X_train)
    model.fit(X_transformed_train, y_train.values)

    # Predict
    X_transformed_test = preprocessor.transform(X_test)
    predictions = model.predict(X_transformed_test)

    # Create output dataframe
    results_df = pd.DataFrame({
        'id': range(len(X_test)),
        'Total Enrollment': predictions
    })

    # Save predictions
    results_df.to_csv('predictions.csv', index=False)

if __name__ == '__main__':
    main()