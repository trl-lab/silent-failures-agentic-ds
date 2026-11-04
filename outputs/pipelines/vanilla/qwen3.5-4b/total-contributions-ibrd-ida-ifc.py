import os
import sys
import glob

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, Imputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from lightgbm import LGBMRegressor

# Suppress warnings to keep output clean
import warnings
warnings.filterwarnings('ignore')

def main():
    # Load data
    train_path = '/path/to/user/workspace/ds_failures/datasets/strable/total-contributions-ibrd-ida-ifc/data/train.parquet'
    test_path = '/path/to/user/workspace/ds_failures/datasets/strable/total-contributions-ibrd-ida-ifc/data/test.parquet'

    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    target_column = 'Total Contribution (USD)'

    # Separate features and target
    feature_cols = [col for col in df_train.columns if col != target_column]
    y_train = df_train[target_column].values
    X_train = df_train[feature_cols]
    X_test = df_test[feature_cols]

    # Identify numeric and categorical columns
    numeric_features = []
    categorical_features = []

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            numeric_features.append(col)
        else:
            categorical_features.append(col)

    # Define transformers
    numerical_transformer = Imputer(missing_values=np.nan, strategy='median')
    
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Prepare full pipeline
    regressor = LGBMRegressor(
        objective='regression',
        metric='rmse',
        num_leaves=31,
        learning_rate=0.05,
        max_depth=-1,
        verbose=-1,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', regressor)
    ])

    # Fit model on training data
    pipeline.fit(X_train, y_train)

    # Predict on test data
    y_pred = pipeline.predict(X_test)

    # Ensure predictions are floats if necessary and handle any NaNs
    y_pred = np.nan_to_num(y_pred, nan=0.0)

    # Prepare output dataframe with 'id' starting from 0
    results_df = pd.DataFrame({
        'id': range(len(df_test)),
        'Total Contribution (USD)': y_pred
    })

    # Save to CSV in current working directory
    results_df.to_csv('predictions.csv', index=False)

if __name__ == '__main__':
    main()