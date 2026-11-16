import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Set paths as per instructions
train_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/mercedes-benz-greener-manufacturing/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/mercedes-benz-greener-manufacturing/data/test.parquet"

def main():
    # Load Data
    try:
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        raise e

    # Identify target column
    target_column = 'y'
    
    # Separate Features and Target
    feature_columns = [col for col in train_df.columns if col != target_column]
    X_train = train_df[feature_columns].copy()
    y_train = train_df[target_column].copy()
    X_test = test_df[feature_columns].copy()

    # Preprocessing Setup
    # Handle missing values and categoricals automatically
    numeric_features = []
    categorical_features = []
    
    # Heuristic identification
    if hasattr(X_train, 'dtype'):
        for col in feature_columns:
            dtype = X_train[col].dtype
            if dtype in ['float64', 'int64', 'int32', 'float32']:
                numeric_features.append(col)
            else:
                # Assumes object or category
                categorical_features.append(col)
    
    if not numeric_features:
        numeric_features = []
    if not categorical_features:
        categorical_features = feature_columns
        
    # Define Imputer and OneHotEncoder
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='NaN')),
        ('onehot', lgb.LGBMFallbackEncoder()) # FallbackEncoder maps object to numeric representation
        ]
    )
    
    # Correct Sklearn approach for categorical handling to ensure compatibility
    # Use LGBM's native handling if possible or standard OHE.
    # Given constraints, we will build a robust pipeline.
    
    transformer = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder=None # Drop remainder
    )
    
    # Prepare Models
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        max_depth=-1,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    
    full_pipeline = Pipeline(steps=[
        ('preprocessor', transformer),
        ('regressor', model)
    ])

    # Fit Model
    full_pipeline.fit(X_train, y_train)

    # Predict
    predictions = full_pipeline.predict(X_test)
    
    # Ensure predictions are numpy array
    if isinstance(predictions, pd.Series):
        predictions = predictions.values
    
    # Create Submission File
    # Columns: id (int index starting 0), y (prediction)
    submission_df = pd.DataFrame({
        'id': range(len(test_df)),
        'y': predictions
    })
    
    submission_df.to_csv('predictions.csv', index=False)
    print("Submission saved successfully to predictions.csv")

if __name__ == '__main__':
    main()