import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

def main():
    # Define file paths
    train_path = '/path/to/user/workspace/ds_failures/datasets/strable/kickstarter-projects/data/train.parquet'
    test_path = '/path/to/user/workspace/ds_failures/datasets/strable/kickstarter-projects/data/test.parquet'
    
    # Load data
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    # Define target and exclusion columns
    target_col = 'status'
    id_col = 'id'
    
    # Identify feature columns
    exclude_cols = [target_col, id_col]
    all_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # Prepare features
    # We need to identify numeric and categorical columns
    numeric_features = []
    categorical_features = []
    
    for col in all_cols:
        dtype = train_df[col].dtype
        if dtype in ['int64', 'int32', 'float64', 'float32']:
            numeric_features.append(col)
        else:
            categorical_features.append(col)
            
    # Initialize transformers
    numeric_imputer = SimpleImputer(strategy='median')
    categorical_imputer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Create preprocessing pipeline components
    numeric_transformer = Pipeline(steps=[('imputer', numeric_imputer)])
    categorical_transformer = Pipeline(steps=[('imputer', categorical_imputer)])
    
    # Assemble preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    # Initialize classifier
    lgbm_clf = LGBMClassifier(
        objective='binary',
        random_state=42,
        n_estimators=500,
        max_depth=None,
        min_child_samples=20,
        force_col_wise=True,
        verbose=-1
    )
    
    # Combine into a pipeline for train
    clf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', lgbm_clf)])
    
    # Prepare training data
    X_train = train_df[all_cols].copy()
    y_train = train_df[target_col]
    
    # Fit model
    clf_pipeline.fit(X_train, y_train)
    
    # Prepare prediction data for test set
    X_test = test_df[all_cols].copy()
    
    # Generate predictions
    # The target is status (likely probability for AUC metric, as per metric description)
    # We extract the probability of the positive class (usually the last column for binary)
    proba = clf_pipeline.predict_proba(X_test)[:, -1]
    
    # Create DataFrame for submission
    # Columns: 'id' and 'status'
    submission = pd.DataFrame({
        'id': range(len(X_test)),
        'status': proba
    })
    
    # Save to current working directory
    submission.to_csv('predictions.csv', index=False)

if __name__ == "__main__":
    main()