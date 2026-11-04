import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMClassifier

def main():
    # Define data paths as specified in the requirements
    train_path = "/path/to/user/workspace/ds_failures/datasets/openml/eucalyptus/data/train.parquet"
    test_path = "/path/to/user/workspace/ds_failures/datasets/openml/eucalyptus/data/test.parquet"
    
    # Load data
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    # Target and Feature identification
    target_col = 'Utility'
    feature_columns = [c for c in train_df.columns if c != target_col]
    
    y_train = train_df[target_col].values
    X_train = train_df[feature_columns]
    X_test = test_df[feature_columns]
    
    # Identify numeric and categorical columns automatically
    numeric_columns = []
    categorical_columns = []
    
    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            numeric_columns.append(col)
        else:
            categorical_columns.append(col)
    
    # Preprocessing: Numeric columns
    numeric_imputer = SimpleImputer(strategy='median')
    num_transformer = Pipeline(steps=[
        ('imputer', numeric_imputer)
    ])
    
    # Preprocessing: Categorical columns
    # We calculate mode for missing values and fit OneHotEncoder
    cat_mode_dict = {}
    for col in categorical_columns:
        # Get modes from training data to handle missing values consistently
        # Filter out nulls to find common non-null value
        non_null_series = X_train[col].dropna()
        if len(non_null_series) > 0:
            mode_val = non_null_series.mode().iloc[0]
            cat_mode_dict[col] = mode_val
        else:
            cat_mode_dict[col] = '?'
    
    # Create a pipeline for categorical encoding with manual imputation logic simulation
    # Since sklearn SimpleImputer does not support string imputation efficiently without version checks,
    # we will use a functional approach for categorical filling followed by OneHotEncoder
    
    categorical_encoder = Pipeline(steps=[
        ('imputer_func', lambda x: x.fillna(cat_mode_dict[x.name])), # Custom fill logic
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
    ])
    
    # Construct Column Transformer combining numeric and categorical processors
    column_transformer = ColumnTransformer(transformers=[
        ('num', num_transformer, numeric_columns),
        ('cat', categorical_encoder, categorical_columns)
    ])
    
    # Initialize Model
    model = LGBMClassifier(n_estimators=1000, random_state=42, learning_rate=0.05)
    
    # Create Pipeline
    full_pipeline = make_pipeline(column_transformer, model)
    
    # Train Model
    full_pipeline.fit(X_train, y_train)
    
    # Generate Predictions
    y_pred = full_pipeline.predict(X_test)
    
    # Prepare Output DataFrame
    # Requirement: 'id' integer row index starting from 0, 'Utility' prediction
    result_df = pd.DataFrame({
        'id': range(len(X_test)),
        'Utility': y_pred
    })
    
    # Save to CSV
    result_df.to_csv('predictions.csv', index=False)
    
if __name__ == '__main__':
    main()