import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor

def main():
    # File paths
    train_file = "/path/to/user/workspace/ds_failures/datasets/strable/financial-management/data/train.parquet"
    test_file = "/path/to/user/workspace/ds_failures/datasets/strable/financial-management/data/test.parquet"
    
    # Load Data
    train_df = pd.read_parquet(train_file)
    test_df = pd.read_parquet(test_file)
    
    target_col = "State Share"
    
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data.")
    if target_col not in test_df.columns:
        # If target not in test, still assume it exists for output but can't predict target column on test itself.
        # Assuming standard setup where target is excluded from features.
        pass
    
    # Extract features and target
    X_train = train_df.copy()
    y_train = train_df[target_col]
    
    X_train_clean = X_train.drop(columns=[target_col], errors='ignore')
    X_test = test_df.copy()
    X_test_clean = X_test.drop(columns=[target_col], errors='ignore')
    
    # Identify column types
    feature_cols = list(X_train_clean.columns)
    num_features = []
    cat_features = []
    
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            num_features.append(col)
        else:
            cat_features.append(col)
    
    # Define Preprocessors
    transformers = []
    
    # Handle numeric features (impute missing values)
    for col in num_features:
        imputer = SimpleImputer(strategy='median')
        transformers.append((imputer, [col]))
    
    # Handle categorical features (one-hot encoding)
    for col in cat_features:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        transformers.append((encoder, [col]))
    
    # Configure ColumnTransformer
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    
    # Create Pipeline
    pipe = Pipeline([
        ('preprocess', preprocessor),
        ('regressor', LGBMRegressor(objective='regression', n_estimators=500, random_state=42, verbose=-1))
    ])
    
    # Fit Model
    pipe.fit(X_train_clean, y_train)
    
    # Predict on Test Data
    y_pred = pipe.predict(X_test_clean)
    
    # Prepare Output
    output_df = pd.DataFrame({
        'id': range(len(y_pred)),
        'State Share': y_pred
    })
    
    # Save to CSV
    output_df.to_csv('predictions.csv', index=False)

if __name__ == "__main__":
    main()