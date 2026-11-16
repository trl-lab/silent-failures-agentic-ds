import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/electric-retail-service-territories/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/electric-retail-service-territories/data/test.parquet"

def main():
    # Load data
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    
    target_col = 'TOTAL_MWH'
    
    # Identify common columns excluding target and 'id'
    # Exclude target and id from training features
    train_cols = [c for c in train_df.columns if c != target_col and c.lower() != 'id']
    test_cols = [c for c in test_df.columns if c != target_col and c.lower() != 'id']
    common_cols = list(set(train_cols) & set(test_cols))
    
    X_train = train_df[common_cols]
    y_train = train_df[target_col]
    X_test = test_df[common_cols]
    
    # Identify numeric and categorical columns for preprocessing
    numeric_features = [col for col in common_cols if pd.api.types.is_numeric_dtype(X_train[col])]
    categorical_features = [col for col in common_cols if pd.api.types.is_object_dtype(X_train[col])]
    
    # Setup Preprocessor
    imputer = SimpleImputer(strategy='median')
    encoder = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', imputer, numeric_features),
            ('cat', encoder, categorical_features)
        ]
    )
    
    # Fit and transform
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Train LightGBM Model
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 5,
        'random_state': 42,
        'verbose': -1
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train_processed, y_train)
    
    # Generate Predictions
    predictions = model.predict(X_test_processed)
    
    # Create Output DataFrame
    output_df = pd.DataFrame({
        'id': range(len(predictions)),
        'TOTAL_MWH': predictions
    })
    
    # Save to predictions.csv
    output_df.to_csv('predictions.csv', index=False)

if __name__ == '__main__':
    main()