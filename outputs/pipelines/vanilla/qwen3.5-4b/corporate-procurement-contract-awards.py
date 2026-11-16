import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Configuration
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/corporate-procurement-contract-awards/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/corporate-procurement-contract-awards/data/test.parquet'
OUTPUT_FILE = 'predictions.csv'
TARGET_COLUMN = "Contract Award Amount (USD)"

# Seed for reproducibility
RANDOM_STATE = 42

def main():
    # Load Data
    print("Loading data...")
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    
    # Extract Features and Target
    target_name = TARGET_COLUMN
    train_y = train_df[target_name].values
    test_df = test_df.copy()
    
    # Features to predict are all columns except the target
    feature_columns = [col for col in train_df.columns if col != target_name]
    test_feature_cols = [col for col in test_df.columns if col != target_name]
    
    # Filter out rows where feature columns don't align to test set (robustness)
    # Assuming schemas are aligned.
    
    train_features = train_df[feature_columns]
    test_features = test_df[test_feature_cols]
    
    # Determine Feature Types
    numeric_cols = train_features.select_dtypes(include=['number']).columns
    categorical_cols = train_features.select_dtypes(['object', 'category', 'bool']).columns
    
    # Setup Transformers
    numeric_imputer = SimpleImputer(strategy='median')
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[('imputer', numeric_imputer)]), numeric_cols),
            ('cat', Pipeline(steps=[('imputer', categorical_imputer), ('encoder', ohe)]), categorical_cols)
        ],
        remainder='drop'
    )
    
    # Fit Preprocessor and Train Model
    print("Training model...")
    X_train = preprocessor.fit_transform(train_features)
    
    # Model Definition
    model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        early_stopping_rounds=50,
        verbose=-1,
        random_state=RANDOM_STATE
    )
    
    model.fit(
        X_train, 
        train_y,
        eval_set=None,
        verbose=False
    )
    
    # Transform Test Data and Predict
    X_test = preprocessor.transform(test_features)
    predictions = model.predict(X_test)
    
    # Prepare Output
    results = pd.DataFrame({
        'id': range(len(test_df)),
        'Contract Award Amount (USD)': predictions.astype(float)
    })
    
    # Save
    results.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()