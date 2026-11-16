import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import lightgbm as lgb

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Configuration
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/global-power-plant/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/global-power-plant/data/test.parquet'
OUTPUT_PATH = 'predictions.csv'

def main():
    # Load data
    df_train = pd.read_parquet(TRAIN_PATH)
    df_test = pd.read_parquet(TEST_PATH)
    
    # Separate target and features
    target_col = 'capacity_mw'
    feature_cols = [col for col in df_train.columns if col != target_col]
    
    X_train = df_train[feature_cols]
    y_train = df_train[target_col]
    X_test = df_test[feature_cols]
    
    # Identify numeric and categorical columns
    # We treat everything else as categorical for safety against mixed types
    num_cols = df_train.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Remove boolean if numeric selection caught some bools (treated as int)
    bool_indices = df_train.select_dtypes(include=['bool']).columns.tolist()
    if bool_indices:
        cat_cols.extend(bool_indices)
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ],
        remainder='passthrough'
    )
    
    # Initialize model
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42, verbosity=-1)
    
    # Full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Fit pipeline
    pipeline.fit(X_train, y_train)
    
    # Predict
    X_test_processed = pipeline.named_steps['preprocessor'].transform(X_test)
    predictions = pipeline.named_steps['model'].predict(X_test_processed)
    
    # Prepare output dataframe
    ids = np.arange(len(X_test))
    result_df = pd.DataFrame({
        'id': ids,
        'capacity_mw': predictions
    })
    
    # Save to CSV
    result_df.to_csv(OUTPUT_PATH, index=False)

if __name__ == '__main__':
    main()