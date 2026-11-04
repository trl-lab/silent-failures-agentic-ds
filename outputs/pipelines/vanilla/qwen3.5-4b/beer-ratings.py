import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings('ignore')

# Configuration
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/beer-ratings/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/beer-ratings/data/test.parquet'
OUTPUT_PATH = 'predictions.csv'

def load_and_prepare_data():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    
    # Identify target column
    target_col = 'review_overall'
    
    # Filter columns for features (exclude target)
    # We assume any non-target column is a potential feature
    feature_cols = [col for col in train_df.columns if col != target_col]
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col]
    
    X_test = test_df[feature_cols].copy()
    
    # Identify numeric and categorical columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Preprocessing: Numeric Imputation + Categorical Encoding
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
    
    # Modeling
    model = lgb.LGBMRegressor(random_state=42, verbose=-1, learning_rate=0.05, n_estimators=1000)
    
    # Combine fit steps
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Train on train data
    model.fit(X_train, y_train)
    
    # Transform test data
    X_test_processed = preprocessor.transform(X_test)
    
    # Predict
    y_pred = model.predict(X_test_processed)
    
    # Prepare Output
    # Requirement: "id" (integer row index starting from 0) and "review_overall"
    test_ids = list(range(len(y_pred)))
    result_df = pd.DataFrame({
        'id': test_ids,
        'review_overall': y_pred
    })
    
    # Ensure types are correct
    result_df['id'] = result_df['id'].astype(int)
    result_df['review_overall'] = result_df['review_overall'].round(2).astype(float) # Round for float representation if needed
    
    result_df.to_csv(OUTPUT_PATH, index=False)
    
    return result_df

if __name__ == '__main__':
    load_and_prepare_data()