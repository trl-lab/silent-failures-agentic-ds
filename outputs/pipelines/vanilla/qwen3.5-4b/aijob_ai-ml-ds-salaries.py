import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from scipy.sparse import hstack
import lightgbm as lgb

# File paths
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/aijob_ai-ml-ds-salaries/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/aijob_ai-ml-ds-salaries/data/test.parquet'

def main():
    # Load Data
    df_train = pd.read_parquet(TRAIN_PATH)
    df_test = pd.read_parquet(TEST_PATH)
    
    target_column = 'salary_in_usd'
    
    # Ensure target is not in features
    if target_column in df_train.columns:
        df_train.drop(columns=[target_column], inplace=True)
    if target_column in df_test.columns:
        df_test.drop(columns=[target_column], inplace=True)
        
    # Identify numeric and categorical columns
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_train.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Preprocessing Pipeline
    # 1. Numeric features: Remove missing values with mean
    imputer_num = SimpleImputer(strategy='mean')
    transformer_num = Pipeline(steps=[
        ('imputer', imputer_num)
    ])
    
    # 2. Categorical features: One Hot Encoding
    imputer_cat = SimpleImputer(strategy='constant', fill_value=0) # Fallback for rare cats or missing
    encoder_cat = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    transformer_cat = Pipeline(steps=[
        ('imputer', imputer_cat),
        ('encoder', encoder_cat)
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', transformer_num, numeric_cols),
        ('cat', transformer_cat, categorical_cols)
    ], remainder='drop')
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', lgb.LGBMRegressor())
    ])
    
    # Fit model
    model.fit(df_train, df_train[target_column])
    
    # Transform test set and predict
    X_test_transformed = model.named_preprocessor.transform(df_test)
    
    # LGBM expects dense arrays or handles sparse depending on version, 
    # we converted to dense with sparse_output=False above, but let's ensure dense
    # LGBM Regressor prefers dense if created with SparseMatrixWarning suppression or dense inputs
    if hasattr(X_test_transformed, 'toarray'):
        X_pred = X_test_transformed.toarray()
    else:
        X_pred = X_test_transformed
        
    # Some older LightGBM versions require dense for reg, newer support sparse. 
    # Assuming dense is safer for standard usage here.
    
    y_preds_raw = model.named_classifier.predict(X_pred)
    y_preds = np.clip(y_preds_raw, 0, None) # Ensure no negative if applicable
    
    # Prepare Output DataFrame
    result_df = pd.DataFrame({
        'id': range(len(df_test)),
        'salary_in_usd': y_preds
    })
    
    # Save to current working directory
    result_df.to_csv('predictions.csv', index=False)

if __name__ == '__main__':
    main()