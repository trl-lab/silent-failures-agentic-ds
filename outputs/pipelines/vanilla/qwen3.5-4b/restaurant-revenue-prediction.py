import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack

def prepare_data(df_train, df_test):
    df_train = df_train.copy()
    df_test = df_test.copy()
    
    target = 'revenue'
    feature_cols = [col for col in df_train.columns if col != target]
    
    # Identify column types
    num_cols = []
    cat_cols = []
    for col in feature_cols:
        if df_train[col].dtype.name in ['int64', 'float64', 'Float64', 'Int64']:
            num_cols.append(col)
        else:
            cat_cols.append(col)
    
    # Prepare transformers
    num_imputer = SimpleImputer(strategy='median')
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    ct = ColumnTransformer(transformers=[
        ('num_imputer', num_imputer, num_cols),
        ('ohe', ohe, cat_cols)
    ])
    
    # Fit preprocessors
    ct.fit(df_train[feature_cols])
    
    return ct, num_cols, cat_cols

def predict(model, transformed):
    return model.predict(transformed)

def main():
    train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/restaurant-revenue-prediction/data/train.parquet'
    test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/restaurant-revenue-prediction/data/test.parquet'
    
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    
    ct, num_cols, cat_cols = prepare_data(df_train, df_test)
    
    # Train model
    X_train = ct.transform(df_train[feature_cols])
    y_train = df_train['revenue']
    
    model = lgb.LGBMRegressor(
        objective='regression',
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        verbosity=-1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Prepare test features
    X_test_full = ct.transform(df_test[feature_cols])
    
    # Generate predictions
    preds = model.predict(X_test_full)
    
    # Create output dataframe with sequential IDs starting from 0
    # Ensure column name matches exactly
    result_df = pd.DataFrame({
        'id': np.arange(len(df_test)),
        'revenue': preds
    })
    
    result_df.to_csv('predictions.csv', index=False)

if __name__ == '__main__':
    main()