import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb

def main():
    train_path = '/path/to/user/workspace/ds_failures/datasets/strable/tobacco-problem/data/train.parquet'
    test_path = '/path/to/user/workspace/ds_failures/datasets/strable/tobacco-problem/data/test.parquet'
    target_col = 'number_health_problems'
    
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    # Separate target and features
    y_train = train_df[target_col]
    X_train = train_df.drop(columns=[target_col])
    X_test = test_df.drop(columns=[target_col])
    
    # Identify column types automatically
    num_cols = X_train.select_dtypes(include=['int64', 'float64', 'uint8', 'float32']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    
    # Setup Transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ], remainder='drop')
    
    # Define Model Pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', lgb.LGBMClassifier(n_estimators=500, verbose=-1, random_state=42))
    ])
    
    # Train Model
    model.fit(X_train, y_train)
    
    # Generate Predictions
    y_pred = model.predict(X_test)
    
    # Format Output
    results = pd.DataFrame({
        'id': range(len(test_df)),
        'number_health_problems': y_pred
    })
    
    # Save to file
    results.to_csv('predictions.csv', index=False)

if __name__ == '__main__':
    main()