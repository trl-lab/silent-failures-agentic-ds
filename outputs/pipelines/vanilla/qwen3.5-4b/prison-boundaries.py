import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

def main():
    # Define file paths
    train_path = '/path/to/user/workspace/ds_failures/datasets/strable/prison-boundaries/data/train.parquet'
    test_path = '/path/to/user/workspace/ds_failures/datasets/strable/prison-boundaries/data/test.parquet'
    
    # Load datasets
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    # Identify target and features
    target_col = 'CAPACITY'
    # Get intersection of columns between train and test
    # Ensure we keep only columns present in test
    all_features = [col for col in train_df.columns if col != target_col]
    valid_features = [col for col in all_features if col in test_df.columns]
    
    train_df = train_df[valid_features + [target_col]].copy()
    test_df = test_df[valid_features].copy()
    
    # Split features into numeric and categorical
    numeric_cols = []
    categorical_cols = []
    
    for col in valid_features:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    
    # Create preprocessing transformers
    # Numeric
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])
    
    # Combine into ColumnTransformer
    # Note: We map transformers to columns
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
    
    # Define model
    # LightGBM regressor
    model = lgb.LGBMRegressor(objective='regression', boosting_type='gbdt', verbose=-1, 
                               learning_rate=0.1, num_leaves=31, n_estimators=500, 
                               random_state=42)
    
    # Pipeline setup
    # Wrap preprocessor and model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Fit model on training data
    model_pipeline.fit(train_df, train_df[target_col])
    
    # Predict on test data
    # Ensure test dataframe is fed with same columns
    # Preprocess test and predict together
    # We reconstruct the input for the pipeline
    test_input = test_df[[numeric_cols + categorical_cols]].copy()
    
    # Apply preprocessor logic manually to get prediction if needed, but Pipeline.fit_transform is easier
    # Actually, Pipeline.predict is best
    predictions_raw = model_pipeline.predict(test_input)
    
    # Prepare output dataframe
    # ID column must be integer row index starting from 0
    n_predictions = len(predictions_raw)
    output_df = pd.DataFrame({
        'id': range(n_predictions),
        'CAPACITY': predictions_raw
    })
    
    # Save predictions
    output_path = './predictions.csv'
    output_df.to_csv(output_path, index=False)
    
    print(f"Predictions saved to {output_path}")

if __name__ == '__main__':
    main()