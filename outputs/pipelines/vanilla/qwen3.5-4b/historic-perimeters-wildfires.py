import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor

def preprocess_data(df, target_col):
    """Separate features and target."""
    if target_col in df.columns:
        df = df.copy()
        # Select columns except target
        feature_cols = [col for col in df.columns if col != target_col]
    else:
        feature_cols = df.columns.tolist()
    return df[feature_cols], df[target_col].to_frame(), feature_cols

def run_main():
    # Paths as specified
    path_train = "/path/to/user/workspace/ds_failures/datasets/strable/historic-perimeters-wildfires/data/train.parquet"
    path_test = "/path/to/user/workspace/ds_failures/datasets/strable/historic-perimeters-wildfires/data/test.parquet"
    target_col = "gisacres"
    output_path = "predictions.csv"
    
    # Load Data
    train_df = pd.read_parquet(path_train)
    test_df = pd.read_parquet(path_test)
    
    # Reset index to start from 0 and drop any existing indices that might conflict
    train_df_reset = train_df.reset_index(drop=True)
    test_df_reset = test_df.reset_index(drop=True)
    
    # Preprocess Features
    X_train = train_df_reset.drop(columns=[target_col], errors='ignore')
    y_train = train_df_reset[target_col]
    
    X_test = test_df_reset.copy()
    
    # Separate numerical and categorical columns for preprocessing
    numeric_features = []
    categorical_features = []
    for col in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            numeric_features.append(col)
        elif pd.api.types.is_string_dtype(X_train[col]) or pd.api.types.is_object_dtype(X_train[col]):
            categorical_features.append(col)
            
    # Initialize Column Transformer
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # For strings, most_frequent acts as mode
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    # Initialize Model
    model = LGBMRegressor(
        objective='regression',
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        n_estimators=1000,
        random_state=42,
        verbose=-1
    )
    
    # Pipeline combining preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train
    pipeline.fit(X_train, y_train.values)
    
    # Predict
    X_test_processed = pipeline.named_steps['preprocessor'].transform(X_test.reset_index(drop=True))
    predictions = pipeline.named_steps['model'].predict(X_test_processed)
    
    # Create output dataframe
    id列 = range(len(predictions))
    result_df = pd.DataFrame({
        'id': id列,
        'gisacres': predictions
    })
    
    # Save
    result_df.to_csv(output_path, index=False)

if __name__ == '__main__':
    run_main()