import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor

def main():
    # Configuration
    train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/allstate-claims-severity/data/train.parquet'
    test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/allstate-claims-severity/data/test.parquet'
    target_column = 'loss'
    output_file = 'predictions.csv'

    # Load Data
    print("Loading data...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Identify Features and Columns
    feature_columns = [col for col in train_df.columns if col != target_column]
    
    if not feature_columns:
        raise ValueError("No feature columns found after removing target.")

    print(f"Features: {len(feature_columns)}")
    
    # Determine numeric and categorical columns
    num_features = []
    cat_features = []
    
    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            num_features.append(col)
        else:
            cat_features.append(col)
            
    print(f"Numeric columns: {num_features}")
    print(f"Categorical columns: {cat_features}")

    # Preprocessing
    # 1. Numeric Imputation
    # 2. Categorical Encoding (Ordinal) + Imputation for missing
    
    # Initialize transformers
    num_imputer = SimpleImputer(strategy='median')
    
    # Ordinal encoder setup
    # Handle unknown values in test set by assigning a negative score (-1)
    # This is done by setting handle_unknown='use_encoded_value'
    ohe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_values=-1)

    # Define ColumnTransformer to handle the columns differently
    # Note: In newer sklearn versions, ColumnTransformer handles mixed types well
    
    transform_pipeline = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median'))
            ]), num_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
                ('enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_values=-1))
            ]), cat_features)
        ]
    )

    print("Fitting preprocessing pipeline on training data...")
    transform_pipeline.fit(train_df[feature_columns])
    
    # Prepare Features
    X_train_transformed = transform_pipeline.transform(train_df[feature_columns])
    y_train = train_df[target_column]
    
    # Transform Test Data
    X_test_transformed = transform_pipeline.transform(test_df[feature_columns])

    # Train Model
    print("Training model...")
    model = LGBMRegressor(random_state=42, n_estimators=500, learning_rate=0.05, 
                          subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=50, verbosity=-1)
    model.fit(X_train_transformed, y_train)
    
    # Evaluate (Optional for self-check, not submitting)
    # preds_on_train = model.predict(X_train_transformed)
    # print("MAE on Training Set:", np.mean(np.abs(preds_on_train - y_train)))

    # Make Predictions
    print("Generating predictions...")
    preds = model.predict(X_test_transformed)
    
    # Create Output DataFrame
    # Requirement: Two columns: "id" (integer row index starting from 0) and "loss"
    output_df = pd.DataFrame({
        'id': range(len(test_df)),
        'loss': preds
    })

    # Save predictions
    print(f"Saving predictions to {output_file}...")
    output_df.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    main()