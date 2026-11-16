import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define file paths
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/gainful-employment/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/gainful-employment/data/test.parquet"
OUTPUT_CSV = "predictions.csv"

def main():
    try:
        # Load data
        train_df = pd.read_parquet(TRAIN_PATH)
        test_df = pd.read_parquet(TEST_PATH)
        
        # Target column
        target = "Debt-to-Earnings Annual Rate"
        
        # Check existence
        if target not in train_df.columns:
            raise ValueError(f"Target column '{target}' not found in train data.")
        
        # Separate features and target
        y_train = train_df[target]
        X_train = train_df.drop(columns=[target])
        X_test = test_df.drop(columns=[target])
        
        # Identify numeric and categorical columns
        num_cols = []
        cat_cols = []
        
        for col in X_train.columns:
            if pd.api.types.is_numeric_dtype(X_train[col]):
                num_cols.append(col)
            elif pd.api.types.is_object_dtype(X_train[col]):
                cat_cols.append(col)
            else:
                # Fallback for other object/string types
                cat_cols.append(col)
                
        # Initialize transformers
        numeric_transformer = SimpleImputer(strategy='median')
        cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_cols),
                ('cat', cat_encoder, cat_cols)
            ]
        )
        
        # Initialize model
        model = LGBMRegressor(
            random_state=42, 
            n_estimators=1000, 
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8
        )
        
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocess', preprocessor),
            ('model', model)
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Predict on test data
        y_pred = pipeline.predict(X_test)
        
        # Create output dataframe
        # Requirements: 'id' (integer row index starting from 0) and 'Debt-to-Earnings Annual Rate'
        results = pd.DataFrame({
            'id': np.arange(len(X_test)),
            'Debt-to-Earnings Annual Rate': y_pred.flatten()
        })
        
        # Save predictions
        results.to_csv(OUTPUT_CSV, index=False)
        print(f"Predictions saved to {OUTPUT_CSV}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()