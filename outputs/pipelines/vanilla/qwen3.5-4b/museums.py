import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from scipy.sparse import hstack as sparse_hstack
import warnings
warnings.filterwarnings('ignore')

# File Paths
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/museums/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/museums/data/test.parquet'

def main():
    # Load Data
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    
    # Define Target
    TARGET_COL = 'REVENUE15'
    
    # Feature Extraction
    feature_cols = [col for col in train_df.columns if col != TARGET_COL]
    
    # Prepare features for both sets
    X_train_raw = train_df[feature_cols].copy()
    X_test_raw = test_df[feature_cols].copy()
    
    # Identify Types
    numeric_cols = X_train_raw.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X_train_raw.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Define Transforms
    imputer = SimpleImputer(strategy='median')
    one_hot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Create ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', ('imputer', imputer, numeric_cols)),
            ('cat', ('onehot', one_hot, categorical_cols))
        ],
        remainder='passthrough'
    )
    
    # Build Pipeline
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LGBMRegressor(objective='regression', metric='rmse', verbose=-1, max_depth=5, n_estimators=500, random_state=42))
    ])
    
    # Fit Pipeline on Train
    pipe.fit(X_train_raw, train_df[TARGET_COL])
    
    # Transform Test Data
    # We fit the encoder and imputer on the transformed data to handle unknown categories safely
    # But since we fit inside the pipeline, it fits automatically.
    # Ensure sparse arrays work correctly if using sparse output
    X_test_trans = pipe.named_steps['preprocessor'].transform(X_test_raw)
    
    # Predict
    # Retrieve predictions from the pipeline
    predictions = pipe.predict(X_test_trans)
    
    # Convert to numpy array for CSV
    preds_arr = np.array(predictions)
    
    # Prepare Output DF
    # "id" (integer row index starting from 0)
    df_out = pd.DataFrame({'id': range(len(preds_arr)), 'REVENUE15': preds_arr})
    
    # Save to CSV
    output_path = 'predictions.csv'
    df_out.to_csv(output_path, index=False)
    
    print(f"Predictions saved to {output_path}")

if __name__ == '__main__':
    main()