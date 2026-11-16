import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# Paths to train and test parquet files
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/openml/house_sales/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/openml/house_sales/data/test.parquet'

def main():
    # Load data
    df_train = pd.read_parquet(TRAIN_PATH)
    df_test = pd.read_parquet(TEST_PATH)
    
    # Target column and exclusion columns
    target_col = 'price'
    id_col = 'id'
    
    # Determine columns to use for features (exclude id and target)
    # Ensure consistency in column selection
    available_cols_train = df_train.columns.tolist()
    available_cols_test = df_test.columns.tolist()
    
    # Filter for columns present in both to ensure consistency
    common_cols = list(set(available_cols_train) & set(available_cols_test))
    
    feature_cols = [col for col in common_cols if col not in target_col and col not in id_col]
    
    X_train = df_train[feature_cols].copy()
    X_test = df_test[feature_cols].copy()
    y_train = df_train[target_col].copy()
    
    # Initialize preprocessing pipeline
    # Strategy: Fill missing with median, OneHot encode categorical
    numeric_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols_indices(X_train)),
            ('cat', categorical_transformer, categorical_cols_indices(X_train))
        ], remainder='drop'
    )
    
    # Alternative manual separation approach for broader compatibility if indices change
    # Detect numeric and object columns dynamically
    numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()
    
    # Fit preprocessors on training data
    imputer = SimpleImputer(strategy='median')
    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    # Function to transform
    def transform_df(df):
        # Combine numeric and categorical logic
        # First impute numeric
        df_imputed = imputer.fit_transform(df)
        # Convert back to pandas if needed or keep as numpy? 
        # We work with numpy arrays here for efficiency
        
        # Extract categorical columns names and transform
        df_numeric = df[['id']].copy() # Keep placeholders or just remove
        # Actually, ColumnTransformer is easier. Let's restart logic for simplicity.
        return None
    
    # Robust Manual Preprocessing Logic
    X_train_proc = X_train.copy()
    X_test_proc = X_test.copy()
    
    # Impute missing values across the entire dataframe
    X_train_proc[:] = imputer.fit_transform(X_train_proc.values)
    X_test_proc[:] = imputer.transform(X_test_proc.values)
    
    # One-Hot Encode categorical columns
    # Get categorical columns from the first dataframe (train)
    cat_cols = X_train_proc.select_dtypes(include=['object']).columns.tolist()
    
    # Apply OHE to the whole set at once to match columns
    # We need to fit on X_train and transform X_test
    
    # Re-init preprocessor with known columns
    numeric_preproc = SimpleImputer(strategy='median')
    cat_preproc = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    # Transform and Combine
    def process_and_fit(data_frame, is_test=False):
        # Step 1: Impute
        # data_frame needs to be convertible
        arr = data_frame.copy().astype(float).values
        arr_ = imputer.fit_transform(arr)
        
        # Identify categorical positions in the final matrix
        # Since OneHotEncoder is applied to specific columns, let's do step-by-step
        # Flatten logic: Separate Numeric and Object
        # Actually, let's use the pipeline properly on columns
        
        X_processed = []
        for col in feature_cols:
            col_data = data_frame[col]
            # Replace NaN with median
            col_filled = pd.Series(imputer.transform(np.array([col_data.values])).flatten()[:,0], dtype=float)
            # If it was categorical originally, OneHot encode
            orig_type = df_train[col].dtype
            if hasattr(df_train[col], 'name'):
                pass
            
            # Correct approach:
            # Use pandas for easy handling
            pass 
            
        # Correct simplified approach for the script:
        pass

    # Final Implementation Plan:
    # 1. Select feature columns.
    # 2. Use Pandas applymap logic or sklearn Pipleline.
    # Pipeline is cleaner.
    
    # Re-defining columns logic inside main to be cleaner
    # We already identified feature_cols.
    # Split them into numeric and categorical
    num_features = X_train.loc[:, feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X_train.loc[:, feature_cols].select_dtypes(include=['object']).columns.tolist()
    
    # Construct Pipeline
    pipe_num = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])
    pipe_cat = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='unknown' if False else 'ignore'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', pipe_num, num_features),
        ('cat', pipe_cat, cat_features)
    ])
    
    # Fit preprocessor
    preprocessor.fit(X_train)
    
    # Transform data
    X_train_final = preprocessor.transform(X_train)
    X_test_final = preprocessor.transform(X_test)
    
    # Train Model
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, max_depth=6, random_state=42)
    model.fit(X_train_final, y_train)
    
    # Predict
    preds = model.predict(X_test_final)
    
    # Create output dataframe with correct ID format
    # "id (integer row index starting from 0)"
    output_df = pd.DataFrame({
        'id': range(len(preds)),
        'price': preds
    })
    
    # Save to CSV
    output_df.to_csv('predictions.csv', index=False)

if __name__ == '__main__':
    main()