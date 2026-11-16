import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb

def main():
    train_path = "/path/to/user/workspace/ds_failures/datasets/strable/electric-generating-plants/data/train.parquet"
    test_path = "/path/to/user/workspace/ds_failures/datasets/strable/electric-generating-plants/data/test.parquet"
    
    # Load Data
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    target_col = 'NET_GEN'
    
    # Identify Feature Columns (exclude target)
    feature_cols = [col for col in train_df.columns if col != target_col]
    
    # Identify Numerical and Categorical columns based on dtype
    numerical_transformers_list = []
    categorical_transformers_list = []
    
    for col in feature_cols:
        col_dtype = train_df[col].dtype
        # Check if numeric (int, float, bool, datetime64)
        if str(col_dtype) not in ['object', 'category']:
            # Include standard numeric types
            if isinstance(col_dtype, np.dtype):
                if col_dtype.kind in 'iufbM':
                    numerical_transformers_list.append(col)
                else:
                    # If it's not recognized as numeric type, treat as object/category
                    categorical_transformers_list.append(col)
            else:
                categorical_transformers_list.append(col)
        else:
            categorical_transformers_list.append(col)

    # Define Preprocessors
    numerical_preproc = SimpleImputer(strategy='median')
    
    cat_preproc = None
    if categorical_transformers_list:
        cat_preproc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    transformers = []
    
    if numerical_transformers_list:
        transformers.append(('num', numerical_preproc, numerical_transformers_list))
        
    if cat_preproc is not None:
        transformers.append(('cat', cat_preproc, categorical_transformers_list))
        
    preprocessor = ColumnTransformer(transformers=transformers)
    
    # Define Model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', lgb.LGBMRegressor(max_depth=10, random_state=42, learning_rate=0.05, num_leaves=31))
    ])
    
    # Prepare Training Data
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    
    # Handle missing values in target explicitly (mean) before fitting
    y_train = y_train.fillna(y_train.median())
    
    # Fit Model
    model.fit(X_train, y_train)
    
    # Prepare Test Data
    X_test = test_df[feature_cols].copy()
    
    # Handle missing values in test features (consistent with training)
    if numerical_transformers_list:
        X_test = numerical_preproc.transform(X_test).reshape(-1, len(feature_cols)) # Wrong shape fix later?
        # Easier: Fit preprocessor on test using fit_transform if possible, but here we already fitted.
        # Since we used ColumnTransformer, we call transform on the fitted preprocessor inside Pipeline
        # But we need to isolate the steps to ensure transform is called.
    
    # Better workflow: use model.predict which calls the whole pipeline automatically
    # But we need to ensure test data structure matches training data exactly
    
    # Get transformed features
    X_test_processed = test_df[feature_cols]
    
    if numerical_transformers_list:
        # Apply numeric imputer logic again or use Pipeline
        # Using Pipeline is cleanest
        pass 
    
    # Re-fitting preprocessor on new data might be wrong. 
    # Use the fitted Pipeline for predict.
    # Wait, I cannot create new pipeline instance easily if I want consistency.
    # But `model.fit` stores the preprocessor state.
    # `model.predict` uses stored state.
    
    # Let's restart variable scope logic cleanly.
    pass 

# Correct Logic Flow:
# 1. Load Data
# 2. Build Preprocessor
# 3. Build Model
# 4. Fit
# 5. Predict

# Refactored Block