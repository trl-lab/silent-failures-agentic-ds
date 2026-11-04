import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import Imputer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
# Use LightGBM or XGBoost or CatBoost as per instructions. 
# Using CatBoost for robust automatic categorical/numeric handling.
try:
    from catboost import CatBoostClassifier, Pool
except ImportError:
    # fallback if catboost fails import, though instructed to use it
    from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings("ignore")

def main():
    # Paths
    train_path = '/path/to/user/workspace/ds_failures/datasets/strable/osha-accidents/data/train.parquet'
    test_path = '/path/to/user/workspace/ds_failures/datasets/strable/osha-accidents/data/test.parquet'
    
    # Load Data
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    
    # Define Target Column
    target_column = "Task Assigned"
    y_train = df_train[target_column].values
    
    # Features columns (exclude target and potential id column)
    exclude_cols = [target_column]
    # Ensure id is excluded if it exists
    if 'id' in df_train.columns:
        exclude_cols.append('id')
    feature_columns = [col for col in df_train.columns if col not in exclude_cols]
    
    # Separate features
    X_train = df_train[feature_columns]
    X_test = df_test[feature_columns]
    
    # Determine data types for preprocessing
    categorical_columns = []
    numerical_columns = []
    
    # Identify categorical (object/dtype str) columns
    for col in feature_columns:
        dtype_info = X_train[col].dtype
        if dtype_info.kind == 'O':
            categorical_columns.append(col)
        else:
            numerical_columns.append(col)
    
    # Initialize Preprocessors
    # Numeric: FillMedian to handle missing, Standardize
    num_imputer = Imputer(strategy='median', missing_values=np.nan)
    num_scaler = None # Optional scaling, but tree models usually don't benefit much, keeping it simple
    
    num_trans = Pipeline([
        ('imputer', num_imputer)
    ])
    
    cat_enc = OneHotEncoder(handle_unknown='ignore')
    cat_trans = Pipeline([
        ('imputer', num_imputer),
        ('encoder', cat_enc)
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_trans, numerical_columns),
            ('cat', cat_trans, categorical_columns)
        ],
        sparse_threshold=0.8
    )
    
    # Initialize Model
    if hasattr(Pool, '__module__'): # Check if catboost is available in env via try block above logic
         # We assume one of them works. Let's use XGBoost for reliability or CatBoost for robustness.
         # Given strict environment, let's use XGBoost as it's very common.
         pass
    
    # Using XGBoost for speed and performance
    from xgboost import XGBClassifier
    model = XGBClassifier(use_label_encoder=False, eval_metric='auc', random_state=42, n_estimators=500, verbosity=0, max_depth=6)
    
    # Combine preprocessing and model into pipeline
    pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('clf', model)
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Predict
    preds_proba = pipeline.predict_proba(X_test)
    # Get predicted class labels
    pred_labels = model.classes_[np.argmax(pred_proba, axis=1)]
    
    # Prepare Submission DataFrame
    submission_df = pd.DataFrame({
        'id': range(len(df_test)),
        'Task Assigned': pred_labels
    })
    
    # Save
    submission_df.to_csv('predictions.csv', index=False)
    print("Predictions saved to predictions.csv")

if __name__ == "__main__":
    main()