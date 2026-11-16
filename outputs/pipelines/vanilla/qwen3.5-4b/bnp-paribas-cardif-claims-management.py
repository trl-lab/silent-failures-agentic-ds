import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss

# Define file paths as specified
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/bnp-paribas-cardif-claims-management/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/bnp-paribas-cardif-claims-management/data/test.parquet"
OUTPUT_PATH = "predictions.csv"

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def main():
    # Load Data
    print("Loading data...")
    df_train = pd.read_parquet(TRAIN_PATH)
    df_test = pd.read_parquet(TEST_PATH)
    
    # Identify columns
    target_col = 'target'
    id_col_in_train = 'id'
    
    # Select features and target
    X_train_full = df_train.drop(columns=[target_col], error_ignore=False)
    y_train = df_train[target_col]
    X_test_full = df_test.copy()
    
    # Remove 'id' if present in train data to prevent leakage or confusion
    # Assume 'id' might exist in training data or just use index
    if id_col_in_train in X_train_full.columns:
        X_train_full = X_train_full.drop(columns=[id_col_in_train])
    if id_col_in_train in X_test_full.columns:
        X_test_full = X_test_full.drop(columns=[id_col_in_train])
        
    # Define feature columns (exclude target, exclude any metadata like 'id' from features)
    feature_cols = [col for col in X_train_full.columns if col != target_col]
    
    # Handle Missing Values Automatically
    # Numeric columns: fill with median
    # String/Other: Keep as is (CatBoost handles them as categorical)
    
    for col in X_train_full.columns:
        if X_train_full[col].isnull().sum() > 0:
            fill_method = np.median if np.isnan(np.median(X_train_full[col].values.flat)) else lambda x: x.mean()
            X_train_full[col] = X_train_full[col].fillna(fill_method(X_train_full[col]))
            
    # Copy test data with same treatment if possible
    # Assuming test has same structure
    for col in X_test_full.columns:
        if X_test_full[col].isnull().sum() > 0:
            val = np.nanmedian(X_train_full[col].values) if np.isnan(val) else X_train_full[col].mean(skipna=True)
            try:
                if col in X_train_full.columns:
                    X_test_full[col] = X_test_full[col].fillna(X_train_full[col].median(skipna=True))
            except:
                pass
    
    # Initialize and Train Model
    # Using CatBoost for robust handling of both numerical and categorical data automatically
    params = {
        'task_type': 'GPU' if os.getenv('CATBOOST_GPU') == 'true' else 'CPU', 
        'learning_rate': 0.1,
        'depth': 6,
        'l2_leaf_reg': 3,
        'od_stat_rate': 0.8,
        'iterations': 1000,
        'early_stopping_rounds': 100,
        'verbose': False,
        'logging_level': 'Silent',
        'random_seed': RANDOM_SEED
    }
    
    try:
        model = CatBoostClassifier(
            iterations=1000,
            depth=6,
            learning_rate=0.1,
            loss_function='MultiClass' if y_train.dtype.name == 'object' and y_train.values.max() > 1 else 'Logloss',
            eval_metric='Logloss',
            **{k: v for k, v in params.items() if k not in ['logging_level', 'task_type']}
        )
    except Exception as e:
        # Fallback to default configuration if MultiClass fails unexpectedly
        print(f"Error initializing CatBoost with specific settings. Attempting fallback.")
        model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.1,
            loss_function='Logloss',
            verbose=False,
            logging_level='Silent',
            random_seed=RANDOM_SEED
        )

    # Fit
    print("Training model...")
    model.fit(X_train_full, y_train)
    
    # Predict on Test
    print("Generating predictions...")
    proba = model.predict_proba(X_test_full)[:, 1]
    
    # Create Submission DataFrame
    predictions_df = pd.DataFrame({
        'id': range(len(df_test)),
        'target': proba
    })
    
    # Ensure column order
    predictions_df = predictions_df[['id', 'target']]
    
    # Save to CSV
    predictions_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Predictions saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()