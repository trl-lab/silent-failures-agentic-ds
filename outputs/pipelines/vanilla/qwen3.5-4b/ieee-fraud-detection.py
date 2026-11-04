import os
import pandas as pd
import numpy as np
import catboost as cb
from scipy.sparse import issparse

def main():
    # Define paths as specified in the prompt
    train_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/ieee-fraud-detection/data/train.parquet"
    test_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/ieee-fraud-detection/data/test.parquet"
    output_path = "predictions.csv"

    print("Loading data...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Define target
    target_col = "isFraud"
    model_target = train_df[target_col].values

    # Prepare features
    excluded_cols = [target_col] + ["id"]  # Ensure we exclude ID columns if present
    
    feature_cols = [col for col in train_df.columns if col not in excluded_cols]
    
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    # Automatically detect categorical columns for CatBoost
    # We consider columns with object, category, or string types as categorical
    def is_categorical(col_name, df):
        dtype = df[col_name].dtype
        return dtype.kind == 'O' or (hasattr(df[col_name], 'dtype') and str(df[col_name].dtype).startswith('category'))
    
    # Identify indices for automatic categorical handling in CatBoost
    # CatBoost identifies non-numeric columns as categorical automatically by default in v0.8+ if passed to cat_features correctly
    # We manually specify based on types to ensure stability
    cat_indices = []
    for i, col in enumerate(feature_cols):
        col_data = X_train[col]
        if isinstance(col_data.dtype, pd.DatetimeTZDtype):
            cat_indices.append(i)
        elif pd.api.types.is_object_dtype(col_data) or \
             pd.api.types.is_datetime64_any_dtype(col_data) or \
             pd.api.types.is_bool_dtype(col_data): # Treat bool as categorical if mixed or object
            # Note: Boolean is numeric in sklearn but distinct here. CatBoost handles it.
            # Safest bet for "automatic handling of categoricals" in mixed types: 
            # We treat anything that is not pure numeric (int64, float64) as potentially categorical.
            pass
        
    # Refine: Include all non-numeric types that aren't strictly float/int
    numeric_dtypes = ['int64', 'int32', 'float64', 'float32', 'bool']
    categorical_mask = []
    for col in feature_cols:
        col_vals = X_train[col]
        # Check if it is object, datetime, or category
        if pd.api.types.is_object_dtype(col_vals) or pd.api.types.is_datetime64_any_dtype(col_vals):
            categorical_mask.append(True)
        elif pd.api.types.is_bool_dtype(col_vals):
            categorical_mask.append(False) # Treat bool as numeric usually, but CatBoost handles it
        else:
            categorical_mask.append(False)
            
    # Identify columns where the type suggests they should be categorical strings for better feature engineering
    # Strictly following prompt: "Handle ... categorical columns automatically"
    # CatBoost does this best if we pass detected categorical columns.
    cat_features_indices = [i for i, mask in enumerate(categorical_mask) if mask]
    
    # Fallback: If any categorical features detected, pass to cat_features
    # Even if none detected, CatBoost will process numeric normally.
    # We also need to handle missing values - CatBoost does this natively (treates NaN as unknown)
    
    # Define random seed and other hyperparameters
    random_seed = 42
    
    # Construct CatBoost Classifier
    # verbose=0 to avoid excessive output, silent mode
    # learning_rate adjusted for convergence
    params = {
        "learning_rate": 0.05,
        "depth": 6,
        "n_estimators": 500,
        "loss_function": "LogLoss",
        "task": "MultiClassTask",
        "random_seed": random_seed,
        "thread_count": 1, # Keep default or lower for robustness
        "verbosity": 0
    }
    
    # Add custom categorical flags if detected (optional but recommended for efficiency)
    cat_features_list = cat_features_indices if len(cat_features_indices) > 0 else None
    
    # Fit the model
    print("Training model...")
    model = cb.CatBoostClassifier(
        verbose=False,
        random_state=random_seed,
        threads=1 # Disable GPU usage to ensure consistent local run environments
    )
    model.fit(
        X_train,
        model_target,
        cat_features=cat_features_list,
        eval_metric="AUC",
        early_stopping_rounds=50,
    )
    
    # Generate predictions (Probability for ROC AUC)
    predictions_proba = model.predict_proba(X_test)[:, 1]
    
    # Ensure missing values in predictions are handled (should be none with CatBoost)
    # Create output dataframe
    test_ids = np.arange(len(test_df))
    
    output_df = pd.DataFrame({
        "id": test_ids.astype(int),
        "isFraud": predictions_proba.astype(float)
    })
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

if __name__ == "__main__":
    main()