import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

def main():
    # Define paths
    train_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/open-problems-single-cell-perturbations/data/train.parquet"
    test_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/open-problems-single-cell-perturbations/data/test.parquet"
    
    # Load data
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    
    # Identify features and target
    target_col = "__multi_output__"
    features = [col for col in train.columns if col != target_col]
    
    # Extract Target (Multi-output)
    # Expected format: A list or array of gene expression values per row
    y_raw = train[target_col].tolist()
    # Flatten into numpy array (n_samples, n_genes)
    # We assume y_raw is a list of lists/arrays
    if len(y_raw) > 0:
        y_matrix = np.array(y_raw, dtype=float)
    else:
        raise ValueError("Train data target column is empty")
        
    # Prepare Feature Matrix
    X_train = train[features]
    X_test = test[features]
    
    # Clean/Prepare Categories for CatBoost
    # CatBoost handles string columns automatically if we enable categorical features
    # We set string columns as categorical features. 
    # Since we imported catboost directly, we can pass feature_columns explicitly if needed
    # but enabling 'allow_lightning' is faster.
    # To ensure numerical safety, we pass original dtypes. CatBoost infers them.
    
    # Hyperparameters
    params = {
        'task_type': 'GPU' if False else 'CPU', # Assuming CPU for stability in restricted env without CUDA check
        'iterations': 1000,
        'depth': 6,
        'verbose': False,
        'random_state': 42
    }
    
    # Initialize Model
    # CatBoost automatically handles missing values and categorical columns (strings) as categorical features
    # We do not perform manual encoding to satisfy "automatically" requirement efficiently.
    model = CatBoostRegressor(**params, loss_function='RMSE')
    
    # Store original feature order to ensure proper alignment if columns shuffled
    model.fit(X_train, y_matrix, eval_set=(X_train, None), verbose=0)
    
    # Predictions
    preds = model.predict(X_test)
    
    # Format for Submission
    # Create DataFrame with id and target predictions
    n_samples = len(test)
    
    # Ensure preds are float array
    preds_float = preds.astype('float32').ravel()
    
    results = []
    for idx in range(n_samples):
        row_preds = preds[idx]
        # Convert to string representation for CSV
        # Using space-separated values is standard for multi-output in this contest context
        # to allow parsing the column back into a list of floats.
        pred_str = " ".join(map(str, row_preds))
        results.append((idx, pred_str))
    
    # Create final dataframe
    submissions_df = pd.DataFrame(results, columns=['id', '__multi_output__'])
    
    # Save to CSV
    output_path = "predictions.csv"
    submissions_df.to_csv(output_path, index=False)
    
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()