import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

def main():
    # 1. Load Data
    train_path = '/path/to/user/workspace/ds_failures/datasets/strable/device-covid19serology/data/train.parquet'
    test_path = '/path/to/user/workspace/ds_failures/datasets/strable/device-covid19serology/data/test.parquet'

    try:
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        raise
    
    # Check for required columns
    target_col = 'antibody_truth'
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in train DataFrame.")
    
    # Drop target column from features for model training
    feature_cols = [col for col in train_df.columns if col != target_col]
    test_features = test_df[feature_cols].copy()
    
    # 2. Preprocessing: Handle Missing Values and Types
    def preprocess(df, columns, target_name):
        df = df.copy()
        
        # Handle numeric NaNs
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_columns:
            if col in columns:
                # Fill missing with median to avoid skew
                df[col] = df[col].fillna(df[col].median())
        
        # Handle categorical (object/string) NaNs
        categorical_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()
        for col in categorical_columns:
            if col in columns:
                # Fill missing with mode
                mode_val = df[col].mode()[0] if len(df[col].dropna()) > 0 else 'Unknown'
                df[col] = df[col].fillna(mode_val)
        
        return df

    # Apply preprocessing to both train and test frames
    train_processed = preprocess(train_df, feature_cols, target_col)
    test_processed = preprocess(test_df, feature_cols, None)

    # Ensure consistent data types
    # Ensure no NaNs remain
    for col in feature_cols:
        assert train_processed[col].notna().all(), f"NaNs remaining in {col} train"
        assert test_processed[col].notna().all(), f"NaNs remaining in {col} test"

    X_train = train_processed[feature_cols]
    y_train = train_processed[target_col]
    X_test = test_processed[feature_cols]

    # 3. Train Model
    # LightGBM handles categorical and numeric columns well by default if no NaNs exist (with proper setup)
    # Using CatBoost/LGBM defaults for automatic handling. 
    # Note: LightGBM default params are good for this.
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss', 
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'verbose': -1,
        'n_jobs': -1
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=1000)

    # 4. Generate Predictions
    # Returning probabilities for ROC AUC metric alignment
    pred_probs = model.predict(X_test)

    # Clip probabilities to valid range just in case
    pred_probs = np.clip(pred_probs, 0, 1)

    # 5. Prepare Output CSV
    n_samples = len(pred_probs)
    results = pd.DataFrame({
        'id': range(n_samples),
        'antibody_truth': pred_probs
    })

    # 6. Save Predictions
    output_path = 'predictions.csv'
    results.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == '__main__':
    main()