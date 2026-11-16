import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

# Paths as specified in the prompt
train_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/porto-seguro-safe-driver-prediction/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/porto-seguro-safe-driver-prediction/data/test.parquet"

def main():
    # Load Data
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Target column definition
    target_col = "target"
    feature_cols = [col for col in train_df.columns if col != target_col]
    
    # Identify feature types for preprocessing
    # We check the training data distribution to decide preprocessing strategy
    num_cols = []
    cat_cols = []
    
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            num_cols.append(col)
        elif pd.api.types.is_object_dtype(train_df[col]):
            cat_cols.append(col)
        
        # Explicitly check for boolean as they are numeric but act as categories in some contexts, 
        # though treating as numeric usually works for tree models.
        # Keeping logic simple: numeric = numeric, else = categorical string.
    
    # Build Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), num_cols),
            ('cat', 'passthrough', cat_cols) # LightGBM handles objects/categoricals internally if passed correctly, 
                                            # but for standardizing sklearn pipelines, we can treat them as such.
                                            # However, LGBM prefers numeric. 
                                            # Let's OneHotEncode categoricals to be safe across environments.
        ]
    )
    from sklearn.preprocessing import OneHotEncoder
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), num_cols),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
        ],
        remainder='drop'
    )
    
    # Fit Preprocessor on Train
    preprocessor.fit(train_df[feature_cols])
    
    # Define the Pipeline with Model
    # LightGBM is chosen for speed and efficiency with tabular data
    model = lgb.LGBMClassifier(objective='binary', eval_metric='auc', random_state=42, verbose=-1)
    
    # Create Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train the model
    # We will train on the full dataset for simplicity as it is a classification task
    # and we need to ensure consistency for the test set
    pipeline.fit(train_df, train_df[target_col])
    
    # Prepare Test Data
    test_df_processed = pipeline.named_transformers_['preprocessor'].transform(train_df[feature_cols]) # Dummy transform line check
    
    # Proper prediction preparation
    X_test = pipeline.named_transformers_['preprocessor'].transform(test_df[feature_cols])
    y_pred_proba = pipeline.named_transformers_['model'].predict_proba(X_test)[:, 1]
    
    # Create Output DataFrame
    # Requirement: "id" (integer row index starting from 0) and "target"
    submission_id = range(len(test_df))
    
    predictions_df = pd.DataFrame({
        'id': submission_id,
        'target': np.round(y_pred_proba).astype(int) # Binary probability rounded for submission
    })
    
    # Save to CSV in current working directory
    predictions_df.to_csv('predictions.csv', index=False)
    
    print("Predictions saved to predictions.csv")

if __name__ == "__main__":
    main()