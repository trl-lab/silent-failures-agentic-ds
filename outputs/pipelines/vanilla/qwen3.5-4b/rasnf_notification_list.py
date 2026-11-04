import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
# Using lightgbm for efficiency and categorical handling
try:
    import catboost as cb
    CB_AVAILABLE = True
except ImportError:
    CB_AVAILABLE = False
    
# Set paths
train_path = "/path/to/user/workspace/ds_failures/datasets/strable/rasnf_notification_list/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/rasnf_notification_list/data/test.parquet"

def main():
    # Load data
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    
    target_col = "num_countries_affected"
    feature_cols = [col for col in df_train.columns if col != target_col]
    
    # Define transformers
    # Numeric imputation
    transformer_numeric = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # One-hot encoding for categorical
    transformer_categorical = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', transformer_numeric, ['_temp_numeric']), # Temp placeholders won't matter here
            ('cat', transformer_categorical, ['_temp_categorical'])
        ],
        remainder='passthrough' # In case of mixed, but we filter by cols below
    )
    
    # Better approach: Explicitly handle columns based on dtype before pipeline for clarity and speed
    # Re-evaluate features
    train_df = df_train.copy()
    test_df = df_test.copy()
    
    # Ensure feature columns are consistent (remove target if present in test train split confusion)
    # Feature columns are everything except target
    
    # Preprocessing Logic
    # Convert columns to types
    numeric_cols = []
    categorical_cols = []
    
    for col in feature_cols:
        if train_df[col].dtype in ['float64', 'int64', 'int32']:
            numeric_cols.append(col)
        elif train_df[col].dtype in ['object', 'str', 'bool']:
            categorical_cols.append(col)
            
    # Fit transformer on numeric
    numeric_imputer = SimpleImputer(strategy='median')
    numeric_scaler = StandardScaler()
    
    # Fit encoder on categorical
    categorical_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Train pipeline components (manually for consistency)
    X_train_raw = train_df[feature_cols].copy()
    X_test_raw = test_df[feature_cols].copy()
    
    X_train_num = numeric_imputer.fit_transform(X_train_raw[numeric_cols].values)
    X_train_cat = categorical_encoder.fit_transform(X_train_raw[categorical_cols].values.reshape(-1, 1))
    
    # We need to flatten the One Hot encoded array
    X_train_processed = np.hstack([X_train_num, X_train_cat.flatten()])
    
    # Map IDs
    y_train = train_df[target_col].values.astype(float)
    ids_to_test = np.arange(len(df_test)).astype(int)
    
    # Prepare test data
    X_test_num = numeric_imputer.transform(X_test_raw[numeric_cols].values)
    X_test_cat = categorical_encoder.transform(X_test_raw[categorical_cols].values.reshape(-1, 1))
    X_test_processed = np.hstack([X_test_num, X_test_cat.flatten()])
    
    # Model Selection: XGBoostClassifier for performance with categorical handling 
    # (using encoded data, XGBoost handles it better than CatBoost for encoded inputs usually, but CB is faster for raw mixed)
    # Since we encoded, we use XGBoost
    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='auc',
        n_estimators=2000,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    )
    
    # Train model
    model.fit(X_train_processed, y_train)
    
    # Predict probabilities
    # XGBClassifier.predict_proba returns probabilities of classes
    # Target is 'num_countries_affected'. We need to align classes.
    # Assuming the target allows us to predict probabilities directly.
    # If multiclass, we need to select the relevant column for AUC. 
    # To satisfy ROC AUC optimally, we typically predict the 'positive' probability.
    # However, since target is 'num...', it acts as continuous/count-like.
    # We will use the probability of the highest impact class or simply predict_raw.
    # Standard practice for AUC with this target: Predict the score directly or proba of positive class.
    # Let's use predict_proba[:, 1] if there are multiple classes, else [:,-1]
    
    if y_train.max() > 1:
        proba = model.predict_proba(X_test_processed)
        # If it is multiclass, we take the probability of the class corresponding to the largest value?
        # Or we take all classes and average?
        # For ROC AUC to work with counts, we must binarize 0 vs >0.
        # Let's perform binaurization on the target for training logic.
        # Re-fit model with binarized target
        y_train_binary = (y_train > 0).astype(float)
        
        model_binary = xgb.XGBClassifier(eval_metric='auc', use_label_encoder=False, n_estimators=1000, max_depth=6, random_state=42)
        model_binary.fit(X_train_processed, y_train_binary)
        predictions = model_binary.predict_proba(X_test_processed)[:, 1]
    else:
        # Binary target assumption
        model_binary = xgb.XGBClassifier(eval_metric='auc', use_label_encoder=False, n_estimators=1000, max_depth=6, random_state=42)
        model_binary.fit(X_train_processed, y_train)
        predictions = model_binary.predict_proba(X_test_processed)[:, 1]
        
    # Create output dataframe
    results_df = pd.DataFrame({
        'id': ids_to_test,
        'num_countries_affected': predictions
    })
    
    # Save
    output_path = "predictions.csv"
    results_df.to_csv(output_path, index=False)
    
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()