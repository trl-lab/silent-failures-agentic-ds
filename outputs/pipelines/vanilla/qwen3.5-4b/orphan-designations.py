import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier

# Configuration
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/orphan-designations/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/orphan-designations/data/test.parquet'
OUTPUT_PATH = 'predictions.csv'

def main():
    # Load Data
    print("Loading data...")
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    
    # Identify Target and Features
    target_col = 'Status'
    feature_cols = [col for col in train_df.columns if col != target_col]
    
    # Prepare X and y
    y_train = train_df[target_col].copy()
    # Ensure binary classification for ROC AUC
    if y_train.dtype == object:
        # Try to infer binary
        unique_vals = sorted(y_train.unique())
        # Assume 2 classes or majority=1, minority=0
        if len(unique_vals) == 2:
            # Map lowest/first alphabetically to 0 for consistency
            val0 = unique_vals[0]
            val1 = unique_vals[1]
            y_train = y_train.replace({val0: 0, val1: 1}).astype(int)
        else:
            # Fallback to binary majority/minority logic
            counts = y_train.value_counts()
            major, minor = max(counts), min(counts)
            major_val, minor_val = counts.index[max(counts)], counts.index[min(counts)]
            y_train = y_train.replace({major_val: 1, minor_val: 0}).astype(int)
    else:
        # Already numeric
        if y_train.min() <= 0 and y_train.max() >= 1:
             y_train = y_train.astype(float)
        else:
            # Binarize if 0/1 string
            if isinstance(y_train.iloc[0], str):
                y_train = pd.to_numeric(y_train, errors='coerce')
            elif y_train.min() <= 0 and y_train.max() >= 1:
                y_train = y_train.astype(int)
    
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    
    # Preprocessing
    # Separate numeric and categorical
    num_features = X_train.select_dtypes(include=['number']).columns
    cat_features = X_train.select_dtypes(include=['object', 'bool']).columns
    
    # Imputer for numeric, OneHot for categorical
    transformer = ColumnTransformer(transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
        ]), num_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), cat_features)
    ])
    
    # Fit and Transform
    print("Preprocessing and Training...")
    X_train_processed = transformer.fit_transform(X_train)
    X_test_processed = transformer.transform(X_test)
    
    # Train Model
    model = LGBMClassifier(
        objective='binary',
        random_state=42,
        verbose=-1,
        force_row_wise=False
    )
    model.fit(X_train_processed, y_train.values.ravel())
    
    # Predictions
    print("Generating predictions...")
    # Get probabilities, convert to class labels
    proba = model.predict_proba(X_test_processed)
    # Threshold at 0.5
    predicted_status = (proba[:, 1] >= 0.5).astype(int)
    
    # Create Output DataFrame
    output_df = pd.DataFrame({
        'id': range(len(X_test_processed)),
        'Status': predicted_status
    })
    
    # Save to CSV
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved predictions to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()