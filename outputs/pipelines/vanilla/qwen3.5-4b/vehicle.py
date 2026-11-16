import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import os

# File paths
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/openml/vehicle/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/openml/vehicle/data/test.parquet'
OUTPUT_PATH = 'predictions.csv'

# Load data
try:
    df_train = pd.read_parquet(TRAIN_PATH)
    df_test = pd.read_parquet(TEST_PATH)
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Prepare variables
y_train = df_train['class']
X_train = df_train.drop(columns=['class'])
y_test = df_test['class']
X_test = df_test.drop(columns=['class'])

# Determine feature columns
feature_cols = X_train.columns.tolist()

# Initialize preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Detect column types for preprocessing
is_numeric_dtype = X_train.apply(lambda col: pd.api.types.is_numeric_dtype(col)).all().result(True)
if is_numeric_dtype:
    # Ensure no non-numeric columns remain
    # Add a dummy encoder for any object columns found during transformation
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, feature_cols)
    ], remainder='drop')
else:
    # Handle mixed types if detected
    # Assuming most are numeric based on problem description, but handling object ones
    numeric_features = []
    cat_features = []
    
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            numeric_features.append(col)
        else:
            cat_features.append(col)
    
    if cat_features:
        numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
        categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, cat_features)
        ])
    else:
        preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, feature_cols)])

# Create Model Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])

# Fit model
try:
    model_pipeline.fit(X_train, y_train)
except Exception as e:
    print(f"Model Training Error (Falling back): {e}")
    # Fallback to LightGBM if sklearn fails (handling mixed data better sometimes)
    lgb_model = lgb.LGBMClassifier()
    X_train_processed = preprocessor.fit_transform(X_train)
    lgb_model.fit(X_train_processed, y_train)
    # Replace sklearn fallback
    model_pipeline.steps[1][0] = 'classifier' # Placeholder
    model_pipeline = lgb_model

# Make predictions
try:
    predictions_sk = model_pipeline.predict(X_test)
except Exception:
    # Apply fallback prediction if needed (already done above logic roughly, ensuring variable exists)
    if hasattr(lgb_model, 'predict'):
        X_test_processed = preprocessor.transform(X_test)
        predictions_sk = lgb_model.predict(X_test_processed)
        # Update model_pipeline to store results correctly or just use predictions_sk directly
        predictions = predictions_sk
    else:
        predictions_sk = model_pipeline.predict(X_test)
    predictions = predictions_sk

# Create ID sequence
prediction_ids = list(range(len(predictions)))

# Create output dataframe
output_df = pd.DataFrame({
    'id': prediction_ids,
    'class': predictions
})

# Save to CSV
output_df.to_csv(OUTPUT_PATH, index=False)
print(f"Predictions saved to {OUTPUT_PATH}")