import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Configuration
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/mlebench/osic-pulmonary-fibrosis-progression/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/mlebench/osic-pulmonary-fibrosis-progression/data/test.parquet"
OUTPUT_PATH = "predictions.csv"

def get_processed_columns(df, target):
    """Extract numeric and categorical columns excluding target."""
    processed_cols = []
    for col in df.columns:
        if col == target:
            continue
        
        # Identify types dynamically
        dtype = df[col].dtype
        
        if dtype in ['int64', 'int32', 'int16', 'int8', 'uint8', 'uint16', 'float64', 'float32']:
            processed_cols.append(f"{col}_num")
        elif dtype == 'object':
            processed_cols.append(f"{col}_cat")
        else:
            # Try to infer float/int, fallback to object
            processed_cols.append(f"{col}_other")
    return processed_cols

# Load Data
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

# Define Target Column
TARGET_COL = 'FVC'

# Prepare Training Data
X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]
processed_cols = get_processed_columns(X_train, TARGET_COL)

# Identify actual columns for splitting
feature_columns = [c.split('_')[0] for c in X_train.columns] 

# Preprocessing
ct = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), [col[4:] if '_' in col else col for col in feature_columns if '.' in col or col.startswith('num')]), # Simplified mapping
        ('cat', Pipeline(steps=[
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), [col[4:] if '_' in col else col for col in feature_columns if '_' in col])
    ], remainder='drop'
)

# Correction for robustness with dynamic column handling
# We will perform preprocessing manually to ensure compatibility across versions and avoid sklearn config errors in production scripts.

def transform_data(df, base_columns):
    df_copy = df.copy()
    
    # Handle Missing Values in Numerical Columns
    numerical_cols = df_copy.select_dtypes(include=['number']).columns.tolist()
    for col in numerical_cols:
        df_copy[col] = df_copy[col].fillna(df_copy[col].median())
        
    # Handle Categorical Columns
    categorical_cols = df_copy.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        dummies = pd.get_dummies(df_copy[categorical_cols], dtype=float)
        # Reset Index to align correctly later if needed, but keep column names
        df_copy[categorical_cols] = dummies
    
    return df_copy

# Apply transformation logic
# Note: In a robust production script, we would use a fitted encoder. Here, we simulate it.
# For consistency between train and test:
# 1. Identify columns in train.
# 2. Apply median fill on numerical.
# 3. One-hot encode categorical (drop rare columns for consistency if needed, but default is fine).
# 4. Ensure columns exist in both train and test sets after processing.

# Clean column names to ensure consistency
final_features = []
for col in train_df.columns:
    if col == TARGET_COL:
        continue
    
    # Rename to normalized form for feature selection
    final_features.append(col)
    
    # Check for object type (categorical)
    # In parquet, dtypes might be inferred as object for string.
    # We process them later.

# Re-process Train
X_train_clean = train_df.copy()
y_train_vals = train_df[TARGET_COL].values

X_train_clean = transform_data(X_train_clean, train_df.columns)

# Remove original columns that were used for encoding to avoid duplicates
cols_to_drop = [c for c in X_train_clean.columns if c.startswith('FVC_')] # Safety check
# Actually, transform_data added dummies. Originals become NaN or redundant.
# Best to clean:
cols_before_num = X_train_clean.select_dtypes(exclude=['number']).columns
cols_before_cat = X_train_clean.select_dtypes(include=['number']).columns

# Manually construct features to ensure alignment with test
# We will rebuild X from scratch based on logical steps.

raw_x_train = X_train
raw_x_test = test_df.drop(columns=[TARGET_COL])

# Fill Numerics
for col in raw_x_train.columns:
    if raw_x_train[col].dtype in ['int64', 'float64']:
        median_val = raw_x_train[col].median()
        raw_x_train[col].fillna(median_val, inplace=True)
        raw_x_test[col].fillna(median_val, inplace=True)

# Fill Categoricals (Map to 'UNK' to avoid index errors during prediction with get_dummies)
for col in raw_x_train.columns:
    if raw_x_train[col].dtype == 'object':
        # Encode
        cats_train = raw_x_train[col].astype(str).value_counts(dropna=True)
        cats_test = raw_x_test[col].astype(str).value_counts(dropna=True)
        
        combined_cats = sorted(set(cats_train.index) | set(cats_test.index))
        unique_count = 0
        cat_map = {v: str(unique_count) for unique_count, v in enumerate(combined_cats)}
        
        raw_x_train[col] = raw_x_train[col].map(cat_map)
        raw_x_test[col] = raw_x_test[col].map(cat_map)
        
        raw_x_train[col].fillna('-1', inplace=True)
        raw_x_test[col].fillna('-1', inplace=True)

# Expand features: Convert object to one-hot style for modeling compatibility or use as is?
# For LightGBM, object strings work natively, but handling NA properly is key.
# LightGBM handles strings internally, so we can keep them string.
# But we must handle missing values consistently.
# Since we filled NA with '-1' and mapped common ones to ints, we can use strings directly.
# To make it fully numeric for sklearn/LightGBM best practice, let's OneHotEncode.
# To minimize code size and potential errors, we will proceed with LightGBM's native string support.

# Prepare Training Matrix (LightGBM accepts pandas DataFrame)
X_lgb_train = X_train_clean.iloc[:, :].copy() # Keep original clean dataframe
X_lgb_test = X_test_clean.iloc[:, :].copy() # Same structure needed

# Re-run cleaner logic specifically for the script output requirement
# Re-load to ensure clean state
train_raw = pd.read_parquet(TRAIN_PATH)
test_raw = pd.read_parquet(TEST_PATH)

target_name = 'FVC'
features_raw = [col for col in train_raw.columns if col != target_name]
X_train_raw = train_raw[features_raw]
y_train_raw = train_raw[target_name]
X_test_raw = test_raw[features_raw]

# Function to normalize features
def normalize_features(dataframe):
    result = dataframe.copy()
    # Identify numeric and object
    num_cols = result.select_dtypes(include=np.number).columns.tolist()
    obj_cols = result.select_dtypes(include=['object']).columns.tolist()
    
    for col in num_cols:
        # Fill missing
        if result[col].isnull().any():
            median = result[col].median()
            result[col] = result[col].fillna(median)
            
    # Handle Object/Categorical (convert to strings and count)
    if obj_cols:
        # Map all object values to integers (One-Hot simulation)
        # But to save space, use Label Encoding for each object column
        for col in obj_cols:
            unique_vals = result[col].unique()
            mapping = dict(zip(unique_vals, list(range(len(unique_vals)))))
            result[col] = result[col].map(mapping).fillna(-1)
            
    return result

X_train_clean = normalize_features(X_train_raw)
X_test_clean = normalize_features(X_test_raw)

# Train Model
params = {
    'learning_rate': 0.1,
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'n_estimators': 1000,
    'early_stopping_rounds': 10,
    'seed': 42
}

train_data = lgb.Dataset(X_train_clean, label=y_train_raw.values)
model = lgb.train(params, train_data, num_boost_round=500) # Fixed rounds for simplicity
# Or use validate set if we wanted early stopping, but here we optimize for completion.

# Prediction
predictions = model.predict(X_test_clean)

# Create Submission
output_df = pd.DataFrame({
    'id': range(len(predictions)),
    'FVC': predictions
})

output_df.to_csv(OUTPUT_PATH, index=False)
print("Prediction saved to", OUTPUT_PATH)