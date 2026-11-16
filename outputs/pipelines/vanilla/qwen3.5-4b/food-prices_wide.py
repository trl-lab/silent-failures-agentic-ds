import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Define paths
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/food-prices_wide/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/food-prices_wide/data/test.parquet'
OUTPUT_PATH = 'predictions.csv'

def preprocess(df):
    """Clean dataframe: fill missing values and encode categoricals."""
    df = df.copy()
    
    # Identify columns
    target_col = 'usdprice'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")
    
    y = df[target_col].copy()
    X_cols = [col for col in df.columns if col != target_col]
    X = df[X_cols].copy()
    
    # Handle Missing Values Numerically
    X.apply(lambda col: col.fillna(col.median() if col.notna().any() else np.nan))
    y.fillna(y.median())
    
    # Handle Categorical Columns Automatically
    categorical_encoders = {}
    numeric_features = []
    for col in X.columns:
        dtype = X[col].dtype
        if dtype == 'object' or pd.api.types.is_categorical_dtype(X[col]):
            # Treat as categorical string/category
            enc = LabelEncoder()
            X[col] = enc.fit_transform(pd.DataFrame({col: pd.Series(X[col])}).apply(lambda x: str(x) if pd.isnull(x) else x).tolist() if dtype==object else pd.Series(X[col]).astype(str).astype(str))
            # Correct approach for direct string object to int
            X[col] = enc.fit_transform(X[col].apply(lambda x: '' if pd.isna(x) else str(x)).astype(str))
        elif pd.api.types.is_numeric_dtype(X[col]) or pd.api.types.is_integer_dtype(X[col]):
            numeric_features.append(col)
        else:
            numeric_features.append(col) # Keep numeric
        
        # Proper numeric filling
        X[col].fillna(X[col].median(), inplace=True)
        
    return X, y, numeric_features

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor

# Load Data
print("Loading data...")
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

# Setup
y = train_df['usdprice'].values
X_train_full = train_df.drop('usdprice', axis=1)
X_test_full = test_df.drop('usdprice', axis=1) # Test set should not have target

# Preprocessing
preprocessed_train, y_train, _ = preprocess(X_train_full)
# Note: The helper function above was pseudocode, I will inline it properly below for execution
    
# Revised Inline Preprocessing
features_list = [col for col in X_train_full.columns if col != 'usdprice']
feature_columns = X_train_full[features_list].copy()
feature_columns_test = X_test_full[feature_columns].copy()

# Handle NA in Train
for col in feature_columns.columns:
    if feature_columns[col].isnull().sum() > 0:
        feature_columns[col].fillna(feature_columns[col].median(), inplace=True)
    if feature_columns_test[col].isnull().sum() > 0:
        feature_columns_test[col].fillna(feature_columns_test[col].median(), inplace=True)

# Handle Categoricals (Object dtype)
# We use LabelEncoder for safety as trees can't handle raw strings directly without special flags
# However, LGBM handles strings if specified. To be safe and robust for all users:
encoders = {}
for col in feature_columns.columns:
    if feature_columns[col].dtype.name == 'object':
        # Convert object to numeric codes
        encoder = LabelEncoder()
        # Fit on train to map unique strings
        encoder.fit(feature_columns[col].astype(str))
        feature_columns[col] = encoder.transform(feature_columns[col].astype(str))
        feature_columns_test[col] = encoder.transform(feature_columns_test[col].astype(str))
        encoders[col] = encoder
    else:
        continue

# Prepare Models
model = LGBMRegressor(random_state=42, n_estimators=1000, verbose=-1)

# Train
model.fit(feature_columns, y_train)

# Predict
# Reset index to start from 0 as requested
test_df_clean = test_df.reset_index(drop=True)
ids = range(len(test_df_clean))
predictions = model.predict(feature_columns_test)

# Construct Output
output_df = pd.DataFrame({'id': ids, 'usdprice': predictions})
output_df.to_csv(OUTPUT_PATH, index=False)
print(f"Predictions saved to {OUTPUT_PATH}")