import os
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# Set warning filters to handle potential pandas/numpy warnings during loading
os.environ['PYTHONWARNINGS'] = 'ignore'
pd.options.mode.chained_assignment = None

# Define file paths
train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/house-prices-advanced-regression-techniques/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/house-prices-advanced-regression-techniques/data/test.parquet'

# Load data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Identify columns to use as features
# Exclude target 'SalePrice' and 'Id' (if present in train/test, though usually separate)
columns_to_use = [col for col in train_df.columns if col != 'SalePrice' and col != 'Id']

# Handle missing values
# Numeric columns: replace NaN with column median
num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    median_val = train_df[col].median()
    train_df[col] = train_df[col].fillna(median_val)
    test_df[col] = test_df[col].fillna(median_val)

# Categorical columns: replace NaN with mode (string objects or int objects)
# We use a helper to fill missing objects in both train and test
def impute_categorical(data):
    df = data.copy()
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in obj_cols:
        # Mode calculation considering NA
        def calc_mode(series):
            return series.mode()[0] if len(series.mode()) > 0 else float('nan')
        
        mode_val = calc_mode(df[col])
        # If mode is NaN, use first non-null value or fill with NaN (let's fill first non-nan)
        if pd.isna(mode_val):
            # Find first non-na value in test too
            pass 
        # More robust: find mode excluding NaN
        clean_series = df[col].dropna()
        if len(clean_series) > 0:
            mode_val = clean_series.mode().iloc[0]
        # Fill
        df[col] = df[col].fillna(pd.Series([mode_val] * df.shape[0]))
    
    # Apply logic to train first
    for col in obj_cols:
        mask = train_df[col].isnull()
        if mask.any():
            train_df.loc[mask, col] = train_df[col].mode().iloc[0] if len(train_df[col].mode()) > 0 else train_df.loc[train_df[col].idxmax(), col]
        # Apply same logic to test
        mask_test = test_df[col].isnull()
        if mask_test.any():
            test_df.loc[mask_test, col] = test_df[col].mode().iloc[0] if len(test_df[col].mode()) > 0 else test_df.loc[test_df[col].idxmax(), col]
    return train_df, test_df

# Better imputation strategy: Use vectorized operations or sklearn preprocessing
# Since "automatically" is required, let's rely on CatBoost to handle categorical/missing
# But we still need to populate NaN values if they exist for compatibility
# Re-implementing simple imputation logic explicitly
for col in columns_to_use:
    median_val = train_df[col].median(skipna=True)
    if pd.notna(median_val):
        train_df[col] = train_df[col].fillna(median_val)
        test_df[col] = test_df[col].fillna(median_val)

for col in columns_to_use:
    if train_df[col].dtype == 'object':
        mode_val = train_df[col].mode().iloc[0]
        train_df[col] = train_df[col].fillna(mode_val)
        test_df[col] = test_df[col].fillna(mode_val)

# Prepare training set
# We include SalePrice as target, convert features
feature_set_train = train_df[columns_to_use]
target_train = train_df['SalePrice']

# Apply log transformation to target for better stability and alignment with RMSLE metric
target_transformed = np.log(target_train + 1)
feature_train_processed = feature_set_train.astype(np.float64)

# Prepare test set for prediction
feature_set_test = test_df[columns_to_use].copy()
feature_set_test = feature_set_test.astype(np.float64)

# Initialize and train model
model = CatBoostRegressor(
    verbose=0,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3.0,
    n_estimators=5000,
    random_seed=42
)

# Fit
model.fit(feature_train_processed, target_transformed, eval_set=None)

# Predict on test set (in log space)
predictions_log = model.predict(feature_set_test)

# Convert back to original scale
predictions_original = np.exp(predictions_log - 1)

# Prepare output dataframe
output_df = pd.DataFrame()
output_df['id'] = range(len(test_df))
output_df['SalePrice'] = predictions_original

# Save predictions
output_df.to_csv('predictions.csv', index=False)