import pandas as pd
import numpy as np
import lightgbm as lgb

# File paths
train_path = "/path/to/user/workspace/ds_failures/datasets/strable/summary-of-deposit_wide/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/summary-of-deposit_wide/data/test.parquet"

# Load datasets
df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

# Identify target
target_col = 'DEPSUM'

# Identify feature columns (excluding target)
feature_cols = [col for col in df_train.columns if col != target_col]

# Determine column types and fill statistics from training data
numeric_cols = []
categorical_cols = []
fill_stats = {}  # Stores fill values per column key (column name)

for col in feature_cols:
    dtype_str = str(df_train[col].dtype)
    if 'object' in dtype_str or 'string' in dtype_str or 'unicode' in dtype_str:
        categorical_cols.append(col)
        # Find most frequent non-null value for categorical
        valid_series = df_train[col].dropna()
        if len(valid_series) > 0:
            fill_stats[col] = valid_series.mode()[0]
        else:
            fill_stats[col] = "UNKNOWN"
    else:
        numeric_cols.append(col)
        # Find median for numeric
        median_val = df_train[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        fill_stats[col] = median_val

# Prepare Training Data with imputation
df_train_processed = df_train[feature_cols].copy()
for col in numeric_cols:
    df_train_processed[col].fillna(fill_stats[col], inplace=True)
for col in categorical_cols:
    df_train_processed[col].fillna(fill_stats[col], inplace=True)

# Prepare Test Data with imputation (re-using train statistics)
df_test_processed = df_test[feature_cols].copy()
for col in numeric_cols:
    df_test_processed[col].fillna(fill_stats[col], inplace=True)
for col in categorical_cols:
    df_test_processed[col].fillna(fill_stats[col], inplace=True)

# Extract Target for Training
y_train = df_train[target_col].values

# Prepare LightGBM Dataset
# LGBM handles string objects in categorical_feature efficiently
train_set = lgb.Dataset(
    df_train_processed,
    label=y_train,
    feature_names=feature_cols,
    categorical_feature=categorical_cols
)

# Parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbose': -1,
    'learning_rate': 0.05,
    'max_depth': 10,
    'num_leaves': 31
}

# Train Model
model = lgb.train(
    params,
    train_set,
    num_boost_round=500
)

# Make Predictions on Test Data
predictions = model.predict(
    df_test_processed,
    init_model=model
)

# Create Output DataFrame
results_df = pd.DataFrame({
    'id': range(len(df_test)),
    'DEPSUM': predictions
})

# Save Predictions
results_df.to_csv("/path/to/user/workspace/ds_failures/datasets/strable/summary-of-deposit_wide/data/predictions.csv", index=False)