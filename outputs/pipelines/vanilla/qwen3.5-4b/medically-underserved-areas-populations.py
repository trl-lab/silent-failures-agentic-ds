import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.impute import SimpleImputer

# Configuration
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/medically-underserved-areas-populations/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/medically-underserved-areas-populations/data/test.parquet'
TARGET_COL = 'IMU Score'
OUTPUT_CSV = 'predictions.csv'
RANDOM_STATE = 42

# Load Data
try:
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load data: {e}")

# Validate Target Column
if TARGET_COL not in train_df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in train data.")

# Identify Feature Columns
feature_cols = [col for col in train_df.columns if col != TARGET_COL]

# Helper function to process missing values and categoricals
def preprocess_data(df, feature_columns):
    df_clean = df[feature_columns].copy()
    
    # Identify numeric and categorical columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Impute numeric missing values with median
    numeric_imputer = SimpleImputer(strategy='median')
    df_clean[numeric_cols] = numeric_imputer.fit_transform(df_clean[numeric_cols])
    
    # Handle categorical missing values
    # Replace NaN with a special marker string 'UNKNOWN'
    cat_missing_mask = df_clean[categorical_cols].isnull().any(axis=0)
    # We also need to apply fillna to each categorical column individually
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            # Store unique values to fill with most frequent if needed, but generic is safer
            # For LightGBM, just fill with a safe object value or drop. 
            # Assuming fill_na is better to keep dimensions.
            df_clean[col] = df_clean[col].fillna('___FILLED___')
            # Convert remaining NaNs if any
            df_clean[col] = df_clean[col].astype(str)
    
    # Convert all object columns to strings to ensure consistency for LightGBM handling if needed
    # LightGBM handles string cat features if marked.
    return df_clean

# Preprocess Train and Test
X_train = preprocess_data(train_df, feature_cols)
X_test = preprocess_data(test_df, feature_cols)

# Prepare Model Parameters
# Detect categorical indices for LightGBM
cat_feature_indices = []
for i, col in enumerate(feature_cols):
    if isinstance(X_train[col].values.dtype, str):
        continue
    
    # Check if column contains mixed types or object
    try:
        sample_val = X_train[col].iloc[0]
        if not isinstance(sample_val, (int, float, bool, np.integer)):
            cat_feature_indices.append(i)
    except TypeError:
        cat_feature_indices.append(i)

# Filter out columns that might be already numerical
cat_feature_indices = [i for i in cat_feature_indices if X_train.iloc[:, i].dtype.kind in ['O', 'U']]
# More precise check for string-like dtypes
final_cat_indices = []
for i, col in enumerate(feature_cols):
    if X_train[col].dtype.kind in ('O', 'U'):
        final_cat_indices.append(i)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    ' boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'n_estimators': 1000,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'random_state': RANDOM_STATE,
    'seed': RANDOM_STATE
}

if len(final_cat_indices) > 0:
    params['cat_features'] = final_cat_indices
else:
    params['cat_features'] = []

# Create Dataset Objects for LightGBM
train_data = lgb.Dataset(X_train, label=train_df[TARGET_COL].values, feature_name=list(feature_cols))
test_data = lgb.Dataset(X_test, feature_name=list(feature_cols))

# Train Model
model = lgb.train(params, train_data, num_boost_round=None, callbacks=[], valid_sets=[(train_data)], verbose_eval=False)

# Make Predictions
predictions = model.predict(test_data, num_iteration=model.best_iteration)

# Format Output DataFrame
ids = list(range(len(predictions)))
output_df = pd.DataFrame({
    'id': ids,
    'IMU Score': predictions
})

# Save to CSV
output_df.to_csv(OUTPUT_CSV, index=False)