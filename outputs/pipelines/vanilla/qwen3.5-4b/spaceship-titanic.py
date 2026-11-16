import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

warnings = __import__('warnings')
warnings.filterwarnings('ignore')

# Configuration
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/kaggle/spaceship-titanic/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/kaggle/spaceship-titanic/data/test.parquet'
OUTPUT_PATH = 'predictions.csv'
TARGET_COL = 'Transported'

# Load Data
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

# Determine Feature Columns
feature_cols = [col for col in train_df.columns if col != TARGET_COL]

# Separate Numeric and Categorical
train_num = train_df[feature_cols].select_dtypes(include=['int64', 'float64', 'int32', 'int16', 'uint8', 'bool', 'float32', 'float16'])
train_cat = train_df[feature_cols].select_dtypes(include=['object', 'category'])

test_num = test_df[feature_cols].select_dtypes(include=['int64', 'float64', 'int32', 'int16', 'uint8', 'bool', 'float32', 'float16'])
test_cat = test_df[feature_cols].select_dtypes(include=['object', 'category'])

# Impute Missing Values
# Numeric: Mean, Categorical: Mode
# Helper function to fill
for col in train_num.columns:
    if train_num[col].isna().any():
        train_num[col].fillna(train_num[col].median(), inplace=True)
for col in train_cat.columns:
    if train_cat[col].isna().any():
        mode_val = train_cat[col].mode()
        if len(mode_val) > 0:
            train_cat[col] = train_cat[col].fillna(mode_val.iloc[0])

for col in test_num.columns:
    if test_num[col].isna().any():
        test_num[col].fillna(test_num[col].median(), inplace=True)
for col in test_cat.columns:
    if test_cat[col].isna().any():
        mode_val = test_cat[col].mode()
        if len(mode_val) > 0:
            test_cat[col] = test_cat[col].fillna(mode_val.iloc[0])

# OneHotEncoding Categoricals
# We use the same encoder for both train and test to ensure consistency
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

train_cat_encoded = ohe.fit_transform(train_cat.values)
test_cat_encoded = ohe.transform(test_cat.values)

# Combine Features
train_X = np.hstack((train_num.values, train_cat_encoded))
train_y = train_df[TARGET_COL].values

test_X = np.hstack((test_num.values, test_cat_encoded))

# Train Model
model = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
model.fit(train_X, train_y)

# Predict
predictions = model.predict(test_X)

# Prepare Output
output_df = pd.DataFrame({
    'id': range(len(predictions)),
    'Transported': predictions.astype(int)
})

# Save
output_df.to_csv(OUTPUT_PATH, index=False)