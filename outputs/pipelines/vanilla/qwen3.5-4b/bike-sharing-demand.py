import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# Define file paths
train_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/bike-sharing-demand/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/bike-sharing-demand/data/test.parquet"

# Load data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Define target and excluded columns (excluding 'count' and 'id' if present)
target_col = "count"
exclude_cols = ["count", "id"]

# Ensure test dataframe doesn't have count column (verify and exclude if present)
if 'count' in test_df.columns:
    test_df = test_df.drop(columns=['count'])
else:
    test_df = test_df.copy()

# Prepare feature sets
# Filter out columns that should not be included in training/test (e.g. if 'id' exists)
cols_to_use = [col for col in train_df.columns if col not in exclude_cols and col not in ['count']]

X_train = train_df[cols_to_use].reset_index(drop=True)
y_train = train_df[target_col].values

X_test = test_df[cols_to_use].copy()

# Imputation for missing values
# LightGBM handles missing values well, but filling ensures robustness
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
object_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

num_imputer = SimpleImputer(strategy='median')
obj_imputer = SimpleImputer(strategy='most_frequent')

preprocess_pipe = ColumnTransformer([
    ('num', num_imputer, numeric_cols),
    ('obj', obj_imputer, object_cols)
])

if numeric_cols or object_cols:
    X_train_processed = preprocess_pipe.fit_transform(X_train)
    X_test_processed = preprocess_pipe.transform(X_test)
else:
    X_train_processed = X_train.values
    X_test_processed = X_test.values

# Model Initialization
model = lgb.LGBMRegressor(n_estimators=500, random_state=42, verbose=-1, early_stopping_rounds=None)

# Train Model
model.fit(X_train_processed, y_train)

# Make Predictions
predictions = model.predict(X_test_processed)

# Format for submission
submission_df = pd.DataFrame({
    'id': range(len(test_df)),
    'count': predictions
})

# Save to CSV
submission_df.to_csv('predictions.csv', index=False)