import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Define file paths
train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/goodreads-books-reviews-290312/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/goodreads-books-reviews-290312/data/test.parquet'

# Load Data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Determine Feature Columns
exclude_cols = {'rating'}
# Ensure 'id' is not included as a feature (even if present in files)
if 'id' in train_df.columns:
    exclude_cols.add('id')

feature_cols = [col for col in train_df.columns if col not in exclude_cols]

# Classify Features
numeric_cols = []
categorical_cols = []
for col in feature_cols:
    if pd.api.types.is_numeric_dtype(train_df[col]):
        numeric_cols.append(col)
    else:
        categorical_cols.append(col)

# Define Preprocessors
num_imputer = SimpleImputer(strategy='median')
cat_preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mode')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_imputer, numeric_cols),
        ('cat', cat_preprocessor, categorical_cols)
    ]
)

# Define Model
model = LGBMRegressor(
    n_estimators=100,
    objective='regression',
    metric='rmse',
    random_state=42,
    verbosity=-1
)

# Define Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', model)
])

# Fit
pipeline.fit(train_df[feature_cols], train_df['rating'])

# Predict
X_test = test_df[feature_cols]
# Ensure columns align between train and test for robustness
valid_cols = list(set(X_test.columns) & set(feature_cols))
X_test_safe = test_df[valid_cols]

y_pred = pipeline.predict(X_test_safe)

# Save Results
output_df = pd.DataFrame({
    'id': range(len(y_pred)),
    'rating': y_pred
})
output_df.to_csv('predictions.csv', index=False)