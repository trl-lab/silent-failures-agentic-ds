import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb

# Data paths
train_path = "/path/to/user/workspace/ds_failures/datasets/uci/online-news-popularity/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/uci/online-news-popularity/data/test.parquet"
output_path = "predictions.csv"

# Load data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Identify target
target_col = 'shares'

# Prepare Feature Matrix
X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.drop(columns=[target_col])

# Handle Categorical/ID columns automatically based on descriptions
# 'url' is described as Categorical/ID which is typically a noisy variable
# We exclude it to ensure clean training, but retain other categorical columns for automatic handling
if 'url' in X_train_raw.columns:
    X_train_raw = X_train_raw.drop(columns=['url'])
    X_test_raw = X_test_raw.drop(columns=['url'])

# Automatically identify numeric and categorical columns for preprocessing
numeric_cols = X_train_raw.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X_train_raw.select_dtypes(include=['object']).columns.tolist()

# Create a ColumnTransformer pipeline to handle missing values and categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_cols),
        ('cat', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ]
)

# Initialize and configure LightGBM Regressor
clf = lgb.LGBMRegressor(
    objective='regression',
    metric='rmse',
    max_depth=4,
    learning_rate=0.05,
    num_leaves=31,
    min_data_in_leaf=20,
    min_sum_hessian_in_leaf=3,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# Full Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', clf)
])

# Fit model on training data
pipeline.fit(X_train_raw.values, train_df[target_col].values)

# Predict on test data
y_pred_proba = pipeline.predict(X_test_raw.values)

# Prepare output DataFrame with 'id' and 'shares'
predictions_df = pd.DataFrame({
    'id': np.arange(len(X_test_raw)),
    'shares': y_pred_proba
})

# Save to CSV
predictions_df.to_csv(output_path, index=False)