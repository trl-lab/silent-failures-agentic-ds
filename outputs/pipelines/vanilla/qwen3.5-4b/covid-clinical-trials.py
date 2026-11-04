import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
import lightgbm as lgb

# Paths as specified
train_path = "/path/to/user/workspace/ds_failures/datasets/strable/covid-clinical-trials/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/covid-clinical-trials/data/test.parquet"

# Load data
df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

target_col = 'days_to_complete'

# Check for existence of target column and ensure it's in the dataframe
assert target_col in df_train.columns, "Target column not found in train data"
assert target_col in df_test.columns, "Target column not found in test data"

# Separate features and target
y_train = df_train[target_col]
X_train = df_train.drop(columns=[target_col])

# Align test features with train features (excluding target) to avoid schema mismatch
valid_cols = [col for col in X_train.columns if col in df_test.columns]
X_test = df_test[valid_cols].copy()

# Identify numeric and categorical columns
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

# Preprocessor to handle missing values and categorical encoding
preprocessor = ColumnTransformer(transformers=[
    ('num', SimpleImputer(strategy='median'), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# Initialize model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', lgb.LGBMRegressor(objective='regression', random_state=42, verbose=-1, n_jobs=-1))
])

# Fit model on training data
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Create output dataframe with id starting from 0
output_df = pd.DataFrame({
    'id': range(len(y_pred)),
    'days_to_complete': y_pred
})

# Save predictions
output_df.to_csv('predictions.csv', index=False)