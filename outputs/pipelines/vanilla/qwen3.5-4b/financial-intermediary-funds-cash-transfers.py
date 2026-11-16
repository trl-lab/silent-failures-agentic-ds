import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import lightgbm as lgb

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/financial-intermediary-funds-cash-transfers/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/financial-intermediary-funds-cash-transfers/data/test.parquet"
TARGET_COLUMN = "Amount in USD"

# Load data
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

# Verify target column
assert TARGET_COLUMN in train_df.columns, f"Target column {TARGET_COLUMN} not found in train"
assert TARGET_COLUMN in test_df.columns, f"Target column {TARGET_COLUMN} not found in test"

# Prepare feature matrices
X_train = train_df.drop(columns=[TARGET_COLUMN])
y_train = train_df[TARGET_COLUMN]
X_test = test_df.drop(columns=[TARGET_COLUMN])

# Identify numeric and categorical columns
# Number columns include int64, float64, int32, uint32
# Categorical columns include object, category
num_features = X_train.select_dtypes(include=['number']).columns.tolist()
cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Preprocessor
numeric_imputer = SimpleImputer(strategy='median')
categorical_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_imputer, num_features),
    ('cat', categorical_encoder, cat_features)
])

# Regressor
model = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('regressor', lgb.LGBMRegressor(
        objective='regression',
        verbose=-1,
        n_estimators=500,
        random_state=42
    ))
])

# Train model
model.fit(X_train, y_train)

# Predict on test set
predictions = model.predict(X_test)

# Prepare submission output
submission = pd.DataFrame({
    'id': range(len(predictions)),
    'Amount in USD': predictions
})

# Save predictions
submission.to_csv("predictions.csv", index=False)