import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Paths defined in the prompt
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/uci/communities-and-crime/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/uci/communities-and-crime/data/test.parquet'

# Load Data
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

# Extract Target
y_train = train_df['ViolentCrimesPerPop'].values
y_test = test_df['ViolentCrimesPerPop'].values
target_col = 'ViolentCrimesPerPop'
feature_cols = [col for col in train_df.columns if col != target_col]

# Preprocessing Configuration
numeric_features = train_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
categorical_features = train_df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()

# Define Transformers
numerics_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', None) # Scaling not strictly required for GB but ok
])
numerics_transformer.set_params(**{'scaler': None}) # Disable scaler to avoid import issues if not present in older versions, focus on imputation

categoricals_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=-999)), # Fill unknown categories
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerics_transformer, numeric_features),
    ('cat', categoricals_transformer, categorical_features)
])

# Combine Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

# Fit Model
model_pipeline.fit(train_df[feature_cols], y_train)

# Prepare Test Data for Prediction
# We need to ensure the same column mapping and preprocessing
X_test_raw = test_df[feature_cols]

# Ensure consistent ordering and types (copy logic from train preprocessing)
# Since we fitted ColumnTransformer on specific column names, transform uses that order.
X_test_processed = model_pipeline.named_steps['preprocessor'].transform(X_test_raw)

# Predictions
y_pred = model_pipeline.named_steps['regressor'].predict(X_test_processed)

# Create Output DataFrame
output_df = pd.DataFrame({
    'id': range(len(y_test)),
    'ViolentCrimesPerPop': y_pred
})

# Save
output_df.to_csv('predictions.csv', index=False)