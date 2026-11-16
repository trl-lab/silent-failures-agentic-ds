import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, IntegerEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# Paths
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/financial-intermediary-funds-funding-decisions/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/financial-intermediary-funds-funding-decisions/data/test.parquet'

# Load Data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Configuration
target_col = "Amount in USD"
random_state = 42
n_estimators = 500

# Feature Columns Definition
exclude_cols = [target_col, 'id'] # Assuming 'id' might exist in parquet, though prompt says generate 0..N
feature_cols = [col for col in test_df.columns if col not in exclude_cols]
feature_cols.sort()
feature_cols = list(set(feature_cols))

print(f"Training Data Shape: {train_df.shape}")
print(f"Test Data Shape: {test_df.shape}")
print(f"Feature Columns: {feature_cols}")

# Identify types
numeric_features = []
categorical_features = []

for col in feature_cols:
    col_val = test_df[col]
    try:
        if pd.api.types.is_numeric_dtype(col_val):
            numeric_features.append(col)
        else:
            categorical_features.append(col)
    except Exception:
        categorical_features.append(col) # Assume categorical if type check fails unexpectedly

# Preprocessing Pipeline
num_imputer = SimpleImputer(strategy='median')
num_transformer = Pipeline(steps=[('imputer', num_imputer)])

cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
cat_transformer = Pipeline(steps=[('encoder', cat_encoder)])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numeric_features),
    ('cat', cat_transformer, categorical_features)
])

model = lgb.LGBMRegressor(
    n_estimators=n_estimators,
    random_state=random_state,
    verbose=-1
)

# Train Model
X_train = preprocessor.fit_transform(train_df[[f for f in feature_cols]])
y_train = train_df[target_col]

# Convert target to list/ndarray
model.fit(X_train, y_train)

# Predict on Test Set
X_test = preprocessor.transform(test_df[[f for f in feature_cols]])
y_pred = model.predict(X_test)

# Prepare Output DataFrame
ids = list(range(len(y_pred)))
predictions_df = pd.DataFrame({
    'id': ids,
    'Amount in USD': y_pred
})

# Save to CSV
output_path = 'predictions.csv'
predictions_df.to_csv(output_path, index=False)

print(f"Predictions saved to {output_path}")