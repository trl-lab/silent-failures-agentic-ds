import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

# Define paths
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/mlr-summary-reports/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/mlr-summary-reports/data/test.parquet"
OUTPUT_PATH = "predictions.csv"

# Load Data
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

# Define Target
TARGET_COL = "Adjusted MLR"

# Ensure Target Column Exists
assert TARGET_COL in train_df.columns, f"Target column {TARGET_COL} not found in train data."

# Prepare Features List
feature_columns = [col for col in train_df.columns if col != TARGET_COL]

# Extract X and y
X_train = train_df[feature_columns]
y_train = train_df[TARGET_COL]
X_test = test_df[feature_columns]

# Preprocessing Setup
# 1. Identify Numeric vs Categorical Columns
numeric_features = []
categorical_features = []

for col in X_train.columns:
    dtype = X_train[col].dtype
    # Check if numeric (int, float) excluding boolean for cleaner handling (treat bool as categorical if present, 
    # or treat as numeric. Here treating non-numeric as categorical)
    if isinstance(dtype, type(pd.StringDtype())) or dtype.name == "category":
        categorical_features.append(col)
    elif np.issubdtype(dtype, np.number):
        numeric_features.append(col)
    # Exclude object if we want, but for auto handling, let's treat 'object' as categorical
    elif dtype.name == "object":
        categorical_features.append(col)

# Define Transformers
imputer_num = SimpleImputer(strategy="median")
encoder_cat = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", imputer_num, numeric_features),
        ("cat", encoder_cat, categorical_features)
    ],
    remainder="passthrough" # In case of remaining columns
)

# Fit Pipeline
preprocessor.fit(X_train)

# Create LightGBM Regressor
model = lgb.LGBMRegressor(
    objective="regression",
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    random_state=42
)

# Build Pipeline
pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", model)
])

# Train Model
pipeline.fit(X_train, y_train)

# Predict on Test Data
y_pred = pipeline.predict(X_test)

# Prepare Output DataFrame with 'id' and 'Adjusted MLR'
predictions_df = pd.DataFrame({
    "id": np.arange(len(y_pred)),
    "Adjusted MLR": y_pred
})

# Save to CSV
predictions_df.to_csv(OUTPUT_PATH, index=False)