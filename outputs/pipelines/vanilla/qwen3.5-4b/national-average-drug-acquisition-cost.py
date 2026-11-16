import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split # Not strictly needed, we use full data
from sklearn.metrics import mean_squared_error # Metric not required to output, but logic implies model building

# Paths
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/national-average-drug-acquisition-cost/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/national-average-drug-acquisition-cost/data/test.parquet"
TARGET_COLUMN = "NADAC Per Unit"
OUTPUT_PATH = "predictions.csv"

# Load Data
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

# Separate Target and Features
X_train = train_df.drop(columns=[TARGET_COLUMN])
y_train = train_df[TARGET_COLUMN]

X_test = test_df.drop(columns=[TARGET_COLUMN])

# Identify Categorical and Numeric Columns
# Categorical includes object and category types
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X_train.select_dtypes(include=['number', 'int64', 'int32', 'int16', 'int8', 'uint8', 'float32', 'float64']).columns.tolist()

# Initialize Transforms
# 1. Imputer for numeric (replace NaN with median)
num_imputer = SimpleImputer(strategy='median')
num_imputer.fit(X_train[numerical_cols])

# 2. OneHotEncoder for categorical (drop_last=False to retain info if needed, default is True which drops last to reduce collinearity, good for regression)
cat_encoder = OneHotEncoder(handle_unknown='ignore')
cat_encoder.fit(X_train[categorical_cols])

# Define Pipeline Logic for Train
X_train_processed = pd.concat([
    num_imputer.transform(X_train[numerical_cols]),
    pd.DataFrame(cat_encoder.transform(X_train[categorical_cols])),
    key="dummy"
], axis=1)
# Correct alignment:
# We need a list of arrays for the transformer
X_train_num = num_imputer.transform(X_train[numerical_cols])
X_train_cat = pd.DataFrame(cat_encoder.transform(X_train[categorical_cols]))
X_train_final = pd.concat([X_train_num, X_train_cat], axis=1)

# Prepare Test Data
X_test_num = num_imputer.transform(X_test[numerical_cols])
X_test_cat = pd.DataFrame(cat_encoder.transform(X_test[categorical_cols]))
X_test_final = pd.concat([X_test_num, X_test_cat], axis=1)

# Define LightGBM Model
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    verbose=-1
)

# Train
model.fit(X_train_final, y_train)

# Predict
y_pred = model.predict(X_test_final)

# Create Output DataFrame
# Columns: "id" (integer row index starting from 0) and "NADAC Per Unit"
output_df = pd.DataFrame({
    'id': range(len(test_df)),
    'NADAC Per Unit': y_pred
})

# Save
output_df.to_csv(OUTPUT_PATH, index=False)