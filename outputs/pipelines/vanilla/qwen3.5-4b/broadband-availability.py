import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import lightgbm as lgb

# Configuration
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/broadband-availability/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/broadband-availability/data/test.parquet'
OUTPUT_CSV = 'predictions.csv'
TARGET_COL = 'Population for whom broadband available, 2019 (%)'

# Load Data
df_train = pd.read_parquet(TRAIN_PATH)
df_test = pd.read_parquet(TEST_PATH)

# Identify Features and Target
# Drop target from both datasets to prepare feature matrix
train_features = df_train.drop(columns=[TARGET_COL])
test_features = df_test.drop(columns=[TARGET_COL], errors='ignore')

# Ensure column alignment between train and test
common_cols = list(train_features.columns)
test_features = test_features[common_cols]

# Separate columns into numeric and categorical based on dtypes
numeric_features = train_features.select_dtypes(include=np.number).columns.tolist()
categorical_features = [col for col in train_features.columns if col not in numeric_features]

# Define Transformers
# Numerical Columns: Impute median, Standardize
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical Columns: Impute mode, One-Hot Encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
])

# Combine Transformers
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Define LightGBM Model
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    min_child_samples=20,
    random_state=42,
    verbose=-1
)

# Complete Pipeline
pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', model)
])

# Train Model
pipeline.fit(train_features, df_train[TARGET_COL].values)

# Generate Predictions
y_pred = pipeline.predict(test_features)

# Create Output DataFrame with 'id' starting from 0
output_df = pd.DataFrame({
    'id': list(range(len(test_features))),
    TARGET_COL: y_pred
})

# Save Predictions
output_df.to_csv(OUTPUT_CSV, index=False)