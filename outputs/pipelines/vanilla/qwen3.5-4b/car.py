import pandas as pd
import numpy as np
import os

# Allowed libraries only
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

# Dataset Paths
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/openml/car/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/openml/car/data/test.parquet'
TARGET_COL = 'class'
OUTPUT_FILE = 'predictions.csv'

# Load Data
try:
    df_train = pd.read_parquet(TRAIN_PATH)
    df_test = pd.read_parquet(TEST_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Training/Test files not found at specified paths: {TRAIN_PATH}, {TEST_PATH}")

# Prepare Target and Features
y_train = df_train[TARGET_COL]
y_test = df_test[TARGET_COL]
X_train = df_train.drop(columns=[TARGET_COL])
X_test = df_test.drop(columns=[TARGET_COL])

# Separate Numeric and Categorical Columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, ['numeric']),
    ('cat', categorical_transformer, ['category', 'object'])
], remainder='passthrough')

# Fit Preprocessor on Train
preprocessor.fit(X_train)

# Transform Features
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Get Feature Names after OHE (not strictly needed for LightGBM but useful for debugging)
feature_names = None

# Train Model
model = LGBMClassifier(random_state=42, n_estimators=100, max_depth=6)
model.fit(X_train_processed, y_train)

# Predict
y_pred_probs = model.predict_proba(X_test_processed)
y_pred = np.argmax(y_pred_probs, axis=1)

# Save Results
result_df = pd.DataFrame({
    'id': range(len(y_pred)),
    'class': y_pred
})
result_df.to_csv(OUTPUT_FILE, index=False)

print(f"Predictions saved to {OUTPUT_FILE}")