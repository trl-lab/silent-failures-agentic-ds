import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Define file paths
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/transmission-lines/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/transmission-lines/data/test.parquet"
OUTPUT_PATH = "predictions.csv"

# Load datasets
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

# Define target column
TARGET_COL = "VOLT_CLASS"

# Separate features from target
X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]

# Align test features with train features to ensure consistency
test_feature_names = list(set(X_train.columns).intersection(set(test_df.columns)))
X_test = test_df[test_feature_names]

# Identify column types for preprocessing
numeric_features = []
categorical_features = []

for col in X_train.columns:
    if pd.api.types.is_numeric_dtype(train_df[col]):
        numeric_features.append(col)
    elif pd.api.types.is_string_dtype(train_df[col]) or train_df[col].dtype == object:
        # Check if numeric conversion is possible but treat as string first
        # If it's int/float stored as object, treat as categorical for safety in ML context
        if pd.api.types.infer_objects_from_dtype(train_df[col]) not in [np.float64, np.int64, np.int32]:
            categorical_features.append(col)
        else:
            numeric_features.append(col)
    else:
        # Treat any other type as categorical
        categorical_features.append(col)

# Preprocessors
# 1. Numerical Imputation and Scaling
# 2. Categorical One-Hot Encoding
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical_features)
    ]
)

# Split preprocessed data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Initialize and Train Model
model = xgb.XGBClassifier(objective="auto", use_label_encoder=True, eval_metric="multiclass", random_state=42)
model.fit(X_train_processed, y_train)

# Predictions
y_pred = model.predict(X_test_processed)

# Prepare Output DataFrame with required columns: "id" (int row index starting from 0), "VOLT_CLASS"
output_df = pd.DataFrame({
    "id": range(len(y_pred)),
    "VOLT_CLASS": y_pred
})

# Save predictions
output_df.to_csv(OUTPUT_PATH, index=False)