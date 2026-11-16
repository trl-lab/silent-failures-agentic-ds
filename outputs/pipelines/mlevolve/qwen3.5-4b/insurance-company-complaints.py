import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Read data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Check data
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Train columns: {train_df.columns.tolist()}")
print(f"Test columns: {test_df.columns.tolist()}")

# Identify features and target
target_col = "Recovery"
feature_cols = [col for col in train_df.columns if col != target_col]

# Split data FIRST to prevent data leakage
X_train, X_val, y_train, y_val = train_test_split(
    train_df[feature_cols], train_df[target_col], test_size=0.2, random_state=42
)
X_test = test_df[feature_cols].copy()

# Combine train, val, and test for fitting encoders (to handle ALL unseen labels)
X_all = pd.concat([X_train, X_val, X_test], ignore_index=True)

# Check for missing values
print(f"Train missing: {X_train.isnull().sum().sum()}")
print(f"Val missing: {X_val.isnull().sum().sum()}")
print(f"Test missing: {X_test.isnull().sum().sum()}")

# Handle missing values - fit on all data, transform on each part
def handle_missing(df, feature_cols):
    result = df.copy()
    for col in feature_cols:
        if result[col].isnull().any():
            if result[col].dtype == 'object':
                # For categorical, fill with mode from ALL data
                mode_val = result[col].mode()[0] if len(result[col].mode()) > 0 else 'missing'
                result[col] = result[col].fillna(mode_val)
            else:
                # For numeric, fill with median from ALL data
                result[col] = result[col].fillna(result[col].median())
    return result

X_train_processed = handle_missing(X_train, feature_cols)
X_val_processed = handle_missing(X_val, feature_cols)
X_test_processed = handle_missing(X_test, feature_cols)

# Convert categorical columns to numeric using label encoding
label_encoders = {}
for col in feature_cols:
    if X_train_processed[col].dtype == 'object':
        le = LabelEncoder()
        # Fit on ALL data combined to handle any unseen labels
        le.fit(X_all[col])
        X_train_processed[col] = le.transform(X_train_processed[col])
        X_val_processed[col] = le.transform(X_val_processed[col])
        X_test_processed[col] = le.transform(X_test_processed[col])
        label_encoders[col] = le

# Verify dtypes after preprocessing
print(f"Train dtypes after preprocessing: {X_train_processed.dtypes.tolist()}")
print(f"Val dtypes after preprocessing: {X_val_processed.dtypes.tolist()}")
print(f"Test dtypes after preprocessing: {X_test_processed.dtypes.tolist()}")

# Define LightGBM parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}

# Create training dataset
train_data = lgb.Dataset(X_train_processed, label=y_train)
val_data = lgb.Dataset(X_val_processed, label=y_val, reference=train_data)

# Train model with early stopping
model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Evaluate on validation set
y_val_pred = model.predict(X_val_processed)
val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
print(f"Final Validation Score: {val_rmse:.4f}")

# Predict on test set
test_predictions = model.predict(X_test_processed)

# Create submission dataframe
submission_df = pd.DataFrame({"id": range(len(test_df)), "Recovery": test_predictions})

# Save submission
submission_df.to_csv("./submission/submission.csv", index=False)
print("Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")
print(f"Submission columns: {submission_df.columns.tolist()}")
print(f"Submission head:\n{submission_df.head()}")