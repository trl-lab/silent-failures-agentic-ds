import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os

# Set random seeds for reproducibility
np.random.seed(42)

# Load data
print("Loading data...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Display data shapes and columns
print(f"Train shape: {train_df.shape}")
print(f"Train columns: {list(train_df.columns)}")
print(f"Test shape: {test_df.shape}")
print(f"Test columns: {list(test_df.columns)}")

# Identify feature columns and target
target_col = "default payment next month"

# Find the identifier column in test data (it may have different name than 'id')
test_cols = list(test_df.columns)
if "id" in test_cols:
    test_id_col = "id"
elif "customer_id" in test_cols:
    test_id_col = "customer_id"
else:
    # Use the first column if no obvious id column
    test_id_col = test_cols[0]

print(f"Using test_id_col: {test_id_col}")

feature_cols = [col for col in train_df.columns if col not in [target_col, test_id_col]]

# Prepare training data
X_train = train_df[feature_cols].values
y_train = train_df[target_col].values

# Prepare test data
X_test = test_df[feature_cols].values
test_ids = test_df[test_id_col].values

# Split training data for validation (80/20)
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Create LightGBM datasets
train_data = lgb.Dataset(X_train_split, label=y_train_split)
val_data = lgb.Dataset(X_val_split, label=y_val_split, reference=train_data)

# Define LightGBM parameters
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

# Train model with callbacks (correct LightGBM API)
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
)

# Validate model
y_val_pred = model.predict(X_val_split)
val_auc = roc_auc_score(y_val_split, y_val_pred)
print(f"Final Validation Score (ROC-AUC): {val_auc:.4f}")

# Generate predictions for test set
test_predictions = model.predict(X_test)

# Create submission dataframe
submission = pd.DataFrame(
    {"id": test_ids, "default payment next month": test_predictions}
)

# Save submission file
submission.to_csv("./submission/submission.csv", index=False)
print("Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission.shape}")
print(
    f"Predictions range: [{test_predictions.min():.4f}, {test_predictions.max():.4f}]"
)
