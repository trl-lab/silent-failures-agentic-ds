import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")

# Load data
train_data = pd.read_parquet("./input/train.parquet")
test_data = pd.read_parquet("./input/test.parquet")

# Preprocessing
# Feature engineering - dynamically discover actual columns
# Check available columns and identify numeric features (excluding target)
available_cols = train_data.columns.tolist()
target_col = "Diagnosis"

# Filter out target column and keep only numeric columns
numeric_cols = [col for col in available_cols if col != target_col and train_data[col].dtype in ['float64', 'float32', 'int64', 'int32']]

# Use actual columns from dataset
feature_cols = numeric_cols

# Remove target column from features
X = train_data[feature_cols]
y = train_data[target_col]

# Create test features - use same feature columns as training
test_X = test_data[feature_cols]

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_X_scaled = scaler.transform(test_X)

# Create LightGBM model with callbacks (compatible with lgbm==4.6.0)
model = lgb.LGBMClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbose_eval=False,
)

# Training with callbacks
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]

model.fit(
    X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], callbacks=callbacks
)

# Validation predictions (must use model's forward pass)
# Use predict_proba to get probability scores for ROC AUC
val_predictions = model.predict_proba(X_val_scaled)[:, 1]

# Calculate validation metric (ROC AUC)
val_score = roc_auc_score(y_val, val_predictions)
print(f"Final Validation Score: {val_score:.4f}")

# Test predictions (must use model's forward pass)
test_predictions = model.predict(test_X_scaled)

# Create submission dataframe
submission = pd.DataFrame({"id": range(len(test_data)), "Diagnosis": test_predictions})

# Save submission
submission.to_csv("./submission/submission.csv", index=False)

print("Submission saved to ./submission/submission.csv")
