import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os

# Task: Gamma Telescope Classification
# Goal: Predict shower class using features from Cherenkov telescope data

# Load training data
train_df = pd.read_parquet('./input/train.parquet')
test_df = pd.read_parquet('./input/test.parquet')

# Inspect available columns
print(f"Training columns: {list(train_df.columns)}")
print(f"Test columns: {list(test_df.columns)}")

# Extract features and target using available columns
# Use columns that exist in both train and test datasets
# From inspection: fLength, fWidth, fSize, fConc, fConc1, fAsym, fM3Long, fM3Trans, fAlpha, fDist
train_cols = set(train_df.columns)
test_cols = set(test_df.columns)

# Find common columns (excluding 'class')
common_cols = sorted(train_cols & test_cols)
feature_cols = [col for col in common_cols if col not in ['class']]

print(f"Using feature columns: {feature_cols}")

X_train = train_df[feature_cols].values
y_train = train_df['class'].values
X_test = test_df[feature_cols].values

# Generate unique IDs for submission
test_ids = np.arange(len(test_df))

# Split training data for validation FIRST to prevent data leakage
X_train_full, X_val, y_train_full, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Define LightGBM model with callbacks
model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    random_state=42,
    verbose=-1
)

# Train model with callbacks
callbacks = [
    lgb.early_stopping(50),
    lgb.log_evaluation(-1)
]

model.fit(
    X_train_full, y_train_full,
    eval_set=[(X_val, y_val)],
    callbacks=callbacks
)

# Make predictions on validation set
y_val_pred_proba = model.predict_proba(X_val)[:, 1]

# Calculate validation score
val_score = roc_auc_score(y_val, y_val_pred_proba)
print(f'Final Validation Score: {val_score}')

# Make predictions on test set
y_test_pred = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    'id': test_ids,
    'class': y_test_pred
})
submission.to_csv('./submission/submission.csv', index=False)

print('Submission saved to ./submission/submission.csv')