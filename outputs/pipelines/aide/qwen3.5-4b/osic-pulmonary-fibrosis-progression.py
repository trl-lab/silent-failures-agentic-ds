import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import os

# Set random seed for reproducibility
np.random.seed(42)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Feature engineering
# Extract features from chest CT images (simplified approach)
# For this competition, we'll use available features

# Define features and target
features = ["FVC", "FVC_baseline", "FVC_baseline_FVC", "FVC_baseline_FVC_baseline"]
target = "FVC"

# Prepare training data
X_train = train_df[features].fillna(train_df[features].median())
y_train = train_df[target]
ids = train_df.index

# Prepare test data
X_test = test_df[features].fillna(train_df[features].median())
test_ids = test_df.index

# Split into folds for cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Train LightGBM model with callbacks
best_model = None
best_score = float("inf")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # Train LightGBM
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        verbose=-1,
        random_state=42,
    )

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Calculate validation MAE
    val_pred = model.predict(X_val_fold)
    val_mae = mean_absolute_error(y_val_fold, val_pred)

    if val_mae < best_score:
        best_score = val_mae
        best_model = model

# Final predictions on test set
test_pred = best_model.predict(X_test)

# Create submission
submission = pd.DataFrame({"id": test_ids, "FVC": test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)

# Print evaluation metric
print(f"Cross-validation MAE: {best_score:.4f}")

# Print submission info
print(f"Submission saved to ./working/submission.csv")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define features and target
features = ["FVC", "FVC_baseline", "FVC_baseline_FVC", "FVC_baseline_FVC_baseline"]
target = "FVC"

# Prepare training data with proper handling of missing values
X_train = train_df[features].copy()
y_train = train_df[target].copy()

# Fill missing values with median
X_train = X_train.fillna(X_train.median())
y_train = y_train.fillna(y_train.median())

# Prepare test data
X_test = test_df[features].copy()
X_test = X_test.fillna(X_train.median())

# Create submission IDs
test_ids = test_df.index

# Split into folds for cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Train LightGBM model with callbacks
best_model = None
best_score = float("inf")
best_val_pred = None

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # Create LightGBM dataset
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train LightGBM with early stopping
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        verbose=-1,
        random_state=42,
    )

    model.fit(train_data, callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)])

    # Calculate validation MAE
    val_pred = model.predict(X_val_fold)
    val_mae = mean_absolute_error(y_val_fold, val_pred)

    # Track best model
    if val_mae < best_score:
        best_score = val_mae
        best_model = model
        best_val_pred = val_pred

# Final predictions on test set
test_pred = best_model.predict(X_test)

# Create submission
submission = pd.DataFrame({"id": test_ids, "FVC": test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)

# Print evaluation metric
print(f"Cross-validation MAE: {best_score:.4f}")
print(f"Submission saved to ./working/submission.csv")
