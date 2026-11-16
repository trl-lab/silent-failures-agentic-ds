import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os

# Set working directory
os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Prepare features
feature_cols = ["cell_type", "sm_name", "sm_lincs_id", "SMILES", "control"]
X_train = train_df[feature_cols]
y_train = train_df["__multi_output__"]

# Test data
X_test = test_df[feature_cols]

# Create 5-fold CV for evaluation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Store validation predictions for mrrmse calculation
all_val_preds = []
all_val_targets = []

# Train model with LightGBM
# We'll use a single model since we need to predict all genes together
# Use column subset to reduce memory usage - select most important features
# For this simple solution, we'll use all features but process carefully

# Create LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, reference=train_data)

# Training parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}

# Train for 5000 iterations with early stopping
best_iteration = 0
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    train_data_fold = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data_fold = lgb.Dataset(X_val_fold, reference=train_data_fold)

    # Early stopping callback
    callbacks = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]

    # Train
    model = lgb.train(
        params,
        train_data_fold,
        num_boost_round=5000,
        callbacks=callbacks,
        valid_sets=[val_data_fold],
        valid_names=["valid"],
    )

    # Get validation predictions
    val_preds = model.predict(X_val_fold)
    all_val_preds.append(val_preds)
    all_val_targets.append(y_val_fold)

# Calculate mrrmse on validation set
all_preds = np.vstack(all_val_preds).mean(axis=0)
all_targets = np.vstack(all_val_targets).mean(axis=0)
rmse_per_row = np.sqrt(np.mean((all_preds - all_targets) ** 2, axis=1))
mrrmse = np.sqrt(np.mean(rmse_per_row))
print(f"Validation mrrmse: {mrrmse:.6f}")

# Generate test predictions
test_preds = model.predict(X_test)
submission = pd.DataFrame(
    {"id": range(len(test_preds)), "__multi_output__": test_preds}
)

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
print(f"Test predictions shape: {test_preds.shape}")
