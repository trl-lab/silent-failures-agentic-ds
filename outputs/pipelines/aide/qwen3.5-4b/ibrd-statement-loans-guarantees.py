import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os

# Set random seed for reproducibility
np.random.seed(42)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define target column and feature columns
target_col = "Original Principal Amount (US$)"
feature_cols = train_df.columns.drop([target_col])

# Prepare features and target
X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]

# Create 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store predictions for each fold
oof_predictions = np.zeros(len(y_train))
cv_scores = []

# Train model on each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Define parameters
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    # Train model with early stopping
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(0),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=callbacks,
        verbose_eval=0,
    )

    # Get predictions for validation fold
    oof_predictions[val_idx] = model.predict(X_train.iloc[val_idx])

    # Calculate RMSE on validation fold
    fold_rmse = mean_squared_error(y_val_fold, oof_predictions[val_idx], squared=False)
    cv_scores.append(fold_rmse)
    print(f"Fold {fold + 1} RMSE: {fold_rmse:.4f}")

# Calculate final CV RMSE
cv_rmse = np.sqrt(np.mean([s**2 for s in cv_scores]))
print(f"\n5-Fold Cross-Validation RMSE: {cv_rmse:.4f}")

# Train final model on all data
train_data_final = lgb.Dataset(X_train, label=y_train)
final_model = lgb.train(
    params,
    train_data_final,
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    verbose_eval=0,
)

# Get predictions for test set
test_predictions = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame(
    {"id": range(len(test_df)), "Original Principal Amount (US$)": test_predictions}
)

submission.to_csv("./working/submission.csv", index=False)
print("\nSubmission saved to ./working/submission.csv")

# Print final test prediction RMSE if we had test labels (we don't, so this is just for reference)
print(f"Test predictions shape: {test_predictions.shape}")
