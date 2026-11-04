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

# Identify target and features
target = "AwardTotal"
feature_cols = [col for col in train_df.columns if col != target]

# Ensure feature columns exist in test data
test_feature_cols = [col for col in test_df.columns if col in feature_cols]

# Create feature matrix and target
X_train = train_df[feature_cols].values
y_train = train_df[target].values
X_test = test_df[test_feature_cols].values

# 5-Fold Cross Validation for evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]

    # LightGBM parameters
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
    }

    # Create datasets
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train with callbacks
    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1, is_train=False)]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    # Predict on validation set
    y_pred_fold = model.predict(X_val_fold)
    fold_scores.append(mean_squared_error(y_val_fold, y_pred_fold))

# Calculate average RMSE from cross-validation
avg_rmse = np.sqrt(np.mean(fold_scores))
print(f"5-Fold Cross-Validation RMSE: {avg_rmse:.4f}")

# Train final model on all training data
final_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

final_data = lgb.Dataset(X_train, label=y_train)
final_model = lgb.train(final_params, final_data, num_boost_round=1000)

# Generate predictions for test set
test_predictions = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame(
    {"id": range(len(test_predictions)), "AwardTotal": test_predictions}
)

# Save to working directory
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
print(f"Test predictions shape: {submission.shape}")
