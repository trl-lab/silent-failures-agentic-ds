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

# Identify feature columns (exclude id and target)
train_features = [col for col in train_df.columns if col not in ["id", "Price"]]
test_features = [col for col in test_df.columns if col not in ["id", "Price"]]

# Prepare training data
X_train = train_df[train_features].values
y_train = train_df["Price"].values
train_id = train_df["id"].values

# Prepare test data
X_test = test_df[test_features].values
test_id = test_df["id"].values

# Create 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store predictions from each fold
fold_predictions = []

# Train and evaluate on each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Define callbacks
    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    # Train model
    model = lgb.train(
        {
            "objective": "regression",
            "metric": "rmse",
            "verbose": -1,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_data_in_leaf": 20,
            "min_sum_hessian_in_leaf": 3,
        },
        train_data,
        num_boost_round=500,
        callbacks=callbacks,
        valid_sets=[val_data],
        valid_names=["valid"],
    )

    # Get validation predictions
    val_pred = model.predict(X_val_fold)
    fold_predictions.append(val_pred)

    # Calculate RMSE for this fold
    fold_rmse = mean_squared_error(y_val_fold, val_pred, squared=False)
    print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}")

# Calculate final RMSE across all folds
fold_rmse_scores = [
    mean_squared_error(y, p, squared=False)
    for y, p in zip(y_train, np.concatenate(fold_predictions))
]
final_rmse = np.sqrt(np.mean(fold_rmse_scores))
print(f"\nFinal 5-fold CV RMSE: {final_rmse:.4f}")

# Prepare submission file
predictions = model.predict(X_test)
submission = pd.DataFrame({"id": test_id, "Price": predictions})
submission.to_csv("./working/submission.csv", index=False)
print("\nSubmission file saved to ./working/submission.csv")
