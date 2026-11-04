import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Create feature columns and target
feature_cols = [col for col in train_df.columns if col not in ["NADAC Rate", "id"]]
target_col = "NADAC Rate"

# Prepare training data
X_train = train_df[feature_cols].values
y_train = train_df[target_col].values

# Prepare test data
X_test = test_df[feature_cols].values
test_ids = test_df["id"].values

# Set random seed for reproducibility
np.random.seed(42)

# 5-fold cross-validation for evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []

# LightGBM parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}

# Train model with early stopping on each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)],
        valid_sets=[val_data],
        valid_names=["valid"],
    )

    # Calculate RMSE on validation set
    y_pred_val = model.predict(X_val_fold)
    rmse = mean_squared_error(y_val_fold, y_pred_val, squared=False)
    rmse_scores.append(rmse)
    print(f"Fold {fold+1} RMSE: {rmse:.4f}")

# Average RMSE from cross-validation
avg_rmse = np.mean(rmse_scores)
print(f"Average CV RMSE: {avg_rmse:.4f}")

# Train final model on all training data
train_data_final = lgb.Dataset(X_train, label=y_train)
final_model = lgb.train(
    params,
    train_data_final,
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(100, verbose=False)],
    valid_sets=[train_data_final],
    valid_names=["valid"],
)

# Generate predictions for test data
y_pred_test = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "NADAC Rate": y_pred_test})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
print(f"Final Test RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_test)):.4f}")
