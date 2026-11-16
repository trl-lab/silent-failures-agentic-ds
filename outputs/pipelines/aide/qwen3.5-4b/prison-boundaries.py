import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (all columns except target and id)
target_col = "CAPACITY"
feature_cols = [col for col in train_df.columns if col != target_col and col != "id"]

# Prepare features and target
X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize LightGBM parameters
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "verbose": -1,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
}

# Training with callbacks
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

# Store predictions from each fold
fold_preds = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(
        X_val_fold, label=y_train.iloc[val_idx], reference=train_data
    )

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    val_preds = model.predict(X_val_fold)
    fold_preds.append(val_preds)

    # Calculate RMSE for this fold
    rmse = mean_squared_error(y_train.iloc[val_idx], val_preds)
    print(f"Fold {fold+1} RMSE: {rmse:.4f}")

# Average RMSE from cross-validation
avg_rmse = np.mean(fold_preds)
print(f"Average Cross-Validation RMSE: {avg_rmse:.4f}")

# Train final model on all training data
train_data = lgb.Dataset(X_train, label=y_train)
final_model = lgb.train(
    lgb_params, train_data, num_boost_round=1000, callbacks=callbacks
)

# Predict on test data
test_preds = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_df["id"], "CAPACITY": test_preds})

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (all columns except target and id)
target_col = "CAPACITY"
feature_cols = [col for col in train_df.columns if col not in [target_col, "id"]]

# Prepare features and target
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

# Ensure numeric types
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
y_train = y_train.astype("float32")

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize LightGBM parameters
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "verbose": -1,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "seed": 42,
}

# Store predictions from each fold
fold_rmse = []
fold_preds = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    val_preds = model.predict(X_val_fold)
    val_rmse = mean_squared_error(y_val_fold, val_preds)
    fold_rmse.append(val_rmse)
    fold_preds.append(val_preds)

    print(f"Fold {fold+1} RMSE: {val_rmse:.4f}")

# Average RMSE from cross-validation
avg_rmse = np.mean(fold_rmse)
print(f"Average Cross-Validation RMSE: {avg_rmse:.4f}")

# Train final model on all training data
train_data = lgb.Dataset(X_train, label=y_train)
final_model = lgb.train(
    lgb_params, train_data, num_boost_round=1000, callbacks=[lgb.early_stopping(50)]
)

# Predict on test data
test_preds = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": range(len(test_df)), "CAPACITY": test_preds})

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
