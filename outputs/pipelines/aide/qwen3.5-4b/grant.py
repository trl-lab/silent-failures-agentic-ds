import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define features and target
target = "Federal Funds"
features = [col for col in train_df.columns if col != target]

# Prepare data
X_train = train_df[features].values
y_train = train_df[target].values
X_test = test_df[features].values

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train model with early stopping callback
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbose": -1,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "min_data_in_leaf": 10,
        "max_depth": -1,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "seed": 42,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Evaluate on validation set
    val_preds = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, val_preds) ** 0.5
    scores.append(fold_rmse)
    print(f"Fold {fold_idx + 1} RMSE: {fold_rmse:.4f}")

# Calculate mean RMSE across folds
mean_rmse = np.mean(scores)
print(f"Mean Cross-Validation RMSE: {mean_rmse:.4f}")

# Train final model on all training data
train_data_final = lgb.Dataset(X_train, label=y_train)
model_final = lgb.train(
    params,
    train_data_final,
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Generate predictions for test set
test_preds = model_final.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_df.index, "Federal Funds": test_preds})

# Save to working directory
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define features and target
target = "Federal Funds"
features = [col for col in train_df.columns if col != target]

# Prepare data
X_train = train_df[features].values
y_train = train_df[target].values
X_test = test_df[features].values

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train model with early stopping callback
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbose": -1,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "min_data_in_leaf": 10,
        "max_depth": -1,
        "lambda_l1": 0.01,
        "lambda_l2": 0.01,
        "seed": 42,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Evaluate on validation set
    val_preds = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, val_preds) ** 0.5
    scores.append(fold_rmse)
    print(f"Fold {fold_idx + 1} RMSE: {fold_rmse:.4f}")

# Calculate mean RMSE across folds
mean_rmse = np.mean(scores)
print(f"Mean Cross-Validation RMSE: {mean_rmse:.4f}")

# Train final model on all training data
train_data_final = lgb.Dataset(X_train, label=y_train)
model_final = lgb.train(
    params,
    train_data_final,
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Generate predictions for test set
test_preds = model_final.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_df.index, "Federal Funds": test_preds})

# Save to working directory
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
