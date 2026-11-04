import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify features and target
target_col = "TOTAL_ACTIVE_ACCOUNTS_"
feature_cols = [col for col in train_df.columns if col != target_col]

# Create feature matrix and target
X_train = train_df[feature_cols].values
y_train = train_df[target_col].values
X_test = test_df[feature_cols].values

# 5-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
best_score = float("inf")
best_model = None

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train model with early stopping
    model = lgb.train(
        {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "verbose": -1,
            "seed": 42,
        },
        train_set=train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    # Calculate RMSE on validation fold
    val_pred = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, val_pred, squared=False)

    print(f"Fold {fold+1}: RMSE = {fold_rmse:.4f}")

    # Track best model
    if fold_rmse < best_score:
        best_score = fold_rmse
        best_model = model

# Final predictions on test set
test_pred = best_model.predict(X_test)
submission = pd.DataFrame(
    {"id": range(len(test_df)), "TOTAL_ACTIVE_ACCOUNTS_": test_pred}
)
submission.to_csv("./working/submission.csv", index=False)

print(f"Final RMSE on validation: {best_score:.4f}")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify features and target
target_col = "TOTAL_ACTIVE_ACCOUNTS_"
feature_cols = [col for col in train_df.columns if col != target_col]

# Create feature matrix and target
X_train = train_df[feature_cols].values
y_train = train_df[target_col].values
X_test = test_df[feature_cols].values

# 5-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
best_score = float("inf")
best_model = None

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train model with early stopping
    model = lgb.train(
        {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "verbose": -1,
            "seed": 42,
        },
        train_set=train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    # Calculate RMSE on validation fold
    val_pred = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, val_pred, squared=False)

    print(f"Fold {fold+1}: RMSE = {fold_rmse:.4f}")

    # Track best model
    if fold_rmse < best_score:
        best_score = fold_rmse
        best_model = model

# Final predictions on test set
test_pred = best_model.predict(X_test)
submission = pd.DataFrame(
    {"id": range(len(test_df)), "TOTAL_ACTIVE_ACCOUNTS_": test_pred}
)
submission.to_csv("./working/submission.csv", index=False)

print(f"Final RMSE on validation: {best_score:.4f}")
