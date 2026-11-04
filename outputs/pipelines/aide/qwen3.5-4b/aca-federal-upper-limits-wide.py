import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify columns
target = "ACA FUL"
feature_cols = [col for col in train_df.columns if col != target]

# Identify categorical columns (object dtype)
categorical_cols = [col for col in feature_cols if train_df[col].dtype == "object"]
numeric_cols = [
    col
    for col in feature_cols
    if train_df[col].dtype in ["int64", "float64", "int32", "float32"]
]

# Handle categorical columns - convert to numeric
for col in categorical_cols:
    # Use ordinal encoding for categorical columns
    train_df[col] = train_df[col].astype("category").cat.codes
    test_df[col] = test_df[col].astype("category").cat.codes

# Prepare training data
X_train = train_df[feature_cols]
y_train = train_df[target]

# Split data for validation (use 5-fold CV)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# LightGBM model parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "force_col_wise": True,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
}

# Train with 5-fold cross-validation
best_score = float("inf")
best_model = None

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    train_data = lgb.Dataset(
        X_train_fold, label=y_train_fold, categorical_feature=categorical_cols
    )
    val_data = lgb.Dataset(
        X_val_fold,
        label=y_val_fold,
        reference=train_data,
        categorical_feature=categorical_cols,
    )

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    # Evaluate on validation set
    y_pred = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, y_pred)

    if fold_rmse < best_score:
        best_score = fold_rmse
        best_model = model

# Generate predictions for test set
test_predictions = best_model.predict(test_df[feature_cols])

# Create submission file
submission = pd.DataFrame({"id": test_df.index, "ACA FUL": test_predictions})

submission.to_csv("./working/submission.csv", index=False)
print(f"Best RMSE on validation: {best_score:.4f}")
print("Submission saved to ./working/submission.csv")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify columns
target = "ACA FUL"
feature_cols = [col for col in train_df.columns if col != target]

# Identify categorical columns (object dtype that need conversion)
categorical_cols = [col for col in feature_cols if train_df[col].dtype == "object"]
numeric_cols = [
    col
    for col in feature_cols
    if train_df[col].dtype in ["int64", "float64", "int32", "float32"]
]

# Handle categorical columns - convert to numeric using ordinal encoding
for col in categorical_cols:
    # Create category mapping for consistent train/test encoding
    category_dict = train_df[col].astype("category").cat.codes
    train_df[col] = category_dict
    test_df[col] = test_df[col].astype("category").cat.codes

# Prepare training data
X_train = train_df[feature_cols]
y_train = train_df[target]

# Split data for validation (use 5-fold CV)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# LightGBM model parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "force_col_wise": True,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
}

# Train with 5-fold cross-validation
best_score = float("inf")
best_model = None
best_params = params.copy()

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    train_data = lgb.Dataset(
        X_train_fold, label=y_train_fold, categorical_feature=categorical_cols
    )
    val_data = lgb.Dataset(
        X_val_fold,
        label=y_val_fold,
        reference=train_data,
        categorical_feature=categorical_cols,
    )

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    # Evaluate on validation set
    y_pred = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, y_pred)

    if fold_rmse < best_score:
        best_score = fold_rmse
        best_model = model

# Generate predictions for test set
test_predictions = best_model.predict(test_df[feature_cols])

# Create submission file
submission = pd.DataFrame({"id": test_df.index, "ACA FUL": test_predictions})

submission.to_csv("./working/submission.csv", index=False)
print(f"Best RMSE on validation: {best_score:.4f}")
print("Submission saved to ./working/submission.csv")
