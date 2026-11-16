import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Separate features and target
X_train = train_df.drop("projectPayment", axis=1)
y_train = train_df["projectPayment"]
X_test = test_df

# Identify and convert object columns to numeric
object_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
X_train = X_train.copy()
X_test = X_test.copy()

for col in object_cols:
    # Convert to numeric, treating non-numeric as NaN
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    # Fill missing values with median
    if X_train[col].isna().any():
        X_train[col] = X_train[col].fillna(X_train[col].median())
        X_test[col] = X_test[col].fillna(X_test[col].median())

# Define feature columns
feature_cols = X_train.columns.tolist()


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Separate features and target
X_train = train_df.drop("projectPayment", axis=1).copy()
y_train = train_df["projectPayment"]
X_test = test_df.copy()

# Identify and convert object columns to numeric
object_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

for col in object_cols:
    # Convert to numeric, treating non-numeric as NaN
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    # Fill missing values with median
    if X_train[col].isna().any():
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

# Define feature columns
feature_cols = X_train.columns.tolist()

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize LightGBM parameters
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

# Initialize callbacks
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

# Store predictions for validation
val_predictions = []
val_true = []

# Cross-validation loop
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    val_pred = model.predict(X_val_fold)
    val_predictions.extend(val_pred)
    val_true.extend(y_val_fold.values)

    print(
        f"Fold {fold + 1} RMSE: {mean_squared_error(y_val_fold, val_pred) ** 0.5:.4f}"
    )

# Calculate final RMSE
cv_rmse = mean_squared_error(np.array(val_true), np.array(val_predictions)) ** 0.5
print(f"5-Fold CV RMSE: {cv_rmse:.4f}")

# Make predictions on test data
test_pred = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_df.index, "projectPayment": test_pred})

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
