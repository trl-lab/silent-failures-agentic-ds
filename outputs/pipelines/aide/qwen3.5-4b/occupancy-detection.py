import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define features and target
feature_cols = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
target_col = "Occupancy"

# Prepare train data - convert to numeric and handle NaN
X_train = train_df[feature_cols].copy()
X_train = X_train.apply(pd.to_numeric, errors="coerce")
X_train = X_train.fillna(0)  # Fill NaN with 0
y_train = train_df[target_col]

# Split into train and validation (70/30) to avoid data leakage
X_train_val, X_val, y_train_val, y_val = train_test_split(
    X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
)

# Prepare training data for LightGBM
train_data = lgb.Dataset(X_train_val, label=y_train_val)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Initialize LightGBM model with simple parameters
params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

# Train model with early stopping and evaluation logging
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Evaluate on validation set
y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
val_auc = roc_auc_score(y_val, y_val_pred)
print(f"Validation ROC AUC: {val_auc:.4f}")

# Prepare test predictions
X_test = test_df[feature_cols].copy()
X_test = X_test.apply(pd.to_numeric, errors="coerce")
X_test = X_test.fillna(0)
test_pred = model.predict(X_test, num_iteration=model.best_iteration)

# Create submission file
submission = pd.DataFrame({"id": range(len(test_df)), "Occupancy": test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define features and target
feature_cols = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
target_col = "Occupancy"

# Prepare train data - convert to numeric and handle NaN
X_train = train_df[feature_cols].copy()
X_train = X_train.apply(pd.to_numeric, errors="coerce")
X_train = X_train.fillna(0)  # Fill NaN with 0
y_train = train_df[target_col]

# Split into train and validation (70/30) to avoid data leakage
X_train_val, X_val, y_train_val, y_val = train_test_split(
    X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
)

# Prepare training data for LightGBM
train_data = lgb.Dataset(X_train_val, label=y_train_val)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Initialize LightGBM model with simple parameters
params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

# Train model with early stopping and evaluation logging
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Evaluate on validation set
y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
val_auc = roc_auc_score(y_val, y_val_pred)
print(f"Validation ROC AUC: {val_auc:.4f}")

# Prepare test predictions
X_test = test_df[feature_cols].copy()
X_test = X_test.apply(pd.to_numeric, errors="coerce")
X_test = X_test.fillna(0)
test_pred = model.predict(X_test, num_iteration=model.best_iteration)

# Create submission file
submission = pd.DataFrame({"id": range(len(test_df)), "Occupancy": test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
