import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Define target and features
target = "Size of Purchase (MW)"
# Identify feature columns (exclude target and id)
train_cols = [col for col in train.columns if col != target and col != "id"]
test_cols = [col for col in test.columns if col != target and col != "id"]

# Ensure feature columns are consistent
feature_cols = (
    train_cols if train_cols == test_cols else list(set(train_cols + test_cols))
)
# If test has extra columns, keep them? No, use train features.
# If train has extra columns, drop them.
feature_cols = [c for c in feature_cols if c in test.columns]

# Prepare Train and Validation Split (80/20)
train_val_split_idx = int(len(train) * 0.8)
train_df = train.iloc[:train_val_split_idx]
val_df = train.iloc[train_val_split_idx:]

# Prepare Data for LightGBM
X_train = train_df[feature_cols].fillna(train_df[feature_cols].mean())
y_train = train_df[target].values
X_val = val_df[feature_cols].fillna(val_df[feature_cols].mean())
y_val = val_df[target].values

X_test = test[feature_cols].fillna(test[feature_cols].mean())

# Create Datasets
train_set = lgb.Dataset(X_train, label=y_train)
val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

# LightGBM Parameters
params = {"objective": "regression", "metric": "rmse", "verbose": -1}

# Train Model with Early Stopping
num_boost_round = 1000
model = lgb.train(
    params,
    train_set,
    num_boost_round=num_boost_round,
    valid_sets=[val_set],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Evaluate on Validation Set
y_val_pred = model.predict(X_val)
val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
print(f"Validation RMSE: {val_rmse}")

# Predict on Test Set
y_test_pred = model.predict(X_test)

# Prepare Submission
submission = pd.DataFrame(
    {"id": range(len(test)), "Size of Purchase (MW)": y_test_pred}
)

# Save Submission
submission.to_csv("./working/submission.csv", index=False)
