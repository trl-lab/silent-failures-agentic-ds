import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Clean data
train = train.dropna(subset=["usdprice"])
if "usdprice" in test.columns:
    test = test.drop(columns=["usdprice"])

# Separate target and features
target = "usdprice"
y = train[target]
X = train.drop(columns=[target])

# Drop id if it exists
if "id" in X.columns:
    X = X.drop(columns=["id"])
if "id" in test.columns:
    test = test.drop(columns=["id"])

# Ensure test has same columns as X
common_cols = X.columns.intersection(test.columns)
X = X[common_cols]
test = test[common_cols]

# Identify categorical features
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Convert categorical features to category dtype
for col in cat_cols:
    X[col] = X[col].astype("category")
    test[col] = test[col].astype("category")

# Split into train and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LightGBM dataset
dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
dval = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=dtrain)

# Train model
params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "verbose": -1,
}

model = lgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    valid_sets=[dval],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Evaluate on validation set
val_pred = model.predict(X_val, num_iteration=model.best_iteration)
val_rmse = root_mean_squared_error(y_val, val_pred)
print(f"Validation RMSE: {val_rmse}")

# Retrain on full training data
dtrain_full = lgb.Dataset(X, label=y, categorical_feature=cat_cols)
model_full = lgb.train(
    params,
    dtrain_full,
    num_boost_round=model.best_iteration,
    callbacks=[lgb.log_evaluation(-1)],
)

# Predict on test data
test_pred = model_full.predict(test, num_iteration=model.best_iteration)

# Save submission
submission = pd.DataFrame({"id": range(len(test)), "usdprice": test_pred})
submission.to_csv("./working/submission.csv", index=False)
