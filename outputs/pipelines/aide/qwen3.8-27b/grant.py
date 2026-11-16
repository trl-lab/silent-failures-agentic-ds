import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
import os

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

target = "Federal Funds"

# Separate target and handle potential id column
y = train[target].values
X = train.drop(columns=[target])

if "id" in X.columns:
    X = X.drop(columns=["id"])
if "id" in test.columns:
    test = test.drop(columns=["id"])

# Align columns between train and test
common_cols = [c for c in X.columns if c in test.columns]
X = X[common_cols]
test = test[common_cols]

# Identify categorical features
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Ensure consistent categories between train and test
for col in cat_cols:
    all_cats = pd.Categorical(np.concatenate([X[col].values, test[col].values]))
    X[col] = pd.Categorical(X[col], categories=all_cats.categories)
    test[col] = pd.Categorical(test[col], categories=all_cats.categories)

# Validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 20,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "seed": 42,
}

# Train with early stopping
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
val_data = lgb.Dataset(
    X_val, label=y_val, categorical_feature=cat_cols, reference=train_data
)

model = lgb.train(
    params,
    train_data,
    num_boost_round=5000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

val_pred = model.predict(X_val)
rmse = root_mean_squared_error(y_val, val_pred)
print(f"Validation RMSE: {rmse:.4f}")

# Retrain on full data with best iteration count
full_data = lgb.Dataset(X, label=y, categorical_feature=cat_cols)
final_model = lgb.train(
    params,
    full_data,
    num_boost_round=model.best_iteration,
    callbacks=[lgb.log_evaluation(-1)],
)

test_pred = final_model.predict(test)

submission = pd.DataFrame({"id": np.arange(len(test)), "Federal Funds": test_pred})

os.makedirs("./working", exist_ok=True)
submission.to_csv("./working/submission.csv", index=False)
print(f"Submission saved with {len(submission)} rows.")
