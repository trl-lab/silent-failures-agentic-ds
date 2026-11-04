import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
import os

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Handle potential ID columns
if "id" in test.columns:
    test.drop(columns=["id"], inplace=True)

target = "credit_amount"
if target not in train.columns:
    raise ValueError(f"Target column '{target}' not found in train data.")

train = train.dropna(subset=[target])
y = train[target].astype("float32")
X = train.drop(columns=[target])

# Drop obvious identifier columns from training data
id_cols = [
    col for col in X.columns if col.lower() in ["id", "index", "row_id", "uuid", "guid"]
]
if id_cols:
    X.drop(columns=id_cols, inplace=True)

# Align train and test columns
common_cols = X.columns.intersection(test.columns)
X = X[common_cols]
test_clean = test[common_cols]

# Preprocess features
for col in X.columns:
    if X[col].dtype == "object" or X[col].dtype == "category":
        X[col] = X[col].astype("category")
        test_clean[col] = test_clean[col].astype("category")
    else:
        X[col] = X[col].astype("float32")
        test_clean[col] = test_clean[col].astype("float32")

# Replace infinities with NaNs (LightGBM handles NaNs natively)
X.replace([np.inf, -np.inf], np.nan, inplace=True)
test_clean.replace([np.inf, -np.inf], np.nan, inplace=True)

# Create validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Model parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "verbose": -1,
}

# Train with early stopping
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[val_data],
    valid_names=["validation"],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Evaluate on validation set
preds_val = model.predict(X_val, num_iteration=model.best_iteration)
rmse = root_mean_squared_error(y_val, preds_val)
print(f"Validation RMSE: {rmse:.4f}")

# Retrain on full training data with the best iteration found
full_train_data = lgb.Dataset(X, label=y)
model_full = lgb.train(params, full_train_data, num_boost_round=model.best_iteration)

# Predict on test set
preds_test = model_full.predict(test_clean, num_iteration=model_full.best_iteration)

# Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": range(len(test_clean)), "credit_amount": preds_test})
submission.to_csv("./working/submission.csv", index=False)
