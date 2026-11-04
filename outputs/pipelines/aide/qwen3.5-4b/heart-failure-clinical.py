import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Separate features and target
X_train = train.drop("death_event", axis=1)
y_train = train["death_event"]
X_test = test.drop("DEATH_EVENT", axis=1)  # Check if DEATH_EVENT exists in test

# Create train/validation split
X_train_val, X_test_split, y_train_val, y_test_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Feature preprocessing - no missing values in this dataset based on description
# LightGBM handles categorical features automatically if they're integers
# For binary features, we can keep them as is (they're already 0/1)

# Prepare training data for LightGBM
train_data = lgb.Dataset(X_train_val, label=y_train_val)
test_data = lgb.Dataset(X_test_split, reference=train_data)

# Create LightGBM model with early stopping
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

# Train with early stopping
model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

# Evaluate on validation set
val_prediction = model.predict(X_test_split)
val_auc = roc_auc_score(y_test_split, val_prediction)
print(f"Validation ROC AUC: {val_auc:.4f}")

# Make predictions on test set
test_prediction = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": range(len(test)), "DEATH_EVENT": test_prediction})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Separate features and target
X_train = train.drop("death_event", axis=1)
y_train = train["death_event"]
X_test = test.copy()  # Test will have no target column

# Create train/validation split (30% validation)
X_train_val, X_test_split, y_train_val, y_test_split = train_test_split(
    X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
)

# Prepare training data for LightGBM
train_data = lgb.Dataset(X_train_val, label=y_train_val, feature_name=X_train.columns)
test_data = lgb.Dataset(
    X_test_split, reference=train_data, feature_name=X_train.columns
)

# Create LightGBM model with early stopping
params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "verbose": -1,
}

# Train with early stopping
model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

# Evaluate on validation set
val_prediction = model.predict(X_test_split)
val_auc = roc_auc_score(y_test_split, val_prediction)
print(f"Validation ROC AUC: {val_auc:.4f}")

# Make predictions on test set
test_prediction = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": range(len(X_test)), "DEATH_EVENT": test_prediction})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
