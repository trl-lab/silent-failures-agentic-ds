import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Encode target
train_df["Diagnosis"] = train_df["Diagnosis"].map({"M": 1, "B": 0})
test_df["Diagnosis"] = test_df["Diagnosis"].map({"M": 1, "B": 0})

# Feature columns
feature_cols = [
    col
    for col in train_df.columns
    if col.endswith("1") or col.endswith("2") or col.endswith("3")
]

# Prepare features
X_train = train_df[feature_cols].values
y_train = train_df["Diagnosis"].values
X_test = test_df[feature_cols].values
y_test = test_df["Diagnosis"].values
test_ids = test_df["ID"].values

# Split into train/validation
X_train_full, X_val, y_train_full, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# LightGBM parameters
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

# Train LightGBM
train_data = lgb.Dataset(X_train_full, label=y_train_full)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Callbacks (early_stopping and log_evaluation)
callbacks = [
    lgb.early_stopping(100, verbose=False),
    lgb.log_evaluation(0, verbose=False),
]

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    callbacks=callbacks,
    verbose_eval=0,
)

# Evaluate on validation set
y_val_pred = model.predict(X_val)
val_auc = roc_auc_score(y_val, y_val_pred)
print(f"Validation ROC AUC: {val_auc:.4f}")

# Predict on test set
test_pred = model.predict(X_test)
test_pred_binary = (test_pred >= 0.5).astype(int)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "Diagnosis": test_pred_binary})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
print(f"Final ROC AUC on validation: {val_auc:.4f}")
