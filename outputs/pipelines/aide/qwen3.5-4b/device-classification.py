import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Define features and target
feature_cols = [col for col in train.columns if col != "device_class"]
X_train = train[feature_cols]
y_train = train["device_class"]
X_test = test[feature_cols]
test_ids = test.index

# Set seed for reproducibility
np.random.seed(42)

# Create 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store predictions for each fold
fold_predictions = []

# Train LightGBM model with callbacks
params = {
    "objective": "multiclass",
    "num_class": len(np.unique(y_train)),
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

# Train multiple folds
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
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    fold_pred = model.predict(X_val_fold)
    fold_predictions.append(fold_pred)

# Combine predictions
all_predictions = np.concatenate(fold_predictions)
fold_accuracy = accuracy_score(y_train, all_predictions)
print(f"5-fold CV Accuracy: {fold_accuracy:.4f}")

# Make predictions on test set
test_pred = model.predict(X_test)
test_id = range(len(test_ids))
submission = pd.DataFrame({"id": test_id, "device_class": test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Define features and target
feature_cols = [col for col in train.columns if col != "device_class"]
X_train = train[feature_cols].astype(np.float32)
y_train = train["device_class"].astype(np.int64)
X_test = test[feature_cols].astype(np.float32)
test_ids = test.index

# Set seed for reproducibility
np.random.seed(42)

# Create 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store predictions for each fold
fold_predictions = []

# Define LightGBM parameters
params = {
    "objective": "multiclass",
    "num_class": len(np.unique(y_train)),
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "max_depth": -1,
    "verbose": -1,
}

# Train multiple folds
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
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    fold_pred = model.predict(X_val_fold.astype(np.float32))
    fold_predictions.append(fold_pred)

# Combine predictions and compute validation accuracy
all_predictions = np.concatenate(fold_predictions)
fold_accuracy = accuracy_score(y_train, all_predictions.astype(np.int64))
print(f"5-Fold Cross-Validation Accuracy: {fold_accuracy:.4f}")

# Make predictions on test set
test_pred = model.predict(X_test.astype(np.float32))
test_pred = np.round(test_pred).astype(np.int64)

# Create submission file
submission = pd.DataFrame({"id": range(len(test_ids)), "device_class": test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
