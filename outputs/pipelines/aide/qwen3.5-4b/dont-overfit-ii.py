import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Feature selection - keep only informative features
# Drop id and target columns
feature_cols = [col for col in train.columns if col not in ["id", "target"]]

X_train = train[feature_cols]
y_train = train["target"]

X_test = test[feature_cols]
test_ids = test["id"]

# 5-fold cross-validation
# Since we have only 250 samples, we need to be careful
# Use LightGBM with early stopping

# Create callbacks
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

# Train model with cross-validation
# For small dataset, use stratified k-fold
from sklearn.model_selection import StratifiedKFold

# Use stratified k-fold for binary classification
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Train and evaluate
best_score = 0
best_model = None

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_fold_train = X_train.iloc[train_idx]
    y_fold_train = y_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]
    y_fold_val = y_train.iloc[val_idx]

    # Train LightGBM
    model = lgb.train(
        params={
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "verbose": -1,
        },
        train_set=lgb.Dataset(X_fold_train, label=y_fold_train),
        valid_sets=[lgb.Dataset(X_fold_val, label=y_fold_val)],
        callbacks=callbacks,
        num_boost_round=1000,
    )

    # Evaluate
    val_pred = model.predict(X_fold_val)
    val_auc = roc_auc_score(y_fold_val, val_pred)

    print(f"Fold {fold+1}: AUC = {val_auc:.4f}")

    if val_auc > best_score:
        best_score = val_auc
        best_model = model

# Generate predictions on test data
test_pred = best_model.predict(X_test)

# Save submission
submission = pd.DataFrame({"id": test_ids, "target": test_pred})
submission.to_csv("./working/submission.csv", index=False)

print(f"Best validation AUC: {best_score:.4f}")


import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Feature selection - drop id and target columns
feature_cols = [col for col in train.columns if col not in ["id", "target"]]
X_train = train[feature_cols].values
y_train = train["target"].values
X_test = test[feature_cols].values
test_ids = test["id"].values

# 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_auc = 0

# Train and evaluate with early stopping
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]

    # Create LightGBM datasets
    train_set = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_set = lgb.Dataset(X_val_fold, label=y_val_fold)

    # Train with callbacks
    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    model = lgb.train(
        params={
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "min_data_in_leaf": 20,
            "min_sum_hessian_in_leaf": 3,
            "verbose": -1,
        },
        train_set=train_set,
        valid_sets=[val_set],
        callbacks=callbacks,
        num_boost_round=500,
    )

    # Evaluate on validation fold
    val_pred = model.predict(X_val_fold)
    val_auc = roc_auc_score(y_val_fold, val_pred)
    print(f"Fold {fold+1}: ROC-AUC = {val_auc:.4f}")

    # Track best model
    if val_auc > best_auc:
        best_auc = val_auc
        best_model = model

# Generate predictions on test data
test_pred = best_model.predict(X_test)

# Save submission
submission = pd.DataFrame({"id": test_ids, "target": test_pred})
submission.to_csv("./working/submission.csv", index=False)

print(f"Best validation ROC-AUC: {best_auc:.4f}")
