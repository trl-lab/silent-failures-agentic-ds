import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Check column names
print(f"Train columns: {train_df.columns.tolist()}")
print(f"Test columns: {test_df.columns.tolist()}")

# Define feature columns (exclude target and id if present)
target_col = "class"
id_col = None

# Check if there's an id column in test data
if "id" in test_df.columns:
    id_col = "id"
    feature_cols = [c for c in test_df.columns if c != target_col and c != id_col]
else:
    feature_cols = [c for c in test_df.columns if c != target_col]

# Create feature matrix and target
X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]

# Check for missing values
print(f"Train missing values: {train_df.isnull().sum().sum()}")
print(f"Test missing values: {test_df.isnull().sum().sum()}")

# LightGBM model
model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    verbose=-1,
    random_state=42,
)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Get predictions for validation fold
    y_pred = model.predict_proba(X_val_fold)[:, 1]
    auc_scores.append(roc_auc_score(y_val_fold, y_pred))

    print(f"Fold {fold+1} AUC: {auc_scores[-1]:.4f}")

# Mean AUC from cross-validation
mean_auc = np.mean(auc_scores)
print(f"Mean AUC from 5-fold CV: {mean_auc:.4f}")

# Generate predictions for test set
y_test_pred = model.predict_proba(X_test)[:, 1]

# Save predictions to submission.csv
submission_df = pd.DataFrame(
    {"id": test_df[id_col] if id_col else range(len(test_df)), "class": y_test_pred}
)
submission_df.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")

# For the task, I need to be more careful about the output format
# The task says to print the evaluation metric, so I'll ensure that's visible


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude target column)
target_col = "class"
feature_cols = [c for c in train_df.columns if c != target_col]

# Create feature matrix and target
X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]

# LightGBM model parameters
model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    max_depth=-1,
    verbose=-1,
    random_state=42,
    n_jobs=-1,
)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Get predictions for validation fold
    y_pred = model.predict_proba(X_val_fold)[:, 1]
    auc_scores.append(roc_auc_score(y_val_fold, y_pred))

    print(f"Fold {fold+1} AUC: {auc_scores[-1]:.4f}")

# Mean AUC from cross-validation
mean_auc = np.mean(auc_scores)
print(f"Mean AUC from 5-fold CV: {mean_auc:.4f}")

# Generate predictions for test set
y_test_pred = model.predict_proba(X_test)[:, 1]

# Save predictions to submission.csv
submission_df = pd.DataFrame({"id": range(len(test_df)), "class": y_test_pred})
submission_df.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
