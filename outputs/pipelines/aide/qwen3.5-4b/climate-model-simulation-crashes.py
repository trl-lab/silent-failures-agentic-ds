KeyError: "None of [Index(['param_1', 'param_2', ...])] are in the [columns]"


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Print actual column names for debugging
print("Train columns:", train_df.columns.tolist())
print("Test columns:", test_df.columns.tolist())

# Feature engineering - use actual column names
# Based on data description, we need 18 feature columns and 1 target column
# Target is "outcome"
target_col = "outcome"

# Get feature columns - exclude target column
feature_cols = [col for col in train_df.columns if col != target_col]
print(f"Using {len(feature_cols)} feature columns: {feature_cols}")

# Prepare training data
X_train = train_df[feature_cols].values
y_train = train_df[target_col].values

# Prepare test data
X_test = test_df[feature_cols].values

# Split for validation (70-30 split)
X_train_val, X_test_temp, y_train_val, y_test_temp = train_test_split(
    X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
)

# 5-fold cross-validation for model evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    lgb.LGBMClassifier(),
    X_train_val,
    y_train_val,
    cv=kf,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=0,
)
print(f"5-Fold CV ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Train final model on full training data
model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    min_data_in_leaf=20,
    bagging_fraction=0.8,
    bagging_freq=5,
    feature_fraction=0.8,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

model.fit(
    X_train_val,
    y_train_val,
    eval_set=[(X_train_val, y_train_val), (X_test_temp, y_test_temp)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Predict on test set
y_pred = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": range(len(X_test)), "outcome": y_pred})

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Print actual column names for debugging
print("Train columns:", train_df.columns.tolist())
print("Test columns:", test_df.columns.tolist())

# Feature engineering - use actual column names
# Target column is "outcome"
target_col = "outcome"

# Get feature columns - exclude target column
feature_cols = [col for col in train_df.columns if col != target_col]
print(f"Using {len(feature_cols)} feature columns: {feature_cols}")

# Prepare training data
X_train = train_df[feature_cols].values
y_train = train_df[target_col].values

# Prepare test data
X_test = test_df[feature_cols].values

# Split for validation (70-30 split)
X_train_val, X_test_temp, y_train_val, y_test_temp = train_test_split(
    X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
)

# 5-fold cross-validation for model evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    lgb.LGBMClassifier(),
    X_train_val,
    y_train_val,
    cv=kf,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=0,
)
print(f"5-Fold CV ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Train final model on full training data
model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    min_data_in_leaf=20,
    bagging_fraction=0.8,
    bagging_freq=5,
    feature_fraction=0.8,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

model.fit(
    X_train_val,
    y_train_val,
    eval_set=[(X_train_val, y_train_val), (X_test_temp, y_test_temp)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Predict on test set
y_pred = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": range(len(X_test)), "outcome": y_pred})

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
