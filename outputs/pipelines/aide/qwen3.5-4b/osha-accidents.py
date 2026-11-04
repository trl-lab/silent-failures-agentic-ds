import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Feature engineering and preprocessing
# Identify features (exclude target column)
target_col = "Task Assigned"
feature_cols = [col for col in train_df.columns if col != target_col]

# Encode categorical features
cat_cols = (
    train_df[feature_cols]
    .select_dtypes(include=["object", "category"])
    .columns.tolist()
)
num_cols = [col for col in feature_cols if col not in cat_cols]

# Handle missing values
train_df[cat_cols] = train_df[cat_cols].fillna(train_df[cat_cols].mode().iloc[0])
test_df[cat_cols] = test_df[cat_cols].fillna(train_df[cat_cols].mode().iloc[0])
train_df[num_cols] = train_df[num_cols].fillna(0)
test_df[num_cols] = test_df[num_cols].fillna(0)

# Encode categorical features
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
    label_encoders[col] = le

# Prepare training data
X_train = train_df[feature_cols].values
y_train = train_df[target_col].values

# Prepare test data
X_test = test_df[feature_cols].values

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_auc = 0
best_model = None

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]

    # Create LightGBM dataset
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train model
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Predict on validation set
    y_pred = model.predict(X_val_fold)

    # Calculate ROC AUC
    auc = roc_auc_score(y_val_fold, y_pred)

    if auc > best_auc:
        best_auc = auc
        best_model = model

# Save predictions for test data
y_test_pred = best_model.predict(X_test)
submission = pd.DataFrame({"id": test_df.index, "Task Assigned": y_test_pred})
submission.to_csv("./working/submission.csv", index=False)

print(f"Best ROC AUC: {best_auc:.4f}")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Task Assigned"
feature_cols = [col for col in train_df.columns if col != target_col]

# Identify categorical columns
cat_cols = (
    train_df[feature_cols]
    .select_dtypes(include=["object", "category"])
    .columns.tolist()
)
num_cols = [col for col in feature_cols if col not in cat_cols]

# Handle missing values
train_df[cat_cols] = train_df[cat_cols].fillna(train_df[cat_cols].mode().iloc[0])
test_df[cat_cols] = test_df[cat_cols].fillna(train_df[cat_cols].mode().iloc[0])
train_df[num_cols] = train_df[num_cols].fillna(0)
test_df[num_cols] = test_df[num_cols].fillna(0)

# Encode categorical features
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
    label_encoders[col] = le

# Prepare training data
X_train = train_df[feature_cols].values
y_train = train_df[target_col].values

# Prepare test data
X_test = test_df[feature_cols].values

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_auc = 0
best_model = None
best_params = None

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train model
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Predict on validation set
    y_pred = model.predict(X_val_fold)

    # Calculate ROC AUC
    auc = roc_auc_score(y_val_fold, y_pred)

    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_params = params.copy()

# Save predictions for test data
y_test_pred = best_model.predict(X_test)
submission = pd.DataFrame({"id": test_df.index, "Task Assigned": y_test_pred})
submission.to_csv("./working/submission.csv", index=False)

print(f"Best ROC AUC: {best_auc:.4f}")
