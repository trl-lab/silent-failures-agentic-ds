import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns and target
target_col = "Class"
feature_cols = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
]

# Encode categorical variables
cat_features = ["job", "marital", "education", "contact", "month", "poutcome"]
label_encoders = {}

for col in cat_features:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))
    label_encoders[col] = le

# Prepare training data
X_train = train_df[feature_cols].values
y_train = train_df[target_col].values

# Prepare test data
X_test = test_df[feature_cols].values

# Handle missing values and fill
X_train = np.nan_to_num(X_train, nan=0.0)
X_test = np.nan_to_num(X_test, nan=0.0)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_preds = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
    X_fold_train = X_train_scaled[train_idx]
    y_fold_train = y_train[train_idx]
    X_fold_val = X_train_scaled[val_idx]
    y_fold_val = y_train[val_idx]

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
    val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)

    # Train model with early stopping
    params = {
        "objective": "binary",
        "metric": "auc",
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
    fold_preds = model.predict(X_fold_val, num_iteration=model.best_iteration)
    all_preds.extend(fold_preds)

# Calculate validation metric
val_preds = np.concatenate(all_preds)
val_metric = roc_auc_score(y_train, val_preds)
print(f"Validation ROC-AUC: {val_metric:.4f}")

# Generate predictions for test set
test_preds = model.predict(X_test_scaled, num_iteration=model.best_iteration)
test_preds = test_preds.reshape(-1, 1)

# Create submission file
submission = pd.DataFrame({"id": range(len(test_df)), "Class": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission file saved to ./working/submission.csv")
