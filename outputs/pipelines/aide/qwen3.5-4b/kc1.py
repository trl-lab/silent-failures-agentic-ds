import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns and target
feature_cols = [c for c in train_df.columns if c not in ["defects", "id"]]
target_col = "defects"

# Handle missing values
train_df = train_df.dropna()
test_df = test_df.dropna()

# Ensure consistent feature set
test_df = test_df[feature_cols]

# Train/validation split (70/30) for initial model training
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    train_df[feature_cols],
    train_df[target_col],
    test_size=0.3,
    random_state=42,
    stratify=train_df[target_col],
)

# LightGBM model parameters
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

# Train model with early stopping
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Evaluate on validation set
y_val_pred = model.predict(X_val)
val_auc = roc_auc_score(y_val, y_val_pred)
print(f"Validation ROC-AUC: {val_auc:.4f}")

# 5-fold cross-validation for final evaluation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    lgb.LGBMClassifier(**params), X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1
)
print(f"5-Fold CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Generate predictions on test set
test_preds = model.predict(test_df[feature_cols])
submission = pd.DataFrame({"id": range(len(test_df)), "defects": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
