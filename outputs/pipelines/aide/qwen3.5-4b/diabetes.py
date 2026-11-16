import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import roc_auc_score
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define features and target
feature_cols = [
    "pregnancies",
    "glucose",
    "bp",
    "skin",
    "insulin",
    "bmi",
    "pedigree",
    "age",
]
X_train = train_df[feature_cols].values
y_train = train_df["class"].values
X_test = test_df[feature_cols].values

# Create LightGBM model
model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]

    # Train model
    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[lgb.early_stopping(50)],
    )

    # Predict validation set
    val_pred = model.predict(X_val_fold)
    auc = roc_auc_score(y_val_fold, val_pred)
    auc_scores.append(auc)
    print(f"Fold {fold+1} ROC AUC: {auc:.4f}")

# Calculate mean ROC AUC
mean_auc = np.mean(auc_scores)
print(f"\nMean ROC AUC on 5-fold CV: {mean_auc:.4f}")

# Generate test predictions
test_pred = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": range(len(test_pred)), "class": test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
