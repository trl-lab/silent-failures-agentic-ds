import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score
import os

# Set working directories
os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Separate features and target
X_train = train_df.drop(columns=["sii", "id"])
y_train = train_df["sii"]
X_test = test_df.drop(columns=["id"])

# Handle missing values
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

# Define feature columns
feature_cols = X_train.columns.tolist()

# LightGBM parameters
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
}

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
predictions = np.zeros((len(X_test),))
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # Create datasets
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Evaluate on validation set
    val_pred = model.predict(X_val_fold)
    val_pred_binary = (val_pred > 0.5).astype(int)
    cv_scores.append(cohen_kappa_score(y_val_fold, val_pred_binary))

    # Store predictions for test set
    test_pred = model.predict(X_test)
    predictions[test_pred > 0.5] = 1

# Calculate mean cross-validation score
mean_cv_score = np.mean(cv_scores)
print(f"5-Fold Cross-Validation Score (Cohen's Kappa): {mean_cv_score:.4f}")

# Generate submission
submission = pd.DataFrame({"id": test_df["id"], "sii": predictions})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
