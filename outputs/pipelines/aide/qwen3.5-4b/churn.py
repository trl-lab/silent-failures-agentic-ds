import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Get feature columns (exclude id if present and target column)
target_col = "class"
feature_cols = [col for col in train.columns if col != target_col and col != "id"]

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(
    train[feature_cols],
    train[target_col],
    test_size=0.2,
    random_state=42,
    stratify=train[target_col],
)

# LightGBM parameters
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
}

# Train model with early stopping
model = lgb.LGBMClassifier(**params)
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
)

# 5-fold cross-validation for evaluation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
cv_auc = np.mean(cv_scores)
print(f"5-fold Cross-Validation ROC-AUC: {cv_auc:.4f}")

# Generate predictions for test set
test_pred = model.predict_proba(test[feature_cols])[:, 1]

# Create submission file
submission = pd.DataFrame({"id": range(len(test)), "class": test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
