import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import accuracy_score
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude target)
feature_cols = [col for col in train_df.columns if col != "target"]
X_train = train_df[feature_cols]
y_train = train_df["target"]
X_test = test_df[feature_cols]

# Initialize LightGBM model
model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

# 5-fold cross-validation for evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Get validation predictions
val_preds = cross_val_predict(model, X_train, y_train, cv=kf)

# Calculate validation accuracy
val_accuracy = accuracy_score(y_train, val_preds)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Train final model on all training data
model.fit(X_train, y_train)

# Generate predictions for test set
test_preds = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": range(len(test_preds)), "target": test_preds})

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
