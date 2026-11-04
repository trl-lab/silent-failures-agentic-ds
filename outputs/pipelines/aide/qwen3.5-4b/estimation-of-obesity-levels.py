import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns
feature_cols = [
    "Gender",
    "Age",
    "Height",
    "Weight",
    "family_history_with_overweight",
    "FAVC",
    "FCVC",
    "NCP",
    "CAEC",
    "SMOKE",
    "CH2O",
    "SCC",
    "FAF",
    "TUE",
    "CALC",
    "MTRANS",
]

# Encode categorical variables
categorical_cols = ["Gender", "CAEC", "CALC", "MTRANS"]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))
    label_encoders[col] = le

# Fill missing values
train_df = train_df.fillna(train_df.median(numeric_only=True))
train_df = train_df.fillna(
    train_df.mode(numeric_only=True).iloc[0] if train_df.isna().any() else 0
)
test_df = test_df.fillna(test_df.median(numeric_only=True))
test_df = test_df.fillna(
    test_df.mode(numeric_only=True).iloc[0] if test_df.isna().any() else 0
)

# Prepare training data
X_train = train_df[feature_cols].values
y_train = train_df["NObeyesdad"].values

# Prepare test data
X_test = test_df[feature_cols].values

# Create LightGBM model
model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

# 5-fold cross-validation for evaluation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_fold_train = X_train[train_idx]
    y_fold_train = y_train[train_idx]
    X_fold_val = X_train[val_idx]
    y_fold_val = y_train[val_idx]

    model.fit(
        X_fold_train,
        y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    y_pred = model.predict(X_fold_val)
    score = accuracy_score(y_fold_val, y_pred)
    cv_scores.append(score)
    print(f"Fold {fold+1}: Accuracy = {score:.4f}")

# Calculate mean CV accuracy
mean_cv_accuracy = np.mean(cv_scores)
print(f"\nMean CV Accuracy: {mean_cv_accuracy:.4f}")

# Train final model on all training data
model.fit(X_train, y_train, callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)])

# Generate predictions for test data
y_pred_test = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": range(len(y_pred_test)), "NObeyesdad": y_pred_test})
submission.to_csv("./working/submission.csv", index=False)
print("\nSubmission saved to ./working/submission.csv")

# Print final evaluation metric
print(f"Final Model CV Accuracy: {mean_cv_accuracy:.4f}")
