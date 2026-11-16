import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score
import numpy as np
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify features and target
target_col = "class"
feature_cols = [col for col in train_df.columns if col != target_col]
X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]

# Encode categorical variables if needed (check column types)
X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()

for col in feature_cols:
    if X_train[col].dtype == "object":
        # Create dummy encoding for categorical features
        X_train_encoded = pd.get_dummies(
            X_train_encoded, columns=[col], drop_first=True
        )
        X_test_encoded = pd.get_dummies(X_test_encoded, columns=[col], drop_first=True)

# Prepare feature matrix
X_train = X_train_encoded[feature_cols]
X_test = X_test_encoded[feature_cols]

# Define LightGBM model
model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

# 5-fold cross-validation for accuracy evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="accuracy")
mean_cv_accuracy = np.mean(cv_scores)
print(f"5-Fold Cross-Validation Accuracy: {mean_cv_accuracy:.4f}")

# Train final model on full training data
model.fit(X_train, y_train)

# Generate predictions for test set
y_pred = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": range(len(y_pred)), "class": y_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
