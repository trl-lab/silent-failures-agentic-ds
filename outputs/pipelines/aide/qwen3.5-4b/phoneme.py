import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import os

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Separate features and target
X_train = train.drop("class", axis=1)
y_train = train["class"]
X_test = test

# Create validation split for final model evaluation
X_train_full, X_val, y_train_full, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# LightGBM model with early stopping
model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)

# Train with early stopping
model.fit(
    X_train_full,
    y_train_full,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50)],
)

# Evaluate on validation set
y_pred_val = model.predict(X_val)
accuracy_val = accuracy_score(y_val, y_pred_val)
print(f"Validation Accuracy: {accuracy_val:.4f}")

# Predict on test set
y_pred_test = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": range(len(y_pred_test)), "class": y_pred_test})
submission.to_csv("./working/submission.csv", index=False)

print("Submission saved to ./working/submission.csv")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import os

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Separate features and target
X_train = train.drop("class", axis=1)
y_train = train["class"]
X_test = test

# Create 5-fold cross-validation for evaluation
cv_scores = cross_val_score(
    lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
    ),
    X_train,
    y_train,
    cv=5,
    scoring="accuracy",
)

print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Final model training on all training data
model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": range(len(y_pred)), "class": y_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
