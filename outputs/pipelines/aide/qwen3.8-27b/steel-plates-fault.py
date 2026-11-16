import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import os

# Create working directory if it doesn't exist
os.makedirs("./working", exist_ok=True)

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Separate target and features
target_col = "target"
y = train[target_col]
X = train.drop(columns=[target_col])
X_test = test.copy()

# Handle target variable
if y.dtype.name in ["int64", "float64"]:
    y = y.astype(int)
    is_integer_target = True
else:
    labels = sorted(y.unique())
    label_to_int = {label: i for i, label in enumerate(labels)}
    int_to_label = {i: label for i, label in enumerate(labels)}
    y = y.map(label_to_int)
    is_integer_target = False

# Convert features to numeric
X_original = X.apply(pd.to_numeric, errors="coerce")
X_test_original = X_test.apply(pd.to_numeric, errors="coerce")

# 5-fold stratified cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_index, val_index) in enumerate(skf.split(X_original, y)):
    X_train_fold = X_original.iloc[train_index]
    y_train_fold = y.iloc[train_index]
    X_val_fold = X_original.iloc[val_index]
    y_val_fold = y.iloc[val_index]

    # Impute using median from training fold only
    medians_fold = X_train_fold.median()
    medians_fold = medians_fold.fillna(0)  # Handle all-NaN columns
    X_train_fold_imputed = X_train_fold.fillna(medians_fold)
    X_val_fold_imputed = X_val_fold.fillna(medians_fold)

    # Train LightGBM model
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train_fold_imputed, y_train_fold)

    # Evaluate on validation fold
    y_pred = model.predict(X_val_fold_imputed)
    score = accuracy_score(y_val_fold, y_pred)
    cv_scores.append(score)
    print(f"Fold {fold + 1} Accuracy: {score:.4f}")

mean_cv_accuracy = np.mean(cv_scores)
print(f"Mean CV Accuracy: {mean_cv_accuracy:.4f}")

# Train final model on entire training set
medians_full = X_original.median()
medians_full = medians_full.fillna(0)
X_train_full = X_original.fillna(medians_full)
X_test_full = X_test_original.fillna(medians_full)

final_model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)
final_model.fit(X_train_full, y)

# Generate test predictions
y_pred_test = final_model.predict(X_test_full)

# Convert back to original labels if needed
if not is_integer_target:
    y_pred_test = [int_to_label[pred] for pred in y_pred_test]

# Create and save submission
submission = pd.DataFrame({"id": range(len(y_pred_test)), "target": y_pred_test})
submission.to_csv("./working/submission.csv", index=False)
