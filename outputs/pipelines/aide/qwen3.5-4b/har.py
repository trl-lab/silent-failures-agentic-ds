import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import os

# Set working directory
os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify feature columns (exclude 'class' and 'id' if present)
exclude_cols = ["class", "id"]
feature_cols = [col for col in train_df.columns if col not in exclude_cols]

# Prepare data
X_train = train_df[feature_cols].values
y_train = train_df["class"].values
X_test = test_df[feature_cols].values

# 5-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]

    # Prepare data for LightGBM
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train model with specified callbacks
    params = {
        "objective": "multiclass",
        "num_class": len(np.unique(y_train)),
        "verbose": -1,
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "max_depth": -1,
        "metric": "accuracy",
    }

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=callbacks,
        verbose_eval=False,
    )

    # Predict on validation fold
    y_pred = model.predict(X_val_fold)
    y_pred_class = np.round(y_pred).astype(int)
    cv_acc = accuracy_score(y_val_fold, y_pred_class)
    cv_scores.append(cv_acc)

    # Print progress (optional, but good for monitoring)
    if fold == 0:
        print(f"Fold {fold + 1} CV Accuracy: {cv_acc:.4f}")

# Print mean CV accuracy
mean_cv_accuracy = np.mean(cv_scores)
print(f"Mean Cross-Validation Accuracy: {mean_cv_accuracy:.4f}")

# Train final model on all training data
final_params = {
    "objective": "multiclass",
    "num_class": len(np.unique(y_train)),
    "verbose": -1,
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "max_depth": -1,
    "metric": "accuracy",
}

final_callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]
final_model = lgb.train(
    final_params,
    lgb.Dataset(X_train, label=y_train),
    num_boost_round=1000,
    valid_sets=[],
    callbacks=final_callbacks,
)

# Predict on test set
y_pred_test = final_model.predict(X_test)
y_pred_class_test = np.round(y_pred_test).astype(int)

# Prepare submission
submission = pd.DataFrame({"id": range(len(X_test)), "class": y_pred_class_test})
submission.to_csv("./working/submission.csv", index=False)

print("Submission saved to ./working/submission.csv")
