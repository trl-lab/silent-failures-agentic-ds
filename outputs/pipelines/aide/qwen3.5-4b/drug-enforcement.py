import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
lgb.set_params(seed=42)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify features and target
target_col = "classification"
feature_cols = [col for col in train_df.columns if col != target_col]

# Prepare training data
X_train = train_df[feature_cols].values
y_train = train_df[target_col].values
X_test = test_df[feature_cols].values
test_ids = test_df["id"].values if "id" in test_df.columns else range(len(test_df))

# Initialize LightGBM parameters
params = {
    "objective": "multiclass",
    "num_class": len(np.unique(y_train)),
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 8,
    "min_child_samples": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_fold_train = X_train[train_idx]
    y_fold_train = y_train[train_idx]
    X_fold_val = X_train[val_idx]
    y_fold_val = y_train[val_idx]

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
    val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)

    # Train model with callbacks
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1, verbose=False),
        ],
    )

    # Get predictions on validation set
    y_pred = model.predict(X_fold_val)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Calculate accuracy for this fold
    fold_acc = accuracy_score(y_fold_val, y_pred_classes)
    cv_scores.append(fold_acc)
    print(f"Fold {fold+1} accuracy: {fold_acc:.4f}")

# Calculate mean CV accuracy
mean_cv_accuracy = np.mean(cv_scores)
print(f"\nMean Cross-Validation Accuracy: {mean_cv_accuracy:.4f}")

# Train final model on full training data
final_train_data = lgb.Dataset(X_train, label=y_train)
final_model = lgb.train(
    params,
    final_train_data,
    num_boost_round=500,
    callbacks=[
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(-1, verbose=False),
    ],
)

# Get predictions on test set
y_test_pred = final_model.predict(X_test)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "classification": y_test_pred_classes})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print(f"\nSubmission saved to ./working/submission.csv")
print(f"Total predictions: {len(submission)}")
