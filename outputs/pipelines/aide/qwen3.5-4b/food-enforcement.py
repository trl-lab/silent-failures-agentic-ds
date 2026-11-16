import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Extract features and target
X = train.drop(columns=["classification"])
y = train["classification"]
test_ids = test["id"]
test_X = test.drop(columns=["id"])

# Set random seed for reproducibility
RANDOM_SEED = 42


# Initialize LightGBM parameters
params = {
    "objective": "multiclass",
    "num_class": len(train["classification"].unique()),
    "boosting_type": "gbdt",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "verbose": -1,
}

# Set up cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# Store predictions for each fold
predictions = []

# Train on each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    # Split data
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_val_fold = y.iloc[val_idx]

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train with early stopping
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        seed=RANDOM_SEED,
    )

    # Get predictions for validation set
    val_pred = model.predict(X_val_fold)
    predictions.append(val_pred)

    # Calculate accuracy for this fold
    fold_accuracy = accuracy_score(y_val_fold, val_pred)
    print(f"Fold {fold+1} Accuracy: {fold_accuracy:.4f}")

# Average accuracy across all folds
avg_accuracy = np.mean(predictions)
print(f"\nAverage Cross-Validation Accuracy: {avg_accuracy:.4f}")

# Train final model on all training data
final_model = lgb.train(
    params,
    lgb.Dataset(X, label=y),
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    seed=RANDOM_SEED,
)

# Generate predictions for test set
test_predictions = final_model.predict(test_X)
test_predictions = np.argmax(test_predictions, axis=1)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "classification": test_predictions})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Extract features and target
X = train.drop(columns=["classification"])
y = train["classification"]
test_ids = test["id"].values
test_X = test.drop(columns=["id"])

# Set random seed for reproducibility
RANDOM_SEED = 42

# Initialize LightGBM parameters
params = {
    "objective": "multiclass",
    "num_class": len(train["classification"].unique()),
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "colsample_bynode": 0.8,
    "verbose": -1,
}

# Set up 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# Store predictions for each fold
fold_accuracies = []

# Train on each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    # Split data
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_val_fold = y.iloc[val_idx]

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train with early stopping
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        seed=RANDOM_SEED,
    )

    # Get predictions for validation set
    val_pred = model.predict(X_val_fold)
    val_pred_int = np.argmax(val_pred, axis=1)

    # Calculate accuracy for this fold
    fold_accuracy = accuracy_score(y_val_fold, val_pred_int)
    fold_accuracies.append(fold_accuracy)
    print(f"Fold {fold+1} Accuracy: {fold_accuracy:.4f}")

# Average accuracy across all folds
avg_accuracy = np.mean(fold_accuracies)
print(f"\nAverage Cross-Validation Accuracy: {avg_accuracy:.4f}")

# Train final model on all training data
final_model = lgb.train(
    params,
    lgb.Dataset(X, label=y),
    num_boost_round=500,
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    seed=RANDOM_SEED,
)

# Generate predictions for test set
test_predictions = final_model.predict(test_X)
test_predictions_int = np.argmax(test_predictions, axis=1)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "classification": test_predictions_int})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
