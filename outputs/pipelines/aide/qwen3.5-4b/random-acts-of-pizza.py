import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import os

# Set random seed for reproducibility
np.random.seed(42)

# Read data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Create submission directory if it doesn't exist
os.makedirs("./working", exist_ok=True)

# Handle missing values
train_df = train_df.dropna()
test_df = test_df.dropna()

# Split data: 70% train, 30% validation
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    train_df.drop("requester_received_pizza", axis=1),
    train_df["requester_received_pizza"],
    test_size=0.3,
    random_state=42,
    stratify=train_df["requester_received_pizza"],
)

# Convert to numpy arrays
X_train = X_train.values
X_val = X_val.values
y_train = y_train.values
y_val = y_val.values

# LightGBM parameters
lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}

# Cross-validation with early stopping
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_auc = 0

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_fold_train = X_train[train_idx]
    y_fold_train = y_train[train_idx]
    X_fold_val = X_train[val_idx]
    y_fold_val = y_train[val_idx]

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
    val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)

    # Train model with callbacks
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1, verbose=False),
        ],
    )

    # Predict on validation set
    y_pred = model.predict(X_fold_val)

    # Calculate ROC-AUC
    fold_auc = roc_auc_score(y_fold_val, y_pred)
    print(f"Fold {fold+1} ROC-AUC: {fold_auc:.4f}")

    # Keep best model
    if fold_auc > best_auc:
        best_auc = fold_auc
        best_model = model

# Final model evaluation on validation set
final_pred = best_model.predict(X_val)
final_auc = roc_auc_score(y_val, final_pred)
print(f"\nFinal Validation ROC-AUC: {final_auc:.4f}")

# Predict on test set
test_pred = best_model.predict(test_df.drop("requester_received_pizza", axis=1).values)

# Create submission file
submission = pd.DataFrame(
    {"id": range(len(test_pred)), "requester_received_pizza": test_pred}
)
submission.to_csv("./working/submission.csv", index=False)
print("\nSubmission saved to ./working/submission.csv")
print(f"Final ROC-AUC on validation set: {final_auc:.4f}")
