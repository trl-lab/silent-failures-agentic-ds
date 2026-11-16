import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# Set random seed for reproducibility
np.random.seed(42)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Separate features and target
X_train = train_df.drop("Status", axis=1)
y_train = train_df["Status"]
X_test = test_df.drop("Status", axis=1)

# Create 5-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store validation scores
val_roc_auc_scores = []

# Train and evaluate on each fold
for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train with early stopping
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Predict on validation set
    val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
    val_roc_auc_scores.append(roc_auc_score(y_val_fold, val_pred))

# Calculate mean validation ROC-AUC
mean_val_auc = np.mean(val_roc_auc_scores)
print(f"5-Fold Cross-Validation ROC-AUC: {mean_val_auc:.4f}")

# Train final model on all training data
train_data = lgb.Dataset(X_train, label=y_train)
final_model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Predict on test data
test_pred = final_model.predict(X_test, num_iteration=final_model.best_iteration)

# Create submission file
submission = pd.DataFrame({"id": range(len(test_df)), "Status": test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
print(f"Test predictions saved with mean ROC-AUC: {mean_val_auc:.4f}")
