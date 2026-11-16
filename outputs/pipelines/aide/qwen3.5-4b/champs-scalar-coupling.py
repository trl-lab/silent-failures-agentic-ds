import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import os

# Set random seed for reproducibility
np.random.seed(42)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify features (all columns except id and target)
feature_cols = [
    col for col in train_df.columns if col not in ["id", "scalar_coupling_constant"]
]
X_train = train_df[feature_cols]
y_train = train_df["scalar_coupling_constant"]
X_test = test_df[feature_cols]

# Create 5-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Store predictions for each fold
fold_preds = []
fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # Prepare LightGBM data
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train model with early stopping
    model = lgb.train(
        {
            "objective": "regression",
            "metric": "mae",
            "verbose": -1,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_data_in_leaf": 20,
            "min_sum_hessian_in_leaf": 3,
            "seed": 42,
        },
        train_data,
        num_boost_round=5000,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    # Predict on validation fold
    val_pred = model.predict(X_val_fold)
    val_pred = np.clip(val_pred, 0, 10)  # Clamp predictions to reasonable range

    fold_preds.append(val_pred)
    fold_scores.append(mean_absolute_error(y_val_fold, val_pred))

# Calculate mean MAE across folds
mean_mae = np.mean(fold_scores)
print(f"Cross-validation MAE: {mean_mae:.4f}")

# Train final model on all training data
train_data_final = lgb.Dataset(X_train, label=y_train)
final_model = lgb.train(
    {
        "objective": "regression",
        "metric": "mae",
        "verbose": -1,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 20,
        "min_sum_hessian_in_leaf": 3,
        "seed": 42,
    },
    train_data_final,
    num_boost_round=5000,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
)

# Predict on test data
test_pred = final_model.predict(X_test)
test_pred = np.clip(test_pred, 0, 10)  # Clamp predictions to reasonable range

# Create submission file
submission = pd.DataFrame({"id": test_df["id"], "scalar_coupling_constant": test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
print(f"Test MAE (cross-validation): {mean_mae:.4f}")
