import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os

# Set random seed for reproducibility
np.random.seed(42)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Create feature matrix and target
train_features = train_df.drop(columns=["Disbursements USD (Trust funds)"])
train_target = train_df["Disbursements USD (Trust funds)"]

test_features = test_df.drop(columns=["Disbursements USD (Trust funds)"])

# Handle any missing values (basic cleaning)
train_features = train_features.fillna(train_features.mean())
test_features = test_features.fillna(test_features.mean())

# Initialize LightGBM parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "verbose": -1,
}

# 5-fold cross-validation for evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_features)):
    X_train_fold = train_features.iloc[train_idx]
    y_train_fold = train_target.iloc[train_idx]
    X_val_fold = train_features.iloc[val_idx]
    y_val_fold = train_target.iloc[val_idx]

    # Prepare data for LightGBM
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train model with early stopping
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    # Predict on validation set
    y_pred = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, y_pred, squared=False)
    fold_scores.append(fold_rmse)

    print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}")

# Calculate cross-validation RMSE
cv_rmse = np.mean(fold_scores)
print(f"\n5-Fold Cross-Validation RMSE: {cv_rmse:.4f}")

# Train final model on all training data
train_data_final = lgb.Dataset(train_features, label=train_target)
final_model = lgb.train(
    params,
    train_data_final,
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

# Generate predictions on test set
test_predictions = final_model.predict(test_features)

# Create submission file
submission = pd.DataFrame(
    {"id": range(len(test_df)), "Disbursements USD (Trust funds)": test_predictions}
)

submission.to_csv("./working/submission.csv", index=False)
print("\nSubmission saved to ./working/submission.csv")
print(f"Final CV RMSE: {cv_rmse:.4f}")
