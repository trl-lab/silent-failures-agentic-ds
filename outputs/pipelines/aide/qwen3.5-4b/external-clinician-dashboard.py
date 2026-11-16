import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify features and target
target_col = "Score"
feature_cols = [col for col in train_df.columns if col != target_col]

# Prepare features
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

# Handle missing values
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

# Create train/validation split for evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
validation_scores = []

# 5-fold cross-validation
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]

    # Train LightGBM model
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, reference=train_data)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Predict on validation set
    y_pred = model.predict(X_val_fold)
    rmse = np.sqrt(mean_squared_error(y_train_fold, y_pred))
    validation_scores.append(rmse)

# Calculate mean RMSE across folds
mean_rmse = np.mean(validation_scores)
print(f"5-Fold Cross-Validation RMSE: {mean_rmse:.4f}")

# Train final model on all training data
train_data = lgb.Dataset(X_train, label=y_train)
final_model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Generate test predictions
test_predictions = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": range(len(test_df)), "Score": test_predictions})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission file saved to ./working/submission.csv")
print(f"Final Test RMSE (on validation): {mean_rmse:.4f}")
