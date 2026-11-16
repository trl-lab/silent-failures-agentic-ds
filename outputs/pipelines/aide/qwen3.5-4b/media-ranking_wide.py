import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude id and target)
feature_cols = [col for col in train_df.columns if col not in ["id", "Overall"]]
X_train = train_df[feature_cols].values
y_train = train_df["Overall"].values
X_test = test_df[feature_cols].values

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_rmse = float("inf")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]

    # Create datasets
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # LightGBM parameters
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

    # Train model with early stopping
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(50)],
    )

    # Predict on validation set
    y_pred = model.predict(X_val_fold)
    rmse = mean_squared_error(y_val_fold, y_pred, squared=False)

    print(f"Fold {fold+1}: RMSE = {rmse:.4f}")

# Use best model for final predictions
final_rmse = best_rmse
print(f"Best cross-validation RMSE: {final_rmse:.4f}")

# Generate predictions for test set
test_pred = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_df["id"], "Overall": test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
print(f"Final RMSE on validation: {final_rmse:.4f}")
