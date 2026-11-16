import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Define target and features
target = "Amount in USD"
feature_cols = [col for col in train.columns if col != target]

# Create feature matrix
X_train = train[feature_cols]
y_train = train[target]
X_test = test[feature_cols]

# 5-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
predictions = []
best_score = float("inf")

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]

    # Create datasets
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(
        X_val_fold, label=y_train.iloc[val_idx], reference=train_data
    )

    # Train model with callbacks
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    # Get validation predictions
    val_pred = model.predict(X_val_fold)
    val_rmse = mean_squared_error(y_train.iloc[val_idx], val_pred)

    print(f"Fold {fold+1}: RMSE = {val_rmse:.4f}")
    predictions.extend(val_pred)

# Calculate final CV RMSE
cv_rmse = mean_squared_error(y_train, predictions, squared=False)
print(f"5-Fold CV RMSE: {cv_rmse:.4f}")

# Train final model on all training data
final_train_data = lgb.Dataset(X_train, label=y_train)
final_model = lgb.train(
    params,
    final_train_data,
    num_boost_round=500,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Predict on test data
test_predictions = final_model.predict(X_test)

# Create submission
submission = pd.DataFrame({"id": range(len(test)), "Amount in USD": test_predictions})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
print(f"Final CV RMSE: {cv_rmse:.4f}")
