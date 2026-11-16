import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Feature engineering
train_features = train_df.drop("NADAC Per Unit", axis=1)
test_features = test_df

# Create submission file
submission = pd.DataFrame(
    {"id": range(len(test_df)), "NADAC Per Unit": np.zeros(len(test_df))}
)

# Model training with 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_rmse = np.inf

for fold, (train_idx, val_idx) in enumerate(kf.split(train_features)):
    X_train, X_val = train_features.iloc[train_idx], train_features.iloc[val_idx]
    y_train = train_df.iloc[train_idx]["NADAC Per Unit"]
    y_val = train_df.iloc[val_idx]["NADAC Per Unit"]

    # LightGBM model parameters
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbose": -1,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "colsample_bytree": 0.8,
        "colsample_bynode": 0.8,
        "min_child_weight": 1,
        "max_depth": -1,
        "random_state": 42,
    }

    # Train model with early stopping
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=callbacks,
        verbose_eval=False,
    )

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    fold_rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}")

    if fold_rmse < best_rmse:
        best_rmse = fold_rmse
        best_model = model

# Final model evaluation
y_train_pred = best_model.predict(train_features)
train_rmse = np.sqrt(mean_squared_error(train_df["NADAC Per Unit"], y_train_pred))
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Best Validation RMSE: {best_rmse:.4f}")

# Generate test predictions
test_predictions = best_model.predict(test_features)

# Save submission file
submission["NADAC Per Unit"] = test_predictions
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
