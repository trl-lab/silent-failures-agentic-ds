import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify features and target
target_col = "days_to_complete"
feature_cols = [col for col in train_df.columns if col != target_col]

# Create feature matrix
X_train = train_df[feature_cols].copy()
X_test = test_df[feature_cols].copy()

# Handle missing values
X_train = X_train.fillna(X_train.mean(numeric_only=True))
X_test = X_test.fillna(X_train.mean(numeric_only=True))

# Handle any remaining NaNs
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# Create callbacks
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1, is_best=True)]

# LightGBM parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbose": -1,
}

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_rmse = float("inf")
best_model = None
best_params = params.copy()

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]

    train_data = lgb.Dataset(X_train_fold, label=train_df.iloc[train_idx][target_col])
    val_data = lgb.Dataset(
        X_val_fold, label=train_df.iloc[val_idx][target_col], reference=train_data
    )

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=callbacks,
        verbose_eval=False,
    )

    # Evaluate on validation set
    val_pred = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(train_df.iloc[val_idx][target_col], val_pred)

    print(f"Fold {fold+1}: RMSE = {fold_rmse:.4f}")

    if fold_rmse < best_rmse:
        best_rmse = fold_rmse
        best_model = model

# Final prediction on test set
test_pred = best_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": range(len(test_pred)), "days_to_complete": test_pred})
submission.to_csv("./working/submission.csv", index=False)

print(f"\nFinal RMSE on validation: {best_rmse:.4f}")
print("Submission saved to ./working/submission.csv")
