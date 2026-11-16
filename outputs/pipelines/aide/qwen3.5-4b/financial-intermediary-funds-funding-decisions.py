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

# Define feature columns (exclude target and id)
target_col = "Amount in USD"
train_features = [col for col in train_df.columns if col != target_col]
test_features = [col for col in test_df.columns if col != target_col]

# Prepare training data
X_train = train_df[train_features].values
y_train = train_df[target_col].values
X_test = test_df[test_features].values

# Create LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=1000, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1
)

# 5-fold cross-validation for RMSE evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]

    # Train model with early stopping
    lgb_train = lgb.Dataset(X_train_fold, label=y_train_fold)
    lgb_val = lgb.Dataset(X_val_fold, label=y_val_fold, reference=lgb_train)

    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]

    model.fit(
        X_train_fold, y_train_fold, eval_set=[lgb_val], callbacks=callbacks, verbose=-1
    )

    # Predict on validation set
    y_pred = model.predict(X_val_fold)
    cv_scores.append(mean_squared_error(y_val_fold, y_pred, squared=False))

# Calculate average CV RMSE
cv_rmse = np.sqrt(np.mean(cv_scores))
print(f"5-Fold CV RMSE: {cv_rmse:.4f}")

# Train final model on all training data
lgb_train = lgb.Dataset(X_train, label=y_train)
callbacks = [lgb.early_stopping(50, verbose=False)]

model.fit(X_train, y_train, callbacks=callbacks, verbose=-1)

# Generate predictions for test set
y_pred_test = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_df.index, "Amount in USD": y_pred_test})

submission.to_csv("./working/submission.csv", index=False)
print(f"Submission saved to ./working/submission.csv")
print(f"Final predictions shape: {submission.shape}")
