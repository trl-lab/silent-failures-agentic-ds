import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os

# Set random seed for reproducibility
np.random.seed(42)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Feature columns (excluding target and id)
target_col = "medianHouseValue"
feature_cols = [
    "longitude",
    "latitude",
    "housingMedianAge",
    "totalRooms",
    "totalBedrooms",
    "population",
    "households",
    "medianIncome",
]

# Train data
X_train = train_df[feature_cols].values
y_train = np.log1p(train_df[target_col].values)  # Log transform as recommended

# Test data
X_test = test_df[feature_cols].values
test_ids = test_df.index.values

# Create KFold for 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Create LightGBM model
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}

# Initialize callbacks
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

# Initialize LightGBM model
lgb_model = lgb.LGBMRegressor(**lgb_params)

# Cross-validation and evaluation
cv_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    # Train model
    lgb_model.fit(
        X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks, verbose=False
    )

    # Predict on validation set
    y_pred_val = lgb_model.predict(X_val)
    y_pred_val = np.expm1(y_pred_val)  # Inverse log transform

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    cv_scores.append(rmse)
    print(f"Fold {fold+1} RMSE: {rmse:.4f}")

# Average CV RMSE
avg_cv_rmse = np.mean(cv_scores)
print(f"\nAverage Cross-Validation RMSE: {avg_cv_rmse:.4f}")

# Train final model on all training data
lgb_model.fit(
    X_train, y_train, eval_set=[(X_train, y_train)], callbacks=callbacks, verbose=False
)

# Predict on test data
y_pred_test = lgb_model.predict(X_test)
y_pred_test = np.expm1(y_pred_test)  # Inverse log transform

# Create submission file
submission = pd.DataFrame({"id": test_ids, "medianHouseValue": y_pred_test})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("\nSubmission saved to ./working/submission.csv")
print(f"Test predictions shape: {submission.shape}")
