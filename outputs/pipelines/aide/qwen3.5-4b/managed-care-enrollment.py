import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
import os

# Set random seed for reproducibility
np.random.seed(42)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Store original IDs for submission
train_ids = (
    train_df.index
    if isinstance(train_df.index, pd.RangeIndex)
    else train_df.index.values
)
test_ids = (
    test_df.index if isinstance(test_df.index, pd.RangeIndex) else test_df.index.values
)

# Feature engineering
# Combine train and test for feature extraction
combined = pd.concat([train_df, test_df], ignore_index=True)

# Identify feature columns (exclude target and id columns)
exclude_cols = ["Total Enrollment", "id"]
feature_cols = [col for col in combined.columns if col not in exclude_cols]

# Handle missing values
for col in feature_cols:
    combined[col] = combined[col].fillna(combined[col].median())

# Prepare features
X = combined[feature_cols].values
y = train_df["Total Enrollment"].values

# Train/validation split (70/30)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# LightGBM model
params = {
    "objective": "regression",
    "metric": "rmse",
    "verbose": -1,
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 20,
    "min_sum_hessian_in_leaf": 3,
}

# Create LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Training with early stopping
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    callbacks=[
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(-1, verbose=False),
    ],
)

# Cross-validation for final evaluation
cv_predictions = cross_val_predict(model, X, y, cv=5)
cv_rmse = np.sqrt(mean_squared_error(y, cv_predictions))
print(f"5-Fold Cross-Validation RMSE: {cv_rmse:.4f}")

# Evaluate on validation set
val_predictions = model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
print(f"Validation Set RMSE: {val_rmse:.4f}")

# Generate test predictions
test_predictions = model.predict(test_df[feature_cols].values)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "Total Enrollment": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
print(f"Final 5-Fold Cross-Validation RMSE: {cv_rmse:.4f}")
