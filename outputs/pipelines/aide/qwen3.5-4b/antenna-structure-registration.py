import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import os

# Set random seeds for reproducibility
np.random.seed(42)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify feature columns and target
feature_cols = [col for col in train_df.columns if col != "Strucht"]
target_col = "Strucht"

# Create train/test split for validation (80/20)
train_size = int(len(train_df) * 0.8)
train_val = train_df.iloc[:train_size]
val_df = train_df.iloc[train_size:]

# Prepare training data
X_train = train_val[feature_cols].values
y_train = train_val[target_col].values
X_val = val_df[feature_cols].values
y_val = val_df[target_col].values

# LightGBM parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

# Train LightGBM model with early stopping
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

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

# Evaluate on validation set
val_pred = model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Validation RMSE: {val_rmse:.4f}")

# Generate predictions for test data
test_pred = model.predict(test_df[feature_cols])

# Create submission file
submission = pd.DataFrame({"id": range(len(test_df)), "Strucht": test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
