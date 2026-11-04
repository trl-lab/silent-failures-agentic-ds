import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Inspect data structure
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Target column: {train_df.columns}")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Prepare features and target
X_train = train_df.drop("Bus age", axis=1)
y_train = train_df["Bus age"]
X_test = test_df

# Check for missing values
print(f"Train missing values: {X_train.isnull().sum().sum()}")
print(f"Test missing values: {X_test.isnull().sum().sum()}")

# Fill missing values with median
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

# Define LightGBM parameters
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "verbose": -1,
    "seed": 42,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "max_depth": -1,
    "colsample_bytree": 0.8,
    "colsample_bynode": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
}

# Initialize KFold for 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store all predictions for validation
y_pred = np.zeros(len(X_train))
y_true = y_train.values

# Train model with cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # Prepare data for LightGBM
    train_data = lgb.Dataset(
        X_train_fold, label=y_train_fold, feature_name=X_train.columns
    )
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train model with callbacks
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1, verbose=False),
        ],
    )

    # Get validation predictions
    val_preds = model.predict(X_val_fold, num_iteration=model.best_iteration)
    y_pred[val_idx] = val_preds

    # Calculate RMSE for this fold
    fold_rmse = np.sqrt(mean_squared_error(y_val_fold, val_preds))
    print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}")

# Calculate final RMSE on validation data
final_rmse = np.sqrt(mean_squared_error(y_train, y_pred))
print(f"\nFinal RMSE on validation set: {final_rmse:.4f}")

# Generate test predictions
test_preds = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": range(len(test_df)), "Bus age": test_preds})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("\nSubmission saved to ./working/submission.csv")
