import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify target and features
target_col = "heating_load"
feature_cols = [col for col in train_df.columns if col != target_col]

# Combine train and test for feature engineering consistency
train_features = train_df[feature_cols].copy()
test_features = test_df[feature_cols].copy()

# Handle any missing values
train_features = train_features.fillna(train_features.median())
test_features = test_features.fillna(test_features.median())

# Create 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_model = None
best_rmse = float("inf")
fold_rmses = []

# Train model on each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(train_features)):
    X_train, X_val = train_features.iloc[train_idx], train_features.iloc[val_idx]
    y_train = train_df.iloc[train_idx][target_col]
    y_val = train_df.iloc[val_idx][target_col]

    # Define dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Train model with early stopping
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

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Calculate RMSE on validation set for this fold
    val_pred = model.predict(X_val)
    fold_rmse = mean_squared_error(y_val, val_pred, squared=False)
    fold_rmses.append(fold_rmse)

    # Track best model
    if fold_rmse < best_rmse:
        best_rmse = fold_rmse
        best_model = model

# Calculate overall CV RMSE
print(f"5-Fold Cross-Validation RMSE: {np.mean(fold_rmses):.6f}")

# Generate test predictions with best model
test_pred = best_model.predict(test_features)

# Create submission file
submission = pd.DataFrame({"id": range(len(test_pred)), "heating_load": test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
