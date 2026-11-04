import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify target column and prepare features
target_col = "APR"
feature_cols = [col for col in train_df.columns if col != target_col]

# Create feature matrix and target
X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]

# Create data for LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, reference=train_data)

# Set up 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Training parameters
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
}

# Initialize results storage
fold_scores = []

# Train on each fold
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]

    train_data_fold = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data_fold = lgb.Dataset(X_val_fold, reference=train_data_fold)

    # Train with callbacks
    model = lgb.train(
        params,
        train_data_fold,
        num_boost_round=1000,
        valid_sets=[val_data_fold],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    # Evaluate on validation set
    preds_val = model.predict(X_val_fold)
    val_rmse = mean_squared_error(y_train.iloc[val_idx], preds_val, squared=False)
    fold_scores.append(val_rmse)

# Calculate cross-validation score
cv_rmse = np.mean(fold_scores)
print(f"Cross-validation RMSE: {cv_rmse:.4f}")

# Train final model on all data
final_data = lgb.Dataset(X_train, label=y_train)
final_model = lgb.train(
    params,
    final_data,
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
)

# Generate predictions for test set
test_preds = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": range(len(test_preds)), "APR": test_preds})
submission.to_csv("./working/submission.csv", index=False)

print("Submission file saved to ./working/submission.csv")
print(f"Final model RMSE on validation: {cv_rmse:.4f}")
