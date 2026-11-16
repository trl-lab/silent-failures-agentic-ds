import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Extract target and feature columns
target_col = "Estimated Total Budget ($)"
target = train[target_col]
feature_cols = [col for col in train.columns if col != target_col]

# Combine train and test for consistent feature engineering
data = pd.concat([train[feature_cols], test[feature_cols]], ignore_index=True)

# Handle missing values
data = data.fillna(0)

# Create one-hot encoding for categorical columns
categorical_cols = data.select_dtypes(include=["object", "category"]).columns
if len(categorical_cols) > 0:
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    feature_cols = data.columns.tolist()

# Split data for validation (80% train, 20% validation)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    data[feature_cols], target, test_size=0.2, random_state=42
)

# Prepare LightGBM data
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Train LightGBM model with early stopping
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

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    callbacks=[
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(100, verbose=False),
    ],
    seed=42,
)

# Evaluate on validation set
y_val_pred = model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"Validation RMSE: {val_rmse:.4f}")

# Cross-validation for more robust evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_fold_train = X_train.iloc[train_idx]
    y_fold_train = y_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]
    y_fold_val = y_train.iloc[val_idx]

    train_data_fold = lgb.Dataset(X_fold_train, label=y_fold_train)
    val_data_fold = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data_fold)

    model_fold = lgb.train(
        params,
        train_data_fold,
        num_boost_round=1000,
        valid_sets=[train_data_fold, val_data_fold],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(100, verbose=False),
        ],
        seed=42,
    )

    y_fold_val_pred = model_fold.predict(X_fold_val)
    fold_rmse = np.sqrt(mean_squared_error(y_fold_val, y_fold_val_pred))
    cv_rmse_scores.append(fold_rmse)
    print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}")

print(f"Mean CV RMSE: {np.mean(cv_rmse_scores):.4f}")

# Predict on test set
test_data = test[feature_cols]
test_predictions = model.predict(test_data)

# Create submission file
submission = pd.DataFrame(
    {"id": range(len(test)), "Estimated Total Budget ($)": test_predictions}
)

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
