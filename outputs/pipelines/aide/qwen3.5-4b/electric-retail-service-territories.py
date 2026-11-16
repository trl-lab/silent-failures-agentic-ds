import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify features and target
target = "TOTAL_MWH"
feature_cols = [col for col in train_df.columns if col != target]

# Create feature matrix and target
X_train = train_df[feature_cols].copy()
y_train = train_df[target].copy()
X_test = test_df[feature_cols].copy()

# Handle missing values
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_train.mean())

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Set up LightGBM parameters
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

# Create KFold for cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize callbacks
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

# Cross-validation
cv_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
    X_train_fold = X_train_scaled[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train_scaled[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=callbacks,
        verbose_eval=10,
    )

    # Compute RMSE on validation set
    y_pred = model.predict(X_val_fold)
    rmse = np.sqrt(np.mean((y_val_fold - y_pred) ** 2))
    cv_scores.append(rmse)
    print(f"Fold {fold+1}: RMSE = {rmse:.4f}")

# Calculate mean CV RMSE
mean_cv_rmse = np.mean(cv_scores)
print(f"\nMean CV RMSE: {mean_cv_rmse:.4f}")

# Train final model on all training data
train_data = lgb.Dataset(X_train_scaled, label=y_train)
model = lgb.train(
    params, train_data, num_boost_round=1000, callbacks=callbacks, verbose_eval=10
)

# Predict on test set
y_test_pred = model.predict(X_test_scaled)

# Create submission
submission = pd.DataFrame({"id": range(len(test_df)), "TOTAL_MWH": y_test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("\nSubmission saved to ./working/submission.csv")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify features and target
target = "TOTAL_MWH"
feature_cols = [col for col in train_df.columns if col != target]

# Create feature matrix and target
X_train = train_df[feature_cols].copy()
y_train = train_df[target].copy()
X_test = test_df[feature_cols].copy()

# Handle missing values
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_train.mean())

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Set up LightGBM parameters
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

# Create KFold for cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize callbacks
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

# Cross-validation
cv_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
    X_train_fold = X_train_scaled[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train_scaled[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=callbacks,
        verbose_eval=10,
    )

    # Compute RMSE on validation set
    y_pred = model.predict(X_val_fold)
    rmse = np.sqrt(np.mean((y_val_fold - y_pred) ** 2))
    cv_scores.append(rmse)
    print(f"Fold {fold+1}: RMSE = {rmse:.4f}")

# Calculate mean CV RMSE
mean_cv_rmse = np.mean(cv_scores)
print(f"\nMean CV RMSE: {mean_cv_rmse:.4f}")

# Train final model on all training data
train_data = lgb.Dataset(X_train_scaled, label=y_train)
model = lgb.train(
    params, train_data, num_boost_round=1000, callbacks=callbacks, verbose_eval=10
)

# Predict on test set
y_test_pred = model.predict(X_test_scaled)

# Create submission
submission = pd.DataFrame({"id": range(len(test_df)), "TOTAL_MWH": y_test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("\nSubmission saved to ./working/submission.csv")
