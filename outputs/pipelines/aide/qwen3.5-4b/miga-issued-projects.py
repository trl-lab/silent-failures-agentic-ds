import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify target and features
target = "Gross Issued Amount (US$)"
features = [col for col in train_df.columns if col != target]

# Handle missing values
train_df = train_df.fillna(0)
test_df = test_df.fillna(0)

# Scale features
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_df[features])
test_features_scaled = scaler.transform(test_df[features])

# LightGBM parameters
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
    "random_state": 42,
}

# Split data for validation
X_train = train_df[features]
y_train = train_df[target]

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # Train LightGBM model
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    # Evaluate
    y_pred = model.predict(X_val_fold)
    rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
    rmse_scores.append(rmse)
    print(f"Fold {fold_idx + 1}: RMSE = {rmse:.4f}")

# Calculate mean RMSE from cross-validation
mean_rmse = np.mean(rmse_scores)
print(f"\nMean RMSE (5-fold CV): {mean_rmse:.4f}")

# Train final model on all training data
train_data = lgb.Dataset(train_features_scaled, label=y_train)
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)

# Generate predictions on test data
test_predictions = model.predict(test_features_scaled)

# Create submission file
submission = pd.DataFrame(
    {"id": test_df.index, "Gross Issued Amount (US$)": test_predictions}
)

submission.to_csv("./working/submission.csv", index=False)
print("\nSubmission saved to ./working/submission.csv")


import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Set random seed for reproducibility
np.random.seed(42)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify target and features
target = "Gross Issued Amount (US$)"
feature_cols = [col for col in train_df.columns if col != target]

# Handle missing values by filling with median
train_df = train_df.fillna(train_df[feature_cols].median())
test_df = test_df.fillna(test_df[feature_cols].median())

# Scale features
scaler = StandardScaler()
X_train = train_df[feature_cols].values
y_train = train_df[target].values
X_test = test_df[feature_cols].values

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# LightGBM parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "random_state": 42,
}

# 5-fold cross-validation for evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
    X_train_fold = X_train_scaled[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train_scaled[val_idx]
    y_val_fold = y_train[val_idx]

    # Create datasets
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train model with early stopping
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Evaluate on validation fold
    y_pred = model.predict(X_val_fold)
    rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
    rmse_scores.append(rmse)
    print(f"Fold {fold_idx + 1}: RMSE = {rmse:.4f}")

# Calculate mean RMSE from cross-validation
mean_rmse = np.mean(rmse_scores)
print(f"\nMean RMSE (5-fold CV): {mean_rmse:.4f}")

# Train final model on all training data
train_data = lgb.Dataset(X_train_scaled, label=y_train)
final_model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Generate predictions on test data
test_predictions = final_model.predict(X_test_scaled)

# Create submission file
submission = pd.DataFrame(
    {"id": test_df.index, "Gross Issued Amount (US$)": test_predictions}
)

submission.to_csv("./working/submission.csv", index=False)
print("\nSubmission saved to ./working/submission.csv")
