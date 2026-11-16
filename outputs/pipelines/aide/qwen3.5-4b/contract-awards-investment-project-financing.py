import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define target column
target_col = "Supplier Contract Amount (USD)"

# Feature columns (everything except target)
feature_cols = [col for col in train_df.columns if col != target_col]

# Create feature matrices
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

# Handle missing values
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define LightGBM parameters
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

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
    X_train_fold = X_train_scaled[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train_scaled[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # Create LightGBM data
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train model with callbacks
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Get predictions
    y_pred = model.predict(X_val_fold)
    cv_scores.append(mean_squared_error(y_val_fold, y_pred))

    print(f"Fold {fold_idx + 1}: RMSE = {np.sqrt(cv_scores[-1]):.4f}")

# Average CV RMSE
cv_rmse = np.sqrt(np.mean(cv_scores))
print(f"\nAverage CV RMSE: {cv_rmse:.4f}")

# Train final model on all training data
train_data_final = lgb.Dataset(X_train_scaled, label=y_train)
final_model = lgb.train(
    params,
    train_data_final,
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Get test predictions
test_predictions = final_model.predict(X_test_scaled)

# Create submission file
submission = pd.DataFrame(
    {"id": test_df.index, "Supplier Contract Amount (USD)": test_predictions}
)
submission.to_csv("./working/submission.csv", index=False)

print("\nSubmission saved to ./working/submission.csv")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define target column
target_col = "Supplier Contract Amount (USD)"

# Feature columns (everything except target)
feature_cols = [col for col in train_df.columns if col != target_col]

# Create feature matrices
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

# Handle missing values
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define LightGBM parameters
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

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
    X_train_fold = X_train_scaled[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train_scaled[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # Create LightGBM data
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train model with callbacks
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Get predictions
    y_pred = model.predict(X_val_fold)
    cv_scores.append(mean_squared_error(y_val_fold, y_pred))

    print(f"Fold {fold_idx + 1}: RMSE = {np.sqrt(cv_scores[-1]):.4f}")

# Average CV RMSE
cv_rmse = np.sqrt(np.mean(cv_scores))
print(f"\nAverage CV RMSE: {cv_rmse:.4f}")

# Train final model on all training data
train_data_final = lgb.Dataset(X_train_scaled, label=y_train)
final_model = lgb.train(
    params,
    train_data_final,
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Get test predictions
test_predictions = final_model.predict(X_test_scaled)

# Create submission file
submission = pd.DataFrame(
    {"id": test_df.index, "Supplier Contract Amount (USD)": test_predictions}
)
submission.to_csv("./working/submission.csv", index=False)

print("\nSubmission saved to ./working/submission.csv")
