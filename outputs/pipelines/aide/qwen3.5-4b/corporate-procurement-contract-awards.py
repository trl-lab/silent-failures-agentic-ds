import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define features and target
target_col = "Contract Award Amount (USD)"
feature_cols = [col for col in train_df.columns if col != target_col]

# Split training data
train_features, val_features, train_labels, val_labels = train_test_split(
    train_df[feature_cols],
    train_df[target_col],
    test_size=0.2,
    random_state=42,
    shuffle=True,
)

# Scale features
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
val_features_scaled = scaler.transform(val_features)

# Prepare LightGBM data
train_data = lgb.Dataset(train_features_scaled, label=train_labels)
val_data = lgb.Dataset(val_features_scaled, label=val_labels, reference=train_data)

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
    "seed": 42,
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Evaluate on validation set
val_predictions = model.predict(val_features_scaled)
val_rmse = mean_squared_error(val_labels, val_predictions)
print(f"Validation RMSE: {val_rmse:.4f}")

# Predict on test set
test_features = test_df[feature_cols]
test_features_scaled = scaler.transform(test_features)
test_predictions = model.predict(test_features_scaled)

# Create submission
submission = pd.DataFrame(
    {"id": range(len(test_df)), "Contract Award Amount (USD)": test_predictions}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define target and features
target_col = "Contract Award Amount (USD)"
feature_cols = [col for col in train_df.columns if col != target_col]

# Prepare training data
X_train = train_df[feature_cols].values
y_train = train_df[target_col].values

# Split for validation (20%)
X_val, X_test_temp, y_val, y_test_temp = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, shuffle=False
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_temp)

# Prepare LightGBM datasets
train_data = lgb.Dataset(X_train_scaled, label=y_train)
val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)

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
    "min_child_samples": 20,
    "seed": 42,
}

# Train model with early stopping
model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Evaluate on validation set
val_predictions = model.predict(X_val_scaled)
val_rmse = mean_squared_error(y_val, val_predictions)
print(f"Validation RMSE: {val_rmse:.4f}")

# 5-Fold Cross Validation for robust evaluation
cv_model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
    X_train_fold = X_train_scaled[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train_scaled[val_idx]
    y_val_fold = y_train[val_idx]

    train_data_fold = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data_fold = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data_fold)

    cv_model_fold = lgb.train(
        params,
        train_data_fold,
        num_boost_round=500,
        valid_sets=[val_data_fold],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    cv_predictions = cv_model_fold.predict(X_val_fold)
    cv_rmse = mean_squared_error(y_val_fold, cv_predictions)
    cv_rmse_scores.append(cv_rmse)

print(f"5-Fold CV RMSE Scores: {cv_rmse_scores}")
print(f"Mean CV RMSE: {np.mean(cv_rmse_scores):.4f}")

# Predict on test set
test_predictions = cv_model.predict(X_test_scaled)

# Create submission
submission = pd.DataFrame(
    {"id": range(len(test_df)), "Contract Award Amount (USD)": test_predictions}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
