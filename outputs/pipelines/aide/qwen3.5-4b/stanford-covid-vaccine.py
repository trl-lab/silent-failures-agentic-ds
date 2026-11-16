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
target = "reactivity"
feature_cols = [col for col in train_df.columns if col != target]

# Split train data for validation (80/20)
X_train, X_val, y_train, y_val = train_test_split(
    train_df[feature_cols].values,
    train_df[target].values,
    test_size=0.2,
    random_state=42,
    shuffle=True,
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Create LightGBM datasets
train_data = lgb.Dataset(X_train_scaled, label=y_train)
val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)

# Train LightGBM model
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
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Predictions on validation set
y_val_pred = model.predict(X_val_scaled)

# Calculate evaluation metric
validation_mcrmse = np.sqrt(np.mean((y_val - y_val_pred) ** 2))
print(f"Validation mcrmse: {validation_mcrmse}")

# Test predictions
X_test = test_df[feature_cols].values
X_test_scaled = scaler.transform(X_test)
test_predictions = model.predict(X_test_scaled)

# Save submission
submission = pd.DataFrame({"id": test_df.index, "reactivity": test_predictions})
submission.to_csv("./working/submission.csv", index=False)


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define features and target
target = "reactivity"
feature_cols = [col for col in train_df.columns if col != target]

# Split train data for validation (80/20)
X_train, X_val, y_train, y_val = train_test_split(
    train_df[feature_cols].values,
    train_df[target].values,
    test_size=0.2,
    random_state=42,
    shuffle=True,
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Create LightGBM datasets
train_data = lgb.Dataset(X_train_scaled, label=y_train)
val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)

# Train LightGBM model
params = {
    "objective": "regression",
    "metric": "mse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "min_child_weight": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "colsample_bylevel": 0.8,
    "random_state": 42,
    "verbose": -1,
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Predictions on validation set
y_val_pred = model.predict(X_val_scaled)

# Calculate evaluation metric (mcrmse = mean centered root mean squared error)
val_mcrmse = np.sqrt(np.mean((y_val - y_val_pred) ** 2))
print(f"Validation mcrmse: {val_mcrmse:.6f}")

# Test predictions
X_test = test_df[feature_cols].values
X_test_scaled = scaler.transform(X_test)
test_predictions = model.predict(X_test_scaled)

# Save submission
submission = pd.DataFrame({"id": test_df.index, "reactivity": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
