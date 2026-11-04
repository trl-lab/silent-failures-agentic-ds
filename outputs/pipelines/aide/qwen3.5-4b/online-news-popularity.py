import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define features and target
target = "shares"
feature_cols = [col for col in train_df.columns if col != target]

# Create a combined dataframe for preprocessing
combined = pd.concat([train_df, test_df], axis=0)

# Handle any missing values
combined[feature_cols] = combined[feature_cols].fillna(0)

# Scale features
scaler = StandardScaler()
combined[feature_cols] = scaler.fit_transform(combined[feature_cols])

# Split into train and validation
train_df = combined[: int(len(combined) * 0.8)]
val_df = combined[int(len(combined) * 0.8) :]

# Train LightGBM model with early stopping
train_data = lgb.Dataset(train_df[feature_cols], label=train_df[target])
val_data = lgb.Dataset(val_df[feature_cols], label=val_df[target])

params = {
    "objective": "regression",
    "metric": "rmse",
    "verbose": -1,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 20,
    "min_sum_hessian_in_leaf": 3,
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(10)],
)

# Predict on test set
test_pred = model.predict(test_df[feature_cols])

# Save predictions
submission = pd.DataFrame({"id": test_df.index, "shares": test_pred})
submission.to_csv("./working/submission.csv", index=False)

# Calculate validation RMSE
val_pred = model.predict(val_df[feature_cols])
rmse = np.sqrt(mean_squared_error(val_df[target], val_pred))
print(f"Validation RMSE: {rmse:.4f}")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define features and target
target = "shares"
feature_cols = [col for col in train_df.columns if col != target]

# Create a combined dataframe for preprocessing
combined = pd.concat([train_df, test_df], axis=0)

# Handle any missing values
combined[feature_cols] = combined[feature_cols].fillna(0)

# Create train/validation split (80/20)
train_df = combined[: int(len(combined) * 0.8)]
val_df = combined[int(len(combined) * 0.8) :]

# Scale features to avoid numerical issues
scaler = StandardScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
val_df[feature_cols] = scaler.transform(val_df[feature_cols])
test_df[feature_cols] = scaler.transform(test_df[feature_cols])

# Prepare LightGBM datasets
train_data = lgb.Dataset(train_df[feature_cols], label=train_df[target])
val_data = lgb.Dataset(val_df[feature_cols], label=val_df[target])

# Define LightGBM parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "verbose": -1,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 20,
    "min_sum_hessian_in_leaf": 3,
    "seed": 42,
}

# Train model with early stopping
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Predict on test set
test_pred = model.predict(test_df[feature_cols])

# Save predictions to submission.csv
submission = pd.DataFrame({"id": test_df.index, "shares": test_pred})
submission.to_csv("./working/submission.csv", index=False)

# Calculate validation RMSE
val_pred = model.predict(val_df[feature_cols])
rmse = np.sqrt(mean_squared_error(val_df[target], val_pred))
print(f"Validation RMSE: {rmse:.4f}")
