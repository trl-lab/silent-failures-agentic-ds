import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify target column
target_col = "Sale_Price"

# Prepare features
train_features = train_df.drop(columns=[target_col])
test_features = test_df

# Split training data for validation (80/20)
train_size = int(len(train_features) * 0.8)
train_X = train_features.iloc[:train_size]
train_y = train_df.iloc[:train_size][target_col]
val_X = train_features.iloc[train_size:]
val_y = train_df.iloc[train_size:][target_col]

# Convert data to LightGBM Dataset
train_dataset = lgb.Dataset(train_X, label=train_y)
val_dataset = lgb.Dataset(val_X, label=val_y, reference=train_dataset)

# Define parameters
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
    "seed": 42,
}

# Train model with callbacks
model = lgb.train(
    params,
    train_dataset,
    num_boost_round=1000,
    valid_sets=[val_dataset],
    valid_names=["val"],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Evaluate on hold-out validation set
val_pred = model.predict(val_X)
val_rmse = np.sqrt(mean_squared_error(val_y, val_pred))
print(f"Validation RMSE: {val_rmse:.4f}")

# Predict on test set
test_pred = model.predict(test_features)

# Prepare submission
submission = pd.DataFrame({"id": test_features.index, "Sale_Price": test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
