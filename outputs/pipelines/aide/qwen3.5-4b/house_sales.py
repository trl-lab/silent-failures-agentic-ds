import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import os

# Set random seed for reproducibility
np.random.seed(42)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify features
target_col = "price"
id_col = "id"
feature_cols = [col for col in train_df.columns if col not in [target_col, id_col]]

# Combine train and test for feature engineering
combined_df = pd.concat([train_df, test_df], ignore_index=True)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import os

# Set random seed for reproducibility
np.random.seed(42)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify features
target_col = "price"
id_col = "id"
feature_cols = [col for col in train_df.columns if col not in [target_col, id_col]]

# Combine train and test for feature engineering
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Create preprocessing pipeline - handle missing values and scale
train_features = combined_df[feature_cols].copy()
test_features = combined_df[feature_cols].copy()

# Fill missing values with median
train_features = train_features.fillna(train_features.median())
test_features = test_features.fillna(train_features.median())

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(
    train_features, train_df[target_col], test_size=0.2, random_state=42, stratify=None
)

# Prepare LightGBM data
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Train LightGBM model with early stopping
params = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Calculate RMSE on validation set
val_predictions = model.predict(X_val)
val_rmse = mean_squared_error(y_val, val_predictions, squared=False)
print(f"Validation RMSE: {val_rmse:.2f}")

# Generate predictions for test set
test_predictions = model.predict(test_features)

# Create submission file
submission = pd.DataFrame({"id": test_df[id_col], "price": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
