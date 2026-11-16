import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Feature columns (exclude target and id)
target = "critical_temp"
feature_cols = [col for col in train_df.columns if col != target]

# Handle missing values - use median for continuous features
for col in feature_cols:
    if train_df[col].isna().any():
        train_df[col].fillna(train_df[col].median(), inplace=True)
        test_df[col].fillna(train_df[col].median(), inplace=True)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_df[feature_cols])
X_test_scaled = scaler.transform(test_df[feature_cols])

# Create validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_train_scaled, train_df[target], test_size=0.2, random_state=42
)

# LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.1,
    num_leaves=31,
    max_depth=10,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)

# Training with early stopping
history = model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
)

# Evaluate on validation set
val_rmse = np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
print(f"Validation RMSE: {val_rmse:.4f}")

# Generate predictions for test data
test_predictions = model.predict(X_test_scaled)

# Save to submission file
submission = pd.DataFrame(
    {"id": range(len(test_df)), "critical_temp": test_predictions}
)
submission.to_csv("./working/submission.csv", index=False)


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Feature columns (exclude target and id)
target = "critical_temp"
feature_cols = [col for col in train_df.columns if col != target]

# Handle missing values - use median for continuous features
for col in feature_cols:
    if train_df[col].isna().any():
        train_df[col].fillna(train_df[col].median(), inplace=True)
        test_df[col].fillna(train_df[col].median(), inplace=True)

# Create validation split
X_train, X_val, y_train, y_val = train_test_split(
    train_df[feature_cols], train_df[target], test_size=0.2, random_state=42
)

# LightGBM model with improved hyperparameters
model = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=12,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)

# Training with early stopping
history = model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
)

# Evaluate on validation set
val_rmse = np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
print(f"Validation RMSE: {val_rmse:.4f}")

# Generate predictions for test data
test_predictions = model.predict(test_df[feature_cols])

# Save to submission file
submission = pd.DataFrame(
    {"id": range(len(test_df)), "critical_temp": test_predictions}
)
submission.to_csv("./working/submission.csv", index=False)

print("Submission saved to ./working/submission.csv")
