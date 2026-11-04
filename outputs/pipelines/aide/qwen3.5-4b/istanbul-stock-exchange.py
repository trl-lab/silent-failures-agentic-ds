import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify columns
target = "ISE"
feature_cols = ["SP", "DAX", "FTSE", "NIKKEI", "BOVESPA", "EU", "EM"]

# Check for ISE in features - if present, exclude it since it's target
if "ISE" in feature_cols and "ISE" in train_df.columns:
    # Keep only non-ISE features for prediction
    feature_cols = [c for c in feature_cols if c != "ISE"]

# Combine features with target
train_features = train_df[feature_cols].copy()
train_target = train_df[target].copy()
test_features = test_df[feature_cols].copy()

# Handle missing values
train_features = train_features.fillna(train_features.mean())
test_features = test_features.fillna(test_features.mean())

# Create temporal train/validation split to avoid data leakage
train_split = len(train_features) // 2
X_train = train_features.iloc[:train_split]
y_train = train_target.iloc[:train_split]
X_val = train_features.iloc[train_split:]
y_val = train_target.iloc[train_split:]

# Define LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    num_leaves=31,
    random_state=42,
    verbose=-1,
)

# Train model with early stopping
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)

# Evaluate on validation set
y_val_pred = model.predict(X_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"Validation RMSE: {rmse_val:.4f}")

# Generate predictions for test set
test_predictions = model.predict(test_features)

# Create submission file
submission = pd.DataFrame({"id": range(len(test_predictions)), "ISE": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
