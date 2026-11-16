import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# Set random seed for reproducibility
np.random.seed(42)

# Load data
train_data = pd.read_parquet("./input/train.parquet")
test_data = pd.read_parquet("./input/test.parquet")

# Check column names in sample submission if available
sample_submission = None
if os.path.exists("./input/sample_submission.csv"):
    sample_submission = pd.read_csv("./input/sample_submission.csv")
    print(f"Sample submission columns: {sample_submission.columns.tolist()}")

# If no sample submission, use standard format based on task description
if sample_submission is None:
    n_samples = len(test_data)
    sample_submission = pd.DataFrame({"id": range(n_samples), "WATERDEPTH": np.zeros(n_samples)})

# Identify features and target
target_col = "WATERDEPTH"

# Convert object columns to numeric types for LightGBM
numeric_columns = []
for col in train_data.columns:
    if col != target_col:
        try:
            # Try to convert to numeric
            pd.to_numeric(train_data[col], errors='raise')
            numeric_columns.append(col)
        except (ValueError, TypeError):
            # Skip object columns that can't be converted
            pass

feature_cols = numeric_columns

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(
    train_data[feature_cols], train_data[target_col], test_size=0.2, random_state=42
)

# Create LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

# Train model with early stopping
callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)

# Get validation predictions and calculate RMSE
y_val_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"Final Validation Score: {rmse:.6f}")

# Generate test predictions
test_predictions = model.predict(test_data[feature_cols])

# Create submission dataframe
submission = pd.DataFrame({"id": test_data.index, "WATERDEPTH": test_predictions})

# Save submission
submission.to_csv("./submission/submission.csv", index=False)
print("Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission.shape}")
print(f"First 5 rows:\n{submission.head()}")
