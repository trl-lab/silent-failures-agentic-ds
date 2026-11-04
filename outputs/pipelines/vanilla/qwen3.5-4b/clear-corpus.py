import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Attempt to import CatBoost, it is pre-installed per requirements
try:
    import catboost as cb
except ImportError:
    import xgboost as xgb
    cb = xgb

# Configuration
train_path = "/path/to/user/workspace/ds_failures/datasets/strable/clear-corpus/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/clear-corpus/data/test.parquet"
output_path = "predictions.csv"
target_col = "BT Easiness"
random_state = 42

# Suppress verbose output for clean execution
cb.verbose = False

# Load datasets
df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

# Verify target column exists in training data
if target_col not in df_train.columns:
    raise ValueError(f"Target column '{target_col}' not found in training data.")

# Identify feature columns (exclude target)
feature_cols = [col for col in df_train.columns if col != target_col]

# Prepare training and testing features
X_train = df_train[feature_cols].copy()
X_test = df_test[feature_cols].copy()

# Extract target variable
y_train = df_train[target_col]

# Initialize model
# CatBoost automatically handles missing values and categorical columns
# Using objective='rmse' to align with evaluation metric
model = cb.CatBoostRegressor(
    objective="rmse",
    random_state=random_state,
    verbose=0,
    n_estimators=1000,
    learning_rate=0.1,
    thread_count=-1  # -1 uses all available threads
)

# Fit the model
model.fit(X_train, y_train)

# Generate predictions
predictions = model.predict(X_test)

# Create output dataframe
output_data = pd.DataFrame({
    "id": range(len(df_test)),
    "BT Easiness": predictions
})

# Save predictions
output_data.to_csv(output_path, index=False)