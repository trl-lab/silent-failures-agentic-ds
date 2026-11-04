import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

# Define file paths
train_path = "/path/to/user/workspace/ds_failures/datasets/strable/drug-shortages/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/drug-shortages/data/test.parquet"

# Load data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Check target column exists
target_col = "availability"
assert target_col in train_df.columns, f"Target column {target_col} not found in train data"

# Prepare features
X_train = train_df.drop(columns=[target_col])
X_test = test_df

# Initialize and train model
# CatBoost handles missing values and categorical features automatically
model = CatBoostClassifier(iterations=1000, verbose=False, random_state=42)
model.fit(X_train, train_df[target_col], eval_set=(X_train, train_df[target_col]))

# Make predictions
# Reset index of test dataframe to align 'id' with sequential integers 0, 1, ...
test_df_reset = test_df.copy()
test_df_reset.index = range(len(test_df_reset))

y_pred = model.predict(X_test)

# Create output dataframe
results = pd.DataFrame({
    "id": range(len(y_pred)),
    "availability": y_pred.astype(str) # Ensure consistent type for CSV
})

# Save predictions
results.to_csv("predictions.csv", index=False)