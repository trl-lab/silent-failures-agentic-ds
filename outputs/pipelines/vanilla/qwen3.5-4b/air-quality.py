import pandas as pd
import numpy as np
import lightgbm as lgb

# Define file paths
train_path = "/path/to/user/workspace/ds_failures/datasets/uci/air-quality/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/uci/air-quality/data/test.parquet"

# Load data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Target variable
target_col = "CO(GT)"

# Extract feature columns from training set (excluding target)
all_cols = train_df.columns.tolist()
feature_cols = [col for col in all_cols if col != target_col]

# Ensure test set has the same features (excluding target)
test_feature_cols = [col for col in feature_cols if col in test_df.columns]

# Prepare X and y
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col]
X_test = test_df[test_feature_cols].copy()

# Handle missing values automatically
# Fill numeric missing values with median. 
# String/categorical values are left as is for LGBM to handle them natively.
for col in feature_cols:
    if col not in X_test.columns:
        continue
        
    # Check if numeric dtype to perform imputation
    if pd.api.types.is_numeric_dtype(X_train[col]):
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)
    # Non-numeric columns (like sensor names) are passed as-is to LightGBM

# Initialize LightGBM Regressor
model = lgb.LGBMRegressor(
    objective='regression',
    metric='rmse',
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=1000,
    subsample=1.0,
    colsample_bytree=1.0,
    random_state=42,
    verbose=-1
)

# Train the model
model.fit(X_train, y_train)

# Generate predictions on test set
predictions = model.predict(X_test)

# Create output DataFrame
output_df = pd.DataFrame({
    'id': range(len(predictions)),
    'CO(GT)': predictions
})

# Save to current working directory
output_df.to_csv('predictions.csv', index=False)