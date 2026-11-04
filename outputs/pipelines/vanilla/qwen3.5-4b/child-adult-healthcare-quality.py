import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import sys

# Define file paths as requested
TRAIN_FILE = '/path/to/user/workspace/ds_failures/datasets/strable/child-adult-healthcare-quality/data/train.parquet'
TEST_FILE = '/path/to/user/workspace/ds_failures/datasets/strable/child-adult-healthcare-quality/data/test.parquet'

# Target variable name
TARGET_COL = 'State Rate'

# Load data
df_train = pd.read_parquet(TRAIN_FILE)
df_test = pd.read_parquet(TEST_FILE)

# Ensure target column exists
assert TARGET_COL in df_train.columns, f"Target column '{TARGET_COL}' not found in training data."

# Extract features and target
y_train = df_train[TARGET_COL].values
X_train = df_train.drop(columns=[TARGET_COL])

# Feature matrix for testing
X_test = df_test.copy()
X_test_columns = X_test.columns

# Handle missing values and categorical columns automatically
# CatBoost handles missing values and categorical types effectively without complex pipelines.
# We identify object-type columns to pass to cat_features for optimal performance
categorical_cols = []
for col in X_train.columns:
    if isinstance(X_train[col].dtypes, pd.Series):
        if X_train[col].dtype == 'object':
            categorical_cols.append(col)
        elif len(X_train[col].unique()) > 10:
            # High cardinality categorical treated as numeric might work differently
            # For safety, keep object columns in cat_features
            pass
    else:
        if pd.api.types.is_object_dtype(X_train[col]):
            categorical_cols.append(col)

# Configure model parameters
params = {
    'objective': 'Regression',
    'task_name': 'Regression',
    'eval_metric': 'RMSE',
    'iteration_count': 500,
    'learning_rate': 0.05,
    'loss_function': 'MAE', # Switching to MAE for stability or keeping MSE/RMSE? Default is LogLoss or RMSE
    'random_seed': 42,
    'verbose': 0,
}
params.update({'metric': 'RMSE'})

# Prepare model arguments
cat_params = {'cat_features': categorical_cols}

# Instantiate model
try:
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        objective='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        verbose=0,
        **cat_params
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Generate predictions
    predictions = model.predict(X_test)
    
except Exception as e:
    # Fallback or error handling if CatBoost fails
    print(f"Warning: An unexpected error occurred: {e}", file=sys.stderr)
    predictions = np.nan * np.ones(len(X_test))

# Prepare output dataframe
output_df = pd.DataFrame({
    'id': range(len(X_test)),
    'State Rate': predictions
})

# Save predictions
output_df.to_csv('predictions.csv', index=False)

print("Completed. Predictions saved to predictions.csv")