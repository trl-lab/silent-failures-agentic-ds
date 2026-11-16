import pandas as pd
import numpy as np
import catboost as ctb

# Input paths
train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/playground-series-s4e2/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/playground-series-s4e2/data/test.parquet'

# Load data
df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

# Configuration
target_col = 'NObeyesdad'
exclude_cols = {target_col, 'id'}

# Determine feature columns present in training set
feature_cols = [col for col in df_train.columns if col not in exclude_cols]

# Prepare training data
X_train = df_train[feature_cols].copy()
y_train = df_train[target_col]

# Prepare test data (ensure columns match)
X_test = df_test[feature_cols].copy()

# Train model using CatBoost (handles missing and categorical automatically)
model = ctb.CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    verbose=False,
    random_state=42,
    loss_function='MultiLogloss',
    metrics='Accuracy'
)

model.fit(X_train, y_train, eval_set=None, verbose=False)

# Generate predictions
predictions = model.predict(X_test)

# Create output dataframe with specific requirements:
# 1. 'id' column (integer row index starting from 0)
# 2. 'NObeyesdad' column
output_df = pd.DataFrame({
    'id': range(len(predictions)),
    'NObeyesdad': predictions
})

# Save to CSV
output_df.to_csv('predictions.csv', index=False)