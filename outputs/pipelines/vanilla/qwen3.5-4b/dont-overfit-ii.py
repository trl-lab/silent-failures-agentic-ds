import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

# Define file paths as specified
train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/dont-overfit-ii/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/dont-overfit-ii/data/test.parquet'

# Load data
df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

# Identify features (exclude 'id' and 'target')
# We assume 'target' exists in train, and 'id' is an integer identifier column
cols_to_exclude_train = ['id', 'target']
cols_to_exclude_test = ['id', 'target']

# Extract feature names from train
train_col_names = df_train.columns.tolist()
feature_cols = [col for col in train_col_names if col not in cols_to_exclude_train]

# Prepare training data
X_train = df_train[feature_cols]
y_train = df_train['target']

# Prepare test data: drop 'target' (should not be there anyway), select only feature columns
# Ensure test data matches train feature columns
test_col_names = df_test.columns.tolist()
# Remove 'id' and 'target' from consideration for features
valid_test_col_names = [col for col in test_col_names if col in feature_cols]

X_test = df_test[valid_test_col_names]

# Initialize model with parameters optimized for overfitting control
model = CatBoostClassifier(
    iterations=1000,
    depth=5,
    learning_rate=0.1,
    loss_function='Logloss',
    eval_metric='Logloss',
    random_seed=42,
    verbose=0,
    n_estimators=1000,
    max_bin=255,
    l2_leaf_reg=3, # Regularization to prevent overfitting
    bagging_freq=5,
    early_stopping_rounds=100
)

# Fit model
model.fit(X_train, y_train)

# Get predictions (probabilities for the positive class)
probs = model.predict_proba(X_test)[:, 1]

# Prepare output dataframe
output_df = pd.DataFrame({
    'id': list(range(len(df_test))),
    'target': probs
})

# Save predictions
output_df.to_csv('predictions.csv', index=False)