

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# Load data
train_df = pd.read_parquet('./input/train.parquet')
test_df = pd.read_parquet('./input/test.parquet')

# Define features and target
target_col = 'CAPACITY'
# Only use numeric columns for modeling (exclude strings/objects)
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [col for col in numeric_cols if col != target_col]

X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]

# Handle test_ids - check if 'id' column exists, otherwise use row indices
if 'id' in test_df.columns:
    test_ids = test_df['id']
else:
    test_ids = range(len(test_df))

# Create validation split
X_train_full, X_val, y_train_full, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# LightGBM model setup
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

train_data = lgb.Dataset(X_train_full, label=y_train_full)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Train with callbacks
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)]
)

# Predictions
val_pred = model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f'Final Validation Score: {val_rmse:.6f}')

# Test predictions
test_pred = model.predict(X_test)
submission = pd.DataFrame({
    'id': test_ids,
    'CAPACITY': test_pred
})
submission.to_csv('./submission/submission.csv', index=False)