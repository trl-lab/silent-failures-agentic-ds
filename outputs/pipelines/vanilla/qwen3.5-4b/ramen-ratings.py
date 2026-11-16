import pandas as pd
import numpy as np
import catboost as catb

# Paths
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/ramen-ratings/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/ramen-ratings/data/test.parquet'

# Load data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Target
target_col = 'Stars'

# Features (exclude target)
feature_cols = [col for col in train_df.columns if col != target_col]
X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]

# Ensure feature dtypes are consistent
# CatBoost handles numeric and categorical automatically, but we set random state for reproducibility
np.random.seed(42)

# Model initialization
# m-logloss is appropriate for multi-class classification (accuracy metric)
params = {
    'iter': 500,
    'depth': 6,
    'learning_rate': 0.1,
    'loss_function': 'mlogloss',
    'random_seed': 42,
    'thread_count': 1
}

# CatBoost handles missing values and categoricals automatically when passed DataFrame
model = catb.CatBoostClassifier(**params)

# Fit
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Prepare output DataFrame
results_df = pd.DataFrame({
    'id': range(len(X_test)),
    'Stars': y_pred.astype(int)
})

# Save
results_df.to_csv('predictions.csv', index=False)