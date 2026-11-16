import pandas as pd
import numpy as np
import catboost as cb
import os

# Set random seed for reproducibility
np.random.seed(42)

# Define file paths
train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/optiver-trading-at-the-close/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/optiver-trading-at-the-close/data/test.parquet'
output_path = 'predictions.csv'

# Load datasets
print("Loading data...")
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Identify feature columns: exclude 'target' and any column named 'id'
feature_cols = [col for col in train_df.columns if col not in ['target', 'id']]

# Ensure columns exist in test set as well (excluding target and id)
# We prepare test features using the same feature columns identified from training
test_cols_to_use = [col for col in test_df.columns if col in feature_cols]

# Prepare feature dataframes
X_train = train_df[feature_cols].copy()
X_test = test_df[test_cols_to_use].copy()
y_train = train_df['target']

# Identify categorical columns for CatBoost
cat_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
# Ensure numerical columns are converted to float if needed
numeric_cols = X_train.select_dtypes(exclude=['object', 'category']).columns.tolist()

# Combine feature list
all_features = list(set(feature_cols + numeric_cols)).union(cat_columns)
# Ensure unique column names
feature_names = list(set(all_features).difference(['id'])) 

# Fill missing values with mode or NaN if CatBoost handles it (CatBoost handles NaNs)
# Here we simply proceed as CatBoost handles NA by default
X_train_processed = X_train.fillna(train_df.mean(numeric_only=True).reindex(feature_names, fill_value=np.nan).round(2))
X_test_processed = X_test.fillna(test_df.mean(numeric_only=True).reindex(feature_names, fill_value=np.nan).round(2))

# Prepare CatBoost Pool
# Convert object dtypes to category to ensure uniform handling
for col in cat_columns:
    if str(X_train[col].dtype) != 'object':
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

train_pool = cb.Pool(X_train_processed, y_train, cat_features=cat_columns, thread_count=4)

# Hyperparameters for CatBoost Regressor
params = {
    'learning_rate': 0.05,
    'loss_function': 'RMSE',
    'max_iter': 200,
    'random_strength': 0.0,
    'random_seed': 42,
    'verbose': 0
}

# Train model
print("Training model...")
model = cb.CatBoostRegressor(**params)
model.fit(train_pool)

# Predict on test set
y_pred = model.predict(cb.Pool(X_test_processed, None, cat_features=cat_columns, thread_count=4))

# Create output dataframe with id (row index starting from 0) and target
prediction_df = pd.DataFrame({
    'id': range(len(y_pred)),
    'target': y_pred
})

# Ensure columns are in correct order
prediction_df = prediction_df[['id', 'target']]

# Save predictions
print(f"Saving predictions to {output_path}...")
prediction_df.to_csv(output_path, index=False)
print("Done.")