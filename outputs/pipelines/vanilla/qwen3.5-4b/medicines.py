import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Paths
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/medicines/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/medicines/data/test.parquet'

# Load Data
print("Loading data...")
df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

# Target column
target_col = 'Medicine status'
# Feature columns (exclude target)
feature_cols = [col for col in df_train.columns if col != target_col]

X_train = df_train[feature_cols].copy()
y_train = df_train[target_col].copy()
X_test = df_test[feature_cols].copy()

# Handle missing values automatically
numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

# Fill missing numeric values with median
for col in numerical_cols:
    X_train[col] = X_train[col].fillna(X_train[col].median())
    X_test[col] = X_test[col].fillna(X_test[col].median())

# Fill missing categorical values with mode
for col in categorical_cols:
    # Ensure mode is computed on original data type or string
    unique_vals = X_train[col].unique()
    if len(unique_vals) == 0:
        continue
    mode_val = X_train[col].mode().iloc[0]
    # If mode is NaN, use the first non-null
    if pd.isna(mode_val):
        mode_val = X_train[col].iloc[0]
    X_train[col] = X_train[col].fillna(mode_val)
    X_test[col] = X_test[col].fillna(mode_val)

# Initialize and Train LightGBM Model
print("Training LightGBM Model...")
# Set parameters to handle missing and categorical data efficiently
params = {'objective': 'binary', 'metric': 'auc', 'verbose': -1}
model = lgb.LGBMClassifier(**params, num_leaves=31, min_data_for_leaf=5, random_state=42)
model.fit(X_train, y_train)

# Predict on Test Data
# Using predict_proba for ROC-AUC friendly outputs (converts to class labels for the 'Medicine status' column)
# Based on requirements, we save predictions. 
# Since the target column is 'Medicine status' (implied categorical/binary), 
# we will output the class label (0 or 1) mapped to that column name.
predictions = model.predict(X_test)

# Create Output DataFrame
output_df = pd.DataFrame({
    'id': range(len(X_test)),
    'Medicine status': predictions.astype(int)
})

# Save to CSV
output_path = 'predictions.csv'
output_df.to_csv(output_path, index=False)

print(f"Saved predictions to {output_path}")