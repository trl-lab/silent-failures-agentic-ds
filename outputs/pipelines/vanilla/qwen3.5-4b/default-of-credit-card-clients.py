import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# Define file paths
train_path = '/path/to/user/workspace/ds_failures/datasets/uci/default-of-credit-card-clients/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/default-of-credit-card-clients/data/test.parquet'
output_file = 'predictions.csv'

# Load data
df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

# Define target column name as specified
target_name = 'default payment next month'

# Find target column
if target_name not in df_train.columns:
    # Check for common variations based on UCI dataset schema
    possible_targets = ['Y', 'y', 'target', 'label']
    for col in possible_targets:
        if col in df_train.columns:
            target_name = col
            break
    # If still not found, take the last column (often target in these old CSV exports) or search for 'payment' pattern
    if target_name == 'Y':
        pass 
else:
    pass

print(f"Using target column: {target_name}")

# Prepare feature names
columns_to_use = df_train.columns.tolist()
# Exclude target column
features = [col for col in columns_to_use if col != target_name]
feature_columns = list(df_train[features])
feature_names = list(df_test[features]) # Ensure we get feature names

# Separate X and y
X_train = df_train[features]
y_train = df_train[target_name]
X_test = df_test[features]
y_test = df_test[target_name] # Not needed for prediction file but useful for verification if requested

# Initialize LightGBM Classifier
model = lgb.LGBMClassifier(random_state=42, max_depth=-1, num_leaves=31, subsample=0.8, colsample_bytree=0.8, n_estimators=500, learning_rate=0.1, verbose=-1, enable_categorical=False)

# Fit the model
model.fit(X_train, y_train, early_stopping_rounds=50)

# Predict on test set
# LightGBM default predict returns probabilities
# We can convert threshold to 0.5 based on ROC AUC context
pred_proba = model.predict_proba(X_test)[:, 1]
threshold = 0.5
pred_class = (pred_proba >= threshold).astype(int)

# Assign IDs (row index starting from 0)
results = pd.DataFrame({
    'id': range(len(pred_class)),
    'default payment next month': pred_class.values
})

# Save to CSV
results.to_csv(output_file, index=False)

print(f"Predictions saved to {output_file}")