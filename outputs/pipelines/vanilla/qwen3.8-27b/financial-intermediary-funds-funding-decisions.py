import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/financial-intermediary-funds-funding-decisions/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/financial-intermediary-funds-funding-decisions/data/test.parquet')

# Separate features and target
target = 'Amount in USD'
X_train = train_df.drop(target, axis=1).copy()
y_train = train_df[target].copy()
X_test = test_df.copy()

# Identify categorical and numeric columns
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()

# Handle missing values and encode categorical columns
for col in categorical_cols:
    X_train[col] = X_train[col].fillna('Unknown').astype(str)
    X_test[col] = X_test[col].fillna('Unknown').astype(str)
    le = LabelEncoder()
    combined = pd.concat([X_train[col], X_test[col]], axis=0)
    le.fit(combined)
    X_train[col] = le.transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

for col in numeric_cols:
    median_val = X_train[col].median()
    X_train[col] = X_train[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

# Create validation split
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define LightGBM model
lgbm_model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)

# Train LightGBM
lgbm_model.fit(X_tr, y_tr)

# Evaluate
val_pred = lgbm_model.predict(X_val)
val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
print(f"LightGBM Validation RMSE: {val_rmse:.6f}")

# Train on full training data
lgbm_final = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)
lgbm_final.fit(X_train, y_train)

# Predict on test
test_pred = lgbm_final.predict(X_test)

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(test_pred)),
    'Amount in USD': test_pred
})
output_df.to_csv('predictions.csv', index=False)