import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/discretionary-grant/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/discretionary-grant/data/test.parquet')

# Define columns to drop (identifiers, addresses, names, and other non-predictive features)
drop_cols = [
    'LogNumber', 'Institution', 'ProjectTitle',
    'InstAddr1', 'InstAddr2', 'InstAddr3', 'InstZip',
    'Prefix', 'FirstName', 'LastName',
    'OrgUnit', 'OUAddr1', 'OUAddr2', 'OUAddr3', 'OUZip',
    'Description'
]

# Keep only columns that exist in both train and test
common_cols = set(train_df.columns) & set(test_df.columns)
drop_cols = [c for c in drop_cols if c in common_cols]

# Separate target
y_train = train_df['AwardTotal'].copy()
X_train_full = train_df.drop('AwardTotal', axis=1)
X_test = test_df.copy()

# Drop non-predictive columns
X_train_full = X_train_full.drop(columns=[c for c in drop_cols if c in X_train_full.columns])
X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])

# Ensure same columns
X_test = X_test.reindex(columns=X_train_full.columns, fill_value=None)

# Identify categorical and numeric columns
cat_cols = X_train_full.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X_train_full.select_dtypes(include=['number']).columns.tolist()

# Encode categorical columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    # Combine train and test to fit encoder (avoid unknown categories)
    combined = pd.concat([X_train_full[col], X_test[col]], axis=0).astype(str)
    le.fit(combined)
    X_train_full[col] = le.transform(X_train_full[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

# Fill missing numeric values with median
for col in num_cols:
    median_val = X_train_full[col].median()
    X_train_full[col] = X_train_full[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

# Create validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train, test_size=0.2, random_state=42
)

# Train LightGBM model
model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)

model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[]
)

# Evaluate on validation set
val_pred = model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Validation RMSE: {val_rmse:.4f}")

# Retrain on full training data
final_model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)
final_model.fit(X_train_full, y_train)

# Make predictions on test set
test_pred = final_model.predict(X_test)

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(test_pred)),
    'AwardTotal': test_pred
})
output_df.to_csv('predictions.csv', index=False)
print(f"Predictions saved. Shape: {output_df.shape}")