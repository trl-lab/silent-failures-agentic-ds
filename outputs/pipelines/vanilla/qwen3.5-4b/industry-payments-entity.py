import pandas as pd
import numpy as np
import lightgbm as lgb

# Define file paths
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/industry-payments-entity/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/industry-payments-entity/data/test.parquet'
OUTPUT_PATH = 'predictions.csv'
TARGET_COLUMN = 'entityPayment'

# Load datasets
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

# Validate target column
assert TARGET_COLUMN in train_df.columns, f"Target column {TARGET_COLUMN} not found."

# Separate features and target
feature_columns = [col for col in train_df.columns if col != TARGET_COLUMN]

X_train = train_df[feature_columns].copy()
X_test = test_df[feature_columns].copy()
y_train = train_df[TARGET_COLUMN].values

# Process Categorical Columns (Object/Dtype String)
# Identify columns to process
categorical_cols = []
numerical_cols = []

for col in feature_columns:
    if pd.api.types.is_object_dtype(X_train[col]) or pd.api.types.is_string_dtype(X_train[col]):
        categorical_cols.append(col)
    else:
        numerical_cols.append(col)

# Label Encoding for Categorical Columns
cat_mappings = {}
for col in categorical_cols:
    # Determine unique valid values from training data
    # Exclude NaNs to create mapping
    valid_values = X_train[col].dropna().unique()
    # Sort to ensure consistent mapping
    sorted_values = np.sort(valid_values)
    mapping = {v: i for i, v in enumerate(sorted_values)}
    cat_mappings[col] = mapping
    # Apply mapping to train and test, filling undefined with -1
    X_train[col] = X_train[col].map(mapping).fillna(-1.0)
    X_test[col] = X_test[col].map(mapping).fillna(-1.0)

# Handle Missing Numeric Values
imputation_dict = {}
for col in numerical_cols:
    # Fill with median for robustness
    med_val = X_train[col].median()
    imputation_dict[col] = med_val
    X_train[col] = X_train[col].fillna(med_val)
    X_test[col] = X_test[col].fillna(med_val)

# Convert to NumPy Float Arrays
X_train_np = X_train.values.astype('float32')
X_test_np = X_test.values.astype('float32')
y_train_np = y_train.astype('float32')

# Initialize LightGBM Model
model = lgb.LGBMRegressor(
    objective='regression',
    n_estimators=500,
    learning_rate=0.1,
    max_depth=-1,
    min_child_samples=20,
    num_leaves=31,
    feature_fraction=1.0,
    bagging_fraction=0.8,
    bagging_freq=0,
    verbosity=-1
)

# Train Model
model.fit(X_train_np, y_train_np)

# Generate Predictions
predictions = model.predict(X_test_np)

# Prepare Output DataFrame
output_ids = list(range(len(test_df)))
results = pd.DataFrame({
    'id': output_ids,
    'entityPayment': predictions
})

# Save Results
results.to_csv(OUTPUT_PATH, index=False)
print(f"Predictions saved to {OUTPUT_PATH}")