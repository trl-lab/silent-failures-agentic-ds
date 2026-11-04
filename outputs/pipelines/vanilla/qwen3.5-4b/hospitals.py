import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import lightgbm as lgb

# Define file paths
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/hospitals/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/hospitals/data/test.parquet'
TARGET_COLUMN = 'BEDS'
OUTPUT_FILE = 'predictions.csv'

# Load data
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

# Separate features and target
y_train = train_df[TARGET_COLUMN]
X_train = train_df.drop(columns=[TARGET_COLUMN])
y_test = test_df.drop(columns=[TARGET_COLUMN]) if TARGET_COLUMN in test_df.columns else test_df.reset_index(drop=True)

# Function to determine feature types
def separate_features(df):
    numeric_cols = []
    categorical_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        elif pd.api.types.is_object_dtype(df[col]):
            categorical_cols.append(col)
    return numeric_cols, categorical_cols

# Identify features common between train and test to ensure consistency
if not hasattr(X_train, 'columns'):
    raise ValueError("X_train columns not found")

common_cols = set(X_train.columns).intersection(set(y_test.columns))
if len(common_cols) == 0:
    # Fallback if schema differs significantly, though unlikely
    raise ValueError("No common features found between train and test")

X_train_clean = X_train[[col for col in X_train.columns if col in common_cols]]
y_train_series = train_df.loc[train_df.index, [col for col in common_cols if col == TARGET_COLUMN]][TARGET_COLUMN] # Just safety check
y_train_series = train_df.loc[train_df.index, TARGET_COLUMN]

X_test_clean = y_test[[col for col in X_test.columns if col in common_cols]] # This variable naming is confusing, fix below

# Reset variables for clarity based on previous scope
all_common_cols = [col for col in X_train.columns if col in test_df.columns]

if TARGET_COLUMN in test_df.columns:
    X_test = test_df[all_common_cols].copy()
else:
    # If target column doesn't exist in test (unusual), proceed without it for prediction setup
    X_test = test_df[all_common_cols]

num_cols = [c for c in X_train[all_common_cols].columns if pd.api.types.is_numeric_dtype(X_train[all_common_cols][c])]
cat_cols = [c for c in X_train[all_common_cols].columns if pd.api.types.is_object_dtype(X_train[all_common_cols][c])]

# Create Preprocessor Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
    ],
    remainder='passthrough'
)

# Wrap in full pipeline (Optional but cleaner)
# Since we want to handle missing values, let's apply preprocessor steps manually or via pipeline.
# Using a custom helper to ensure consistent fitting.

# Step 1: Impute numeric missing values with median
X_train['temp_numeric'] = pd.concat([X_train[c].fillna(X_train[c].median()) for c in num_cols], axis=1)
X_test['temp_numeric'] = pd.concat([X_test[c].fillna(X_test[c].median()) for c in num_cols], axis=1)

# Step 2: Impute categorical missing values with constant (since OneHotEncoder handles sparse/nans via strategy)
# OneHotEncoder defaults to ignoring unknown, but NaNs in object columns result in None values.
# Safe fill for object columns is mode or most frequent string, then encode.
X_train['temp_cat'] = pd.concat([X_train[c].fillna(X_train[c].mode()[0]) if X_train[c].isnull().any() else X_train[c] for c in cat_cols], axis=1)
X_test['temp_cat'] = pd.concat([X_test[c].fillna(X_test[c].mode()[0]) if X_test[c].isnull().any() else X_test[c] for c in cat_cols], axis=1)

# Step 3: Drop duplicates if filled incorrectly and prepare encoding matrix
# We can use OneHotEncoder here but simpler is to concat all processed columns and encode.
# Let's use LightGBM directly for speed and compatibility with object strings if possible.
# However, for generalization and robustness with Sklearn tools provided:

combined_X_train = pd.concat([X_train[num_cols], X_train[cat_cols]], ignore_index=True)
combined_X_test = pd.concat([X_test[num_cols], X_test[cat_cols]], ignore_index=True)

# Re-encode categorical to ensure uniformity across train/test splits if possible
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(categories=X_train[cat_cols], sparse_output=False, handle_unknown='ignore', force_all_finite=True)
encoded_train = encoder.fit_transform(X_train[cat_cols])
encoded_test = encoder.transform(X_test[cat_cols])

# Combine encoded numeric (already medians applied) and encoded categorical
X_final_train = pd.concat([
    pd.DataFrame(combined_X_train[num_cols].values, columns=num_cols), 
    pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(cat_cols))
])

X_final_test = pd.concat([
    pd.DataFrame(combined_X_test[num_cols].values, columns=num_cols), 
    pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(cat_cols))
])

# Train LightGBM
clf = lgb.LGBMRegressor(
    objective='regression',
    learning_rate=0.05,
    num_leaves=31,
    max_depth=8,
    n_estimators=1000,
    random_state=42,
    verbose=-1,
    silent=True
)
clf.fit(X_final_train.to_numpy(), y_train.values)

# Predictions
predictions = clf.predict(X_final_test.to_numpy()).reshape(-1,)

# Output file preparation
output_columns = [['id', 'BEDS']]
result_data = {'id': list(range(len(predictions))), 'BEDS': predictions}
df_predictions = pd.DataFrame(result_data)

# Save
df_predictions.to_csv(OUTPUT_FILE, index=False)