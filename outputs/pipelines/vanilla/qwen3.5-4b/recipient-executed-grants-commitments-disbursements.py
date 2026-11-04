import pandas as pd
import numpy as np
import lightgbm as lgb

# Define file paths
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/recipient-executed-grants-commitments-disbursements/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/recipient-executed-grants-commitments-disbursements/data/test.parquet"

def preprocess_df(df):
    """
    Handles missing values and categorical encoding automatically.
    Returns a tuple of (features_dataframe, categorical_columns_used)
    """
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    working_df = df.copy()
    
    # Fill missing values in numeric columns with median
    for col in numeric_cols:
        median_val = working_df[col].median()
        working_df[col] = working_df[col].fillna(median_val)
        
    # Fill missing values in categorical columns with a placeholder
    for col in categorical_cols:
        # Mode is usually a safe default, but we convert to string to ensure consistency
        mode_val = working_df[col].mode().iloc[0] if len(working_df[col].mode()) > 0 else "UNKNOWN"
        working_df[col] = working_df[col].fillna(mode_val)
        working_df[col] = working_df[col].astype(str)

    # One-Hot Encode categorical columns
    # We use the processed df, so NaNs are already filled and converted to strings
    onehot_encoded = pd.get_dummies(working_df, columns=categorical_cols)
    
    # Ensure feature alignment (in case of slight dtypes differences, though unlikely)
    return onehot_encoded, categorical_cols

# Load Data
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

# Define Target Column
TARGET_NAME = "Disbursements USD (Trust funds)"

# Prepare Features and Targets
y_train = train_df[TARGET_NAME]
X_train_raw = train_df.drop(columns=[TARGET_NAME])

y_test = test_df[TARGET_NAME]
X_test_raw = test_df.drop(columns=[TARGET_NAME])

# Preprocess Train and Test separately to ensure identical column structures
X_train_processed, _ = preprocess_df(X_train_raw)
X_test_processed, _ = preprocess_df(X_test_raw)

# Extract feature columns to ensure consistency
train_cols = list(X_train_processed.columns)
test_cols_to_keep = [col for col in X_test_processed.columns if col in train_cols]
X_test_clean = X_test_processed[test_cols_to_keep]

# Instantiate LightGBM Regressor
model = lgb.LGBMRegressor(
    objective='regression', 
    n_estimators=500,
    learning_rate=0.05,
    max_depth=10,
    min_child_samples=20,
    random_state=42,
    verbose=-1,
    n_jobs=-1
)

# Train Model
model.fit(X_train_processed.values, y_train.values)

# Generate Predictions
predictions = model.predict(X_test_clean.values)

# Create Output DataFrame with specified IDs
output_df = pd.DataFrame({
    'id': range(len(predictions)),
    TARGET_NAME: predictions
})

# Save to CSV
output_df.to_csv('predictions.csv', index=False)