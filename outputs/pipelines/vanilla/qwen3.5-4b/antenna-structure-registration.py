import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# File paths
TRAIN_FILE = '/path/to/user/workspace/ds_failures/datasets/strable/antenna-structure-registration/data/train.parquet'
TEST_FILE = '/path/to/user/workspace/ds_failures/datasets/strable/antenna-structure-registration/data/test.parquet'

# Load Data
train_df = pd.read_parquet(TRAIN_FILE)
test_df = pd.read_parquet(TEST_FILE)

# Target Column
TARGET_COL = 'Strucht'

# Separate features and target
X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]

# Prepare Test Data (ensure columns match training set)
test_columns = set(test_df.columns)
train_columns = set(X_train.columns)
valid_test_columns = test_columns.intersection(train_columns)

X_test = test_df[list(valid_test_columns)]

# Automatic Handling of Categorical and Missing Values
# Identify columns where dtype is 'object' (common representation of strings/categoricals)
categorical_cols = [col for col in X_train.columns if X_train[col].dtype == object]

# Remaining columns are assumed numeric for filling
numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

fill_map = {}

for col in numeric_cols:
    try:
        val = X_train[col].median()
        # Safety check for NaN medians (all missing in training column)
        if np.isnan(val):
            val = 0.0
        fill_map[col] = val
        
        # Apply fill to training and testing data using the same statistic
        X_train[col] = X_train[col].fillna(val)
        X_test[col] = X_test[col].fillna(val)
    except Exception:
        # Skip problematic columns
        pass

# Initialize CatBoost Regressor
model = CatBoostRegressor(
    depth=6,
    learning_rate=0.1,
    iterations=1000,
    loss_function='RMSE',
    verbose=0,
    random_state=42,
    cat_features=categorical_cols
)

# Fit Model
model.fit(X_train, y_train)

# Predict on Test Data
test_predictions = model.predict(X_test)

# Prepare Output Data
# Requirement: columns "id" (integer row index starting from 0) and "Strucht"
submission_data = pd.DataFrame({
    'id': range(len(test_predictions)),
    'Strucht': test_predictions
})

# Save Predictions
submission_data.to_csv('predictions.csv', index=False)