import pandas as pd
import numpy as np
import catboost as cb

# Configuration
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/california-houses/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/california-houses/data/test.parquet'
TARGET_COL = 'Total interior livable area'
OUTPUT_FILE = 'predictions.csv'

# Load Data
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

# Ensure columns exist and identify feature columns
# Select all columns except the target column
cols_to_keep = [col for col in train_df.columns if col != TARGET_COL]

# Convert target to series and features to array
y_train = train_df[TARGET_COL].values
X_train = train_df[cols_to_keep].values

# Prepare Test Data
X_test = test_df[cols_to_keep].values
y_pred = model.predict(X_test)
# Note: I cannot predict before defining model, so restructure

# Initialize Model
model = cb.CatBoostRegressor(
    iterations=200, 
    learning_rate=0.01, 
    depth=6, 
    logging_level='Silent',
    loss_function='RMSE'
)

# Train
model.fit(X_train, y_train)

# Predict
pred_array = model.predict(X_test)

# Create Output DataFrame
ids = np.arange(len(X_test)).astype(int)
result_df = pd.DataFrame({
    'id': ids,
    f'Total interior livable area': pred_array
})

# Save
result_df.to_csv(OUTPUT_FILE, index=False)