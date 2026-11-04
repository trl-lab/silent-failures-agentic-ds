import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

try:
    import catboost as cb
    USE_CATBOOST = True
except ImportError:
    USE_CATBOOST = False
    # Fallback logic if CatBoost not found (though prompt says installed)
    try:
        import xgboost as xgb
        USE_XGB = True
    except ImportError:
        try:
            import lightgbm as lgb
            USE_LGBM = True
        except ImportError:
            raise Exception("Required libraries not found. Please use catboost, xgboost, or lightgbm.")

if not USE_CATBOOST and not USE_XGB and not USE_LGBM:
    # Should not happen based on prompt constraints
    sys.exit(1)

# Paths defined in task
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/commitments-in-trust-funds/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/commitments-in-trust-funds/data/test.parquet"

# Load Data
print("Loading data...")
df_train = pd.read_parquet(TRAIN_PATH)
df_test = pd.read_parquet(TEST_PATH)

target_col = "New Commitments (US$)"

# Verify target column exists
if target_col not in df_train.columns:
    raise ValueError(f"Target column '{target_col}' not found in train data. Available columns: {list(df_train.columns)}")

# Prepare features
feature_cols = [col for col in df_train.columns if col != target_col]
if len(feature_cols) == 0:
    raise ValueError("No feature columns found after removing target.")

X_train = df_train[feature_cols].copy()
y_train = df_train[target_col].values

X_test = df_test[feature_cols].copy()

# Create Model
model_type = None
params = {}

if USE_CATBOOST:
    from catboost import CatBoostRegressor
    model_type = CatBoostRegressor(verbose=False, iterations=500, max_depth=6, learning_rate=0.1, eval_metric='RMSE', loss_function='RMSE')
    # Enable automatic handling of missing values and categories
elif USE_XGB:
    from xgboost import XGBRegressor
    model_type = XGBRegressor(random_state=42, objective='reg:squarederror', eval_metric='rmse')
elif USE_LGBM:
    from lightgbm import LGBMRegressor
    model_type = LGBMRegressor(random_state=42, objective='regression', metric='mse', verbosity=-1)

if model_type is None:
    raise ValueError("Could not determine valid model instance.")

# Train Model
print("Training model...")
model = model_type.fit(X_train, y_train)

# Make Predictions
print("Generating predictions...")
y_pred = model.predict(X_test)

# Convert predictions to DataFrame with ID
output_df = pd.DataFrame({
    'id': range(len(y_pred)),
    'New Commitments (US$)': y_pred
})

# Save Output
OUTPUT_FILE = "predictions.csv"
output_df.to_csv(OUTPUT_FILE, index=False)
print(f"Predictions saved to {OUTPUT_FILE}")

# Optional: Calculate local RMSE for sanity check on train split if needed, 
# but submission file is the main deliverable.
# train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
# print(f"Internal RMSE: {train_rmse:.4f}")