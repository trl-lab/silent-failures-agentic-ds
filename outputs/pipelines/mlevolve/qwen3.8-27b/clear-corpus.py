import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# Paths
INPUT_DIR = Path("./input")
SUBMISSION_DIR = Path("./submission")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# Load data
train = pd.read_parquet(INPUT_DIR / "train.parquet")
test = pd.read_parquet(INPUT_DIR / "test.parquet")

TARGET = "BT Easiness"

# Separate target
y = train[TARGET].values
X = train.drop(columns=[TARGET])

# Exclude id column if present
if 'id' in X.columns:
    X = X.drop(columns=['id'])

feature_cols = X.columns.tolist()

# Convert object/string columns to category dtype for LightGBM native handling
cat_cols = [c for c in X.columns if X[c].dtype == 'object' or str(X[c].dtype) == 'string']
for col in cat_cols:
    X[col] = X[col].astype('category')

# Split FIRST to avoid any data leakage
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

# Prepare test features with same categorical treatment
X_test = test[feature_cols].copy()
for col in cat_cols:
    if col in X_test.columns:
        X_test[col] = X_test[col].astype('category')

# Train LightGBM (handles NaN natively, no pre-imputation needed)
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': -1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'n_jobs': -1,
}

model = lgb.LGBMRegressor(n_estimators=5000, **params)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    categorical_feature=cat_cols,
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
)

# Validation score using official metric (RMSE)
val_preds = model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"Final Validation Score (RMSE): {val_rmse:.6f}")

# Generate test predictions from trained model
test_preds = model.predict(X_test)

# Create and save submission
submission = pd.DataFrame({
    'id': range(len(test)),
    'BT Easiness': test_preds
})
submission.to_csv(SUBMISSION_DIR / "submission.csv", index=False)
print(f"Submission saved with {len(submission)} rows.")