import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import os

# Read data
train = pd.read_parquet('./input/train.parquet')
test = pd.read_parquet('./input/test.parquet')

target = 'Financial Assistance'
y = train[target].values

# Prepare features
drop_cols = [target]
if 'id' in train.columns:
    drop_cols.append('id')
X = train.drop(columns=drop_cols)

test_drop_cols = []
if 'id' in test.columns:
    test_drop_cols.append('id')
test_X = test.drop(columns=test_drop_cols)

# Align columns
for col in X.columns:
    if col not in test_X.columns:
        test_X[col] = 0
test_X = test_X[X.columns]

# Convert datetime columns to numeric seconds since epoch (NaT -> NaN)
datetime_cols = [col for col in X.columns if pd.api.types.is_datetime64_any_dtype(X[col])]
for col in datetime_cols:
    X[col] = pd.to_datetime(X[col], errors='coerce').astype('int64') / 1e9
    test_X[col] = pd.to_datetime(test_X[col], errors='coerce').astype('int64') / 1e9
    na_sentinel = np.iinfo(np.int64).min / 1e9
    X[col] = X[col].replace(na_sentinel, np.nan)
    test_X[col] = test_X[col].replace(na_sentinel, np.nan)

# Encode categoricals
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    combined = pd.concat([X[col], test_X[col]], axis=0).astype(str)
    categories = combined.unique().tolist()
    mapping = {c: i for i, c in enumerate(categories)}
    X[col] = X[col].astype(str).map(mapping).fillna(-1).astype(int)
    test_X[col] = test_X[col].astype(str).map(mapping).fillna(-1).astype(int)

# Split FIRST, then fit (no leakage)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Fill NaNs (fit on train only)
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    median_val = X_train[col].median()
    X_train[col] = X_train[col].fillna(median_val)
    X_val[col] = X_val[col].fillna(median_val)
    test_X[col] = test_X[col].fillna(median_val)

# Train model
model = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    verbose=-1
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
)

# Validate with official metric (RMSE)
val_pred = model.predict(X_val)
rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
print(f"Final Validation Score: {rmse:.6f}")

# Test prediction
test_pred = model.predict(test_X)

# Build submission
submission = pd.DataFrame({
    'id': np.arange(len(test)),
    'Financial Assistance': test_pred
})
os.makedirs('./submission', exist_ok=True)
submission.to_csv('./submission/submission.csv', index=False)
print(f"Submission saved with {len(submission)} rows.")