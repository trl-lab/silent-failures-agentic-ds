import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/oil-natural-gas-platform/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/oil-natural-gas-platform/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'WATERDEPTH'
y = train_df[target_col].values

# Combine train and test for consistent preprocessing
n_train = len(train_df)
df = pd.concat([train_df.drop(columns=[target_col]), test_df], axis=0).reset_index(drop=True)

# Drop non-predictive unique identifiers if present
if 'GlobalID' in df.columns:
    df.drop(columns=['GlobalID'], inplace=True)

# Process date columns
date_cols = ['SOURCEDATE', 'INSTALL', 'REMOVAL']
for col in date_cols:
    if col in df.columns:
        dt = pd.to_datetime(df[col], errors='coerce')
        df[f'{col}_YEAR'] = dt.dt.year
        df[f'{col}_MONTH'] = dt.dt.month
        df[f'{col}_DAY'] = dt.dt.day
        # Convert timestamp to numeric seconds (handling NaT)
        ts = dt.astype('int64') // 10**9
        df[f'{col}_TIMESTAMP'] = np.where(dt.isna(), np.nan, ts)
        df.drop(columns=[col], inplace=True)

# Calculate duration if INSTALL and REMOVAL timestamps exist
if 'INSTALL_TIMESTAMP' in df.columns and 'REMOVAL_TIMESTAMP' in df.columns:
    df['SERVICE_DURATION'] = df['REMOVAL_TIMESTAMP'] - df['INSTALL_TIMESTAMP']

# Encode categorical columns
for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
        df[col] = df[col].astype(str).fillna('missing').astype('category')

X_train = df.iloc[:n_train].copy()
X_test = df.iloc[n_train:].copy()

# 5-Fold Cross Validation with LightGBM
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]
    
    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    test_preds += model.predict(X_test) / kf.n_splits

# Save predictions
output = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'WATERDEPTH': test_preds
})
output.to_csv('predictions.csv', index=False)