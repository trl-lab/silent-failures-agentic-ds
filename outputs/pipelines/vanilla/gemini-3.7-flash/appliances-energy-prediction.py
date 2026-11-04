import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/uci/appliances-energy-prediction/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/appliances-energy-prediction/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

def preprocess(df):
    df = df.copy()
    # Fix date string formatting if needed and extract datetime features
    date_series = df['date'].astype(str).str.replace(r'(\d{4}-\d{2}-\d{2})(\d{2}:\d{2}:\d{2})', r'\1 \2', regex=True)
    dt = pd.to_datetime(date_series, errors='coerce')
    
    df['hour'] = dt.dt.hour
    df['minute'] = dt.dt.minute
    df['dayofweek'] = dt.dt.dayofweek
    df['day'] = dt.dt.day
    df['month'] = dt.dt.month
    df['time_of_day_min'] = df['hour'] * 60 + df['minute']
    
    # Cyclical encoding of time features
    df['sin_time'] = np.sin(2 * np.pi * df['time_of_day_min'] / 1440.0)
    df['cos_time'] = np.cos(2 * np.pi * df['time_of_day_min'] / 1440.0)
    df['sin_dow'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
    df['cos_dow'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)
    
    # Temperature differences
    if 'T6' in df.columns and 'T_out' in df.columns:
        df['T6_diff_Tout'] = df['T6'] - df['T_out']
    if 'T1' in df.columns and 'T_out' in df.columns:
        df['T1_diff_Tout'] = df['T1'] - df['T_out']
    if 'RH_6' in df.columns and 'RH_out' in df.columns:
        df['RH6_diff_RHout'] = df['RH_6'] - df['RH_out']
        
    df = df.drop(columns=['date'])
    return df

target_col = 'Appliances'
y_train = train_df[target_col].values

X_train_full = preprocess(train_df.drop(columns=[target_col]))
X_test_full = preprocess(test_df)

# Ensure numeric columns only
numeric_cols = [c for c in X_train_full.columns if pd.api.types.is_numeric_dtype(X_train_full[c])]
X_train = X_train_full[numeric_cols].fillna(0)
X_test = X_test_full[numeric_cols].fillna(0)

# K-Fold CV Ensembling
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    # LightGBM model
    lgb = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbose=-1
    )
    lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[]
    )
    
    # CatBoost model
    cb = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    cb.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    
    pred_lgb = lgb.predict(X_test)
    pred_cb = cb.predict(X_test)
    
    test_preds += (0.5 * pred_lgb + 0.5 * pred_cb) / kf.n_splits

# Ensure non-negative predictions
test_preds = np.clip(test_preds, a_min=0, a_max=None)

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'Appliances': test_preds
})
output_df.to_csv('predictions.csv', index=False)