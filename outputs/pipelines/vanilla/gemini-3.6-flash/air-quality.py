import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold

train_path = '/path/to/user/workspace/ds_failures/datasets/uci/air-quality/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/air-quality/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = 'CO(GT)'

train = train[train[target_col].notna()].copy()
train = train[train[target_col] >= 0].reset_index(drop=True)

y_train = train[target_col].values
X_train_raw = train.drop(columns=[target_col])
X_test_raw = test.drop(columns=[target_col], errors='ignore')

def preprocess(df):
    df = df.copy()
    
    if 'Date' in df.columns and 'Time' in df.columns:
        datetime_str = df['Date'].astype(str) + ' ' + df['Time'].astype(str)
        dt = pd.to_datetime(datetime_str, format='mixed', errors='coerce')
        
        df['hour'] = dt.dt.hour
        df['dayofweek'] = dt.dt.dayofweek
        df['month'] = dt.dt.month
        df['day'] = dt.dt.day
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
        
        df = df.drop(columns=['Date', 'Time'], errors='ignore')
        
    sensor_cols = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)']
    gt_cols = ['NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
    other_cols = ['T', 'RH', 'AH']
    
    num_cols = sensor_cols + gt_cols + other_cols
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df.loc[df[col] <= -100, col] = np.nan
            
    existing_sensors = [c for c in sensor_cols if c in df.columns]
    if existing_sensors:
        df['sensor_mean'] = df[existing_sensors].mean(axis=1)
        df['sensor_std'] = df[existing_sensors].std(axis=1)
        df['sensor_min'] = df[existing_sensors].min(axis=1)
        df['sensor_max'] = df[existing_sensors].max(axis=1)
        
    for c in df.columns:
        if df[c].dtype == 'object':
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
    return df

df_all = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)
df_all_proc = preprocess(df_all)

n_train = len(X_train_raw)
X_train = df_all_proc.iloc[:n_train].reset_index(drop=True)
X_test = df_all_proc.iloc[n_train:].reset_index(drop=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))

for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[trn_idx], y_train[trn_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    lgb = LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    lgb.fit(X_tr, y_tr)
    
    cb = CatBoostRegressor(
        iterations=800,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=100)
    
    xgb = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=100
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    
    p_lgb = lgb.predict(X_test)
    p_cb = cb.predict(X_test)
    p_xgb = xgb.predict(X_test)
    
    fold_pred = (p_lgb + p_cb + p_xgb) / 3.0
    test_preds += fold_pred / kf.n_splits

test_preds = np.clip(test_preds, a_min=0, a_max=None)

sub = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'CO(GT)': test_preds
})

sub.to_csv('predictions.csv', index=False)