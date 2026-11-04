import os
import re
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/medically-underserved-areas-populations/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/medically-underserved-areas-populations/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = 'IMU Score'

train = train.dropna(subset=[target_col]).reset_index(drop=True)
y = train[target_col].values.astype(np.float64)

X_train = train.drop(columns=[target_col])
X_test = test.drop(columns=[target_col]) if target_col in test.columns else test.copy()

common_cols = [c for c in X_train.columns if c in X_test.columns]
X_tr_raw = X_train[common_cols].copy()
X_te_raw = X_test[common_cols].copy()

clean_cols = [re.sub(r'[\[\]<>\:,"]', '_', str(c)) for c in common_cols]
X_tr_raw.columns = clean_cols
X_te_raw.columns = clean_cols

processed_tr = pd.DataFrame(index=X_tr_raw.index)
processed_te = pd.DataFrame(index=X_te_raw.index)

for col in clean_cols:
    s_tr = X_tr_raw[col]
    s_te = X_te_raw[col]
    
    is_dt = False
    if pd.api.types.is_datetime64_any_dtype(s_tr) or 'date' in col.lower() or 'time' in col.lower():
        try:
            dt_tr = pd.to_datetime(s_tr, errors='coerce')
            dt_te = pd.to_datetime(s_te, errors='coerce')
            if dt_tr.notna().sum() > 0:
                processed_tr[f'{col}_year'] = dt_tr.dt.year
                processed_tr[f'{col}_month'] = dt_tr.dt.month
                processed_tr[f'{col}_day'] = dt_tr.dt.day
                processed_tr[f'{col}_dayofweek'] = dt_tr.dt.dayofweek
                is_dt = True
        except Exception:
            pass
            
    if is_dt:
        continue
        
    if s_tr.dtype == 'object':
        try:
            str_tr = s_tr.astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.replace('%', '', regex=False).str.strip()
            str_te = s_te.astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.replace('%', '', regex=False).str.strip()
            num_tr = pd.to_numeric(str_tr, errors='coerce')
            num_te = pd.to_numeric(str_te, errors='coerce')
            if num_tr.notna().mean() > 0.5:
                processed_tr[col] = num_tr.astype(np.float32)
                processed_te[col] = num_te.astype(np.float32)
                continue
        except Exception:
            pass

    if s_tr.dtype == 'object' or s_tr.dtype.name == 'category' or s_tr.dtype == 'bool':
        freq = pd.concat([s_tr, s_te]).value_counts(dropna=False).to_dict()
        processed_tr[f'{col}_freq'] = s_tr.map(freq).astype(np.float32)
        processed_te[f'{col}_freq'] = s_te.map(freq).astype(np.float32)
        
        combined = pd.concat([s_tr.astype(str), s_te.astype(str)], axis=0)
        codes, _ = pd.factorize(combined)
        processed_tr[col] = codes[:len(s_tr)].astype(np.int32)
        processed_te[col] = codes[len(s_tr):].astype(np.int32)
    else:
        processed_tr[col] = s_tr.astype(np.float32)
        processed_te[col] = s_te.astype(np.float32)

X_tr = processed_tr.loc[:, ~processed_tr.columns.duplicated()].copy()
X_te = processed_te.loc[:, ~processed_te.columns.duplicated()][X_tr.columns].copy()

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X_tr))
oof_xgb = np.zeros(len(X_tr))
oof_cb  = np.zeros(len(X_tr))

preds_lgb = np.zeros(len(X_te))
preds_xgb = np.zeros(len(X_te))
preds_cb  = np.zeros(len(X_te))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_tr, y)):
    X_train_f, y_train_f = X_tr.iloc[train_idx], y[train_idx]
    X_val_f, y_val_f = X_tr.iloc[val_idx], y[val_idx]
    
    model_lgb = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    model_lgb.fit(
        X_train_f, y_train_f,
        eval_set=[(X_val_f, y_val