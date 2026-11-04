import pandas as pd
import numpy as np
import warnings
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings('ignore')

train_path = "/path/to/user/workspace/ds_failures/datasets/strable/financial-intermediary-funds-cash-transfers/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/financial-intermediary-funds-cash-transfers/data/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = "Amount in USD"
y = train[target_col].values.astype(np.float64)

X_train = train.drop(columns=[target_col], errors='ignore').copy()
X_test = test.drop(columns=[target_col], errors='ignore').copy()

n_train = len(X_train)
df_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)

cols_to_drop = []
for col in df_all.columns:
    if col.lower() in ['id', 'index', 'unnamed: 0']:
        cols_to_drop.append(col)
df_all.drop(columns=cols_to_drop, errors='ignore', inplace=True)

for col in list(df_all.columns):
    if df_all[col].dtype == 'object' or pd.api.types.is_datetime64_any_dtype(df_all[col]):
        try:
            parsed = pd.to_datetime(df_all[col], errors='coerce')
            if parsed.notna().sum() > 0.3 * len(df_all):
                df_all[col + '_year'] = parsed.dt.year
                df_all[col + '_month'] = parsed.dt.month
                df_all[col + '_day'] = parsed.dt.day
                df_all[col + '_dayofweek'] = parsed.dt.dayofweek
                df_all[col + '_quarter'] = parsed.dt.quarter
                df_all[col + '_timestamp'] = parsed.astype('float64') // 10**9
        except Exception:
            pass

cat_cols = [col for col in df_all.columns if df_all[col].dtype == 'object' or df_all[col].dtype.name == 'category']

X_tr_proc = df_all.iloc[:n_train].copy()
X_te_proc = df_all.iloc[n_train:].copy()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
prior = np.mean(y)
m_smooth = 10.0

for col in cat_cols:
    str_col_tr = X_tr_proc[col].astype(str)
    str_col_te = X_te_proc[col].astype(str)
    
    freq_map = df_all[col].astype(str).value_counts().to_dict()
    X_tr_proc[col + '_freq'] = str_col_tr.map(freq_map).fillna(0).astype(np.float32)
    X_te_proc[col + '_freq'] = str_col_te.map(freq_map).fillna(0).astype(np.float32)
    
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    oe.fit(pd.concat([str_col_tr, str_col_te]).values.reshape(-1, 1))
    X_tr_proc[col + '_ord'] = oe.transform(str_col_tr.values.reshape(-1, 1)).ravel()
    X_te_proc[col + '_ord'] = oe.transform(str_col_te.values.reshape(-1, 1)).ravel()
    
    target_enc_col = col + '_target_enc'
    X_tr_proc[target_enc_col] = prior
    X_te_proc[target_enc_col] = prior
    
    for tr_idx, val_idx in kf.split(X_tr_proc):
        tr_cat = str_col_tr.iloc[tr_idx]
        tr_y = y[tr_idx]
        stats = pd.DataFrame({'cat': tr_cat, 'y': tr_y}).groupby('cat')['y'].agg(['count', 'mean'])
        smooth = (stats['count'] * stats['mean'] + m_smooth * prior) / (stats['count'] + m_smooth)
        X_tr_proc.loc[X_tr_proc.index[val_idx], target_enc_col] = str_col_tr.iloc[val_idx].map(smooth).fillna(prior).values
        
    stats_full = pd.DataFrame({'cat': str_col_tr, 'y': y}).groupby('cat')['y'].agg(['count', 'mean'])
    smooth_full = (stats_full['count'] * stats_full['mean'] + m_smooth * prior) / (stats_full['count'] + m_smooth)
    X_te_proc[target_enc_col] = str_col_te.map(smooth_full).fillna(prior).values

X_tr_proc.drop(columns=cat_cols, errors='ignore', inplace=True)
X_te_proc.drop(columns=cat_cols, errors='ignore', inplace=True)

for col in X_tr_proc.columns:
    X_tr_proc[col] = pd.to_numeric(X_tr_proc[col], errors='coerce').astype(np.float32)
    X_te_proc[col] = pd