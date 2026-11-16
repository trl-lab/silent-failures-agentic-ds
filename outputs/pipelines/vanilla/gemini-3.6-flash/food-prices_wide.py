import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/food-prices_wide/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/food-prices_wide/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = 'usdprice'

train = train.dropna(subset=[target_col]).reset_index(drop=True)
y_train = train[target_col].values

n_train = len(train)
df_all = pd.concat([train.drop(columns=[target_col], errors='ignore'), test.drop(columns=[target_col], errors='ignore')], ignore_index=True)

for col in list(df_all.columns):
    if df_all[col].dtype == 'object':
        try:
            converted = pd.to_datetime(df_all[col], errors='coerce')
            if converted.notna().sum() > 0.5 * len(df_all):
                df_all[col + '_year'] = converted.dt.year
                df_all[col + '_month'] = converted.dt.month
                df_all[col + '_day'] = converted.dt.day
                df_all[col + '_dayofweek'] = converted.dt.dayofweek
                df_all[col + '_quarter'] = converted.dt.quarter
        except Exception:
            pass

cat_cols = []
num_cols = []

for col in list(df_all.columns):
    if df_all[col].dtype == 'object' or df_all[col].dtype.name == 'category':
        cat_cols.append(col)
        df_all[col] = df_all[col].fillna('missing').astype(str)
    elif np.issubdtype(df_all[col].dtype, np.number):
        num_cols.append(col)

for col in cat_cols:
    freq = df_all[col].value_counts().to_dict()
    df_all[col + '_freq'] = df_all[col].map(freq)

X_train = df_all.iloc[:n_train].copy().reset_index(drop=True)
X_test = df_all.iloc[n_train:].copy().reset_index(drop=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for c in cat_cols:
    target_mean_col = c + '_target_mean'
    X_train[target_mean_col] = np.nan
    X_test[target_mean_col] = np.nan
    
    global_mean = y_train.mean()
    
    for tr_idx, val_idx in kf.split(X_train):
        val_cats = X_train.iloc[val_idx][c]
        tr_df = pd.DataFrame({c: X_train.iloc[tr_idx][c], 'y': y_train[tr_idx]})
        group_means = tr_df.groupby(c)['y'].mean()
        X_train.iloc[val_idx, X_train.columns.get_loc(target_mean_col)] = val_cats.map(group_means).values
        
    X_train[target_mean_col] = X_train[target_mean_col].fillna(global_mean)
    
    full_df = pd.DataFrame({c: X_train[c], 'y': y_train})
    overall_group_means = full_df.groupby(c)['y'].mean()
    X_test[target_mean_col] = X_test[c].map(overall_group_means).fillna(global_mean)

for col in cat_cols:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

lgb_preds = np.zeros(len(X_test))

use_log = bool((y_train >= 0).all())
if use_log:
    y_train_log = np.log1p(y_train)
    lgb_log_preds = np.zeros(len(X_test))
    cb_log_preds = np.zeros(len(X_test))

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'n_estimators': 1500,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1
}

cb_params = {
    'iterations': 1500,
    'learning_rate': 0.05,
    'depth': 6,
    'random_seed': 42,
    'verbose': 0
}

X_train_cb = X_train.copy()
X_test_cb = X_test.copy()
for col in cat_cols:
    X_train_cb[col] = X_train_cb[col].astype(str)
    X_test_cb[col] = X_test_cb[col].astype(str)

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_preds += model_lgb.predict(X_test) / kf.n_splits
    
    if use_log:
        y_tr_log, y_va_log = y_train_log[tr_idx], y_train_log[val_idx]
        
        model_lgb_log = lgb.LGBMRegressor(**lgb_params)
        model_lgb_log.fit(
            X_tr, y_tr_log,
            eval_set=[(X_va, y_va_log)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        lgb_log_preds += np.expm1(model_lgb_log.predict(X_test)) / kf.n_splits
        
        model_cb = CatBoostRegressor(**cb_params, cat_features=cat_cols)
        model_cb.fit(
            X_train_cb.iloc[tr_idx], y_tr_log,
            eval_set=(X_train_cb.iloc[val_idx], y_va_log),
            early_stopping_rounds=50,
            verbose=False
        )
        cb_log_preds += np.expm1(model_cb.predict(X_test_cb)) / kf.n_splits

if use_log:
    final_preds = 0.4 * lgb_preds + 0.3 * lgb_log_preds + 0.3 * cb_log_preds
else:
    final_preds = lgb_preds

if (y_train >= 0).all():
    final_preds = np.clip(final_preds, a_min=0, a_max=None)

sub = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'usdprice': final_preds
})

sub.to_csv('predictions.csv', index=False)