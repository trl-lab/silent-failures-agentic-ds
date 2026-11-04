import os
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/contract-awards-investment-project-financing/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/contract-awards-investment-project-financing/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = "Supplier Contract Amount (USD)"
y = train[target_col].values.astype(np.float64)

train_len = len(train)
df_all = pd.concat([
    train.drop(columns=[target_col], errors='ignore'),
    test.drop(columns=[target_col], errors='ignore')
], axis=0, ignore_index=True)

df_all.columns = [re.sub(r'[\[\]<>\:,"]', '_', str(c)) for c in df_all.columns]

cols_to_process = list(df_all.columns)
for col in cols_to_process:
    if pd.api.types.is_datetime64_any_dtype(df_all[col]) or 'date' in col.lower():
        dt_s = pd.to_datetime(df_all[col], errors='coerce')
        if dt_s.notna().sum() > 0:
            df_all[col + '_year'] = dt_s.dt.year
            df_all[col + '_month'] = dt_s.dt.month
            df_all[col + '_day'] = dt_s.dt.day
            df_all[col + '_dayofweek'] = dt_s.dt.dayofweek
            df_all[col + '_dayofyear'] = dt_s.dt.dayofyear
            df_all[col + '_timestamp'] = pd.to_numeric(dt_s, errors='coerce') / 1e9
            df_all.drop(columns=[col], inplace=True)

cat_cols = []
for col in df_all.columns:
    if df_all[col].dtype == 'object' or df_all[col].dtype.name == 'category':
        cat_cols.append(col)

for c in cat_cols:
    freq = df_all[c].value_counts().to_dict()
    df_all[c + '_freq'] = df_all[c].map(freq).fillna(0)
    df_all[c] = df_all[c].astype(str).astype('category').cat.codes

X_train = df_all.iloc[:train_len].copy()
X_test = df_all.iloc[train_len:].copy()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for c in cat_cols:
    te_train = np.zeros(train_len)
    te_test = np.zeros(len(test))
    
    for train_idx, val_idx in kf.split(X_train):
        cat_means = pd.DataFrame({'cat': X_train[c].iloc[train_idx], 'y': y[train_idx]}).groupby('cat')['y'].mean()
        te_train[val_idx] = X_train[c].iloc[val_idx].map(cat_means).fillna(y[train_idx].mean())
        te_test += X_test[c].map(cat_means).fillna(y[train_idx].mean()) / kf.n_splits
        
    X_train[c + '_te'] = te_train
    X_test[c + '_te'] = te_test

const_cols = [c for c in X_train.columns if X_train[c].nunique(dropna=False) <= 1]
X_train.drop(columns=const_cols, inplace=True)
X_test.drop(columns=const_cols, inplace=True)

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]
    
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        random_state=42 + fold,
        n_jobs=-1,
        objective='regression',
        verbose=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_preds += model_lgb.predict(X_test) / kf.n_splits
    
    model_cat = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.04,
        depth=6,
        random_seed=42 + fold,
        verbose=0,
        loss_function='RMSE'
    )
    model_cat.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50
    )
    cat_preds += model_cat.predict(X_test) / kf.n_splits

    model_xgb = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        random_state=42 + fold,
        n_jobs=-1,
        tree_method='hist',
        objective='reg:squarederror',
        early_stopping_rounds=50
    )
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    xgb_preds += model_xgb.predict(X_test) / kf.n_splits

final_preds = lgb_preds * 0.4 + cat_preds * 0.3 + xgb_preds * 0.3

if np.min(y) >= 0:
    final_preds = np.clip(final_preds, 0, None)

submission = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    target_col: final_preds
})

submission.to_csv('predictions.csv', index=False)