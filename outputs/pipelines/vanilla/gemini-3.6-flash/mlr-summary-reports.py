import os
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb

warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/mlr-summary-reports/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/mlr-summary-reports/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = 'Adjusted MLR'

if train[target_col].dtype == 'object':
    train[target_col] = train[target_col].astype(str).str.replace('%', '', regex=False).str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip()

train[target_col] = pd.to_numeric(train[target_col], errors='coerce')
train = train.dropna(subset=[target_col]).reset_index(drop=True)
y_train = train[target_col].values

train_features = train.drop(columns=[target_col], errors='ignore')
test_features = test.drop(columns=[target_col], errors='ignore')

n_train = len(train_features)
df_all = pd.concat([train_features, test_features], axis=0).reset_index(drop=True)

for col in list(df_all.columns):
    if df_all[col].dtype == 'object' or pd.api.types.is_datetime64_any_dtype(df_all[col]):
        try:
            dt = pd.to_datetime(df_all[col], errors='coerce')
            if dt.notna().sum() > 0.5 * len(df_all) and not pd.api.types.is_numeric_dtype(df_all[col]):
                df_all[f'{col}_year'] = dt.dt.year
                df_all[f'{col}_month'] = dt.dt.month
                df_all[f'{col}_day'] = dt.dt.day
                df_all[f'{col}_dayofweek'] = dt.dt.dayofweek
                df_all.drop(columns=[col], inplace=True)
                continue
        except Exception:
            pass

    if df_all[col].dtype == 'object':
        s_cleaned = df_all[col].astype(str).str.replace('$', '', regex=False).str.replace('%', '', regex=False).str.replace(',', '', regex=False).str.strip()
        num = pd.to_numeric(s_cleaned, errors='coerce')
        if num.notna().sum() > 0.5 * len(df_all):
            df_all[col] = num
            continue

    if df_all[col].dtype == 'object' or isinstance(df_all[col].dtype, pd.CategoricalDtype):
        df_all[col] = df_all[col].astype(str)
        freq = df_all[col].value_counts(normalize=True).to_dict()
        df_all[f'{col}_freq'] = df_all[col].map(freq).astype(float)
        df_all[col] = pd.factorize(df_all[col])[0]

df_all.columns = [re.sub(r'[^\w\s]', '_', str(c)).replace(' ', '_') for c in df_all.columns]

for col in df_all.columns:
    df_all[col] = pd.to_numeric(df_all[col], errors='coerce').fillna(0)

df_all = df_all.replace([np.inf, -np.inf], 0)

X_train = df_all.iloc[:n_train].copy()
X_test = df_all.iloc[n_train:].copy()

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds = np.zeros(len(X_test))

for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[trn_idx], y_train[trn_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    model_lgb = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        random_state=42 + fold,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    model_cb = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        random_state=42 + fold,
        verbose=0,
        early_stopping_rounds=50
    )
    model_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va))

    model_xgb = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        random_state=42 + fold,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        early_stopping_rounds=50
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    pred_te = (model_lgb.predict(X_test) + model_cb.predict(X_test) + model_xgb.predict(X_test)) / 3.0
    test_preds += pred_te / n_splits

sub = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    target_col: test_preds
})
sub.to_csv('predictions.csv', index=False)