import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/industry-payments-entity/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/industry-payments-entity/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = 'entityPayment'

train = train.dropna(subset=[target_col]).reset_index(drop=True)
y = train[target_col].values.astype(np.float64)

X = train.drop(columns=[target_col])
X_test = test.copy()

num_cols = []
cat_cols = []
date_cols = []

for col in X.columns:
    if pd.api.types.is_datetime64_any_dtype(X[col]):
        date_cols.append(col)
    elif pd.api.types.is_numeric_dtype(X[col]):
        num_cols.append(col)
    else:
        cat_cols.append(col)

for col in cat_cols.copy():
    try:
        sample = X[col].dropna().head(100)
        if len(sample) > 0:
            parsed = pd.to_datetime(sample, errors='coerce')
            if parsed.notna().mean() > 0.8:
                X[col] = pd.to_datetime(X[col], errors='coerce')
                X_test[col] = pd.to_datetime(X_test[col], errors='coerce')
                cat_cols.remove(col)
                date_cols.append(col)
    except Exception:
        pass

for col in date_cols:
    for df in [X, X_test]:
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
        df[f'{col}_day'] = df[col].dt.day
        df[f'{col}_dayofweek'] = df[col].dt.dayofweek
        df[f'{col}_quarter'] = df[col].dt.quarter
    X.drop(columns=[col], inplace=True)
    X_test.drop(columns=[col], inplace=True)

for col in cat_cols:
    X[col] = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)
    
    freq = pd.concat([X[col], X_test[col]]).value_counts()
    X[f'{col}_freq'] = X[col].map(freq).fillna(0).astype(np.float32)
    X_test[f'{col}_freq'] = X_test[col].map(freq).fillna(0).astype(np.float32)
    
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')

X_xgb = X.copy()
X_test_xgb = X_test.copy()
for col in cat_cols:
    X_xgb[col] = X_xgb[col].cat.codes
    X_test_xgb[col] = X_test_xgb[col].cat.codes

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]
    
    lgb_model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_preds += lgb_model.predict(X_test) / n_splits
    
    cat_cols_list = list(cat_cols)
    cb_model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.04,
        depth=6,
        eval_metric='RMSE',
        random_seed=42 + fold,
        verbose=0,
        cat_features=cat_cols_list if len(cat_cols_list) > 0 else None
    )
    cb_model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cb_preds += cb_model.predict(X_test) / n_splits
    
    X_tr_xgb, X_va_xgb = X_xgb.iloc[train_idx], X_xgb.iloc[val_idx]
    xgb_model = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        random_state=42 + fold,
        n_jobs=-1,
        tree_method='hist',
        early_stopping_rounds=50
    )
    xgb_model.fit(
        X_tr_xgb, y_tr,
        eval_set=[(X_va_xgb, y_va)],
        verbose=False
    )
    xgb_preds += xgb_model.predict(X_test_xgb) / n_splits

final_preds = 0.4 * lgb_preds + 0.4 * cb_preds + 0.2 * xgb_preds

sub = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'entityPayment': final_preds
})

sub.to_csv('predictions.csv', index=False)