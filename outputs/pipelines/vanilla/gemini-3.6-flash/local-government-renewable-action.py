import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/local-government-renewable-action/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/local-government-renewable-action/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = 'Size of Purchase (MW)'

train = train.dropna(subset=[target_col]).reset_index(drop=True)
y_train = train[target_col].values.astype(float)
X_train = train.drop(columns=[target_col])
X_test = test.copy()

id_cols_to_drop = [c for c in X_train.columns if c.lower() in ['id', 'unnamed: 0', 'index']]
if id_cols_to_drop:
    X_train = X_train.drop(columns=id_cols_to_drop)
    X_test = X_test.drop(columns=id_cols_to_drop, errors='ignore')

for col in list(X_train.columns):
    if X_train[col].dtype == 'object' or str(X_train[col].dtype).startswith('string'):
        non_null_samples = X_train[col].dropna()
        if len(non_null_samples) > 0:
            sample_str = str(non_null_samples.iloc[0])
            if any(sep in sample_str for sep in ['-', '/', ':']) and len(sample_str) >= 8:
                try:
                    dt_train = pd.to_datetime(X_train[col], errors='coerce')
                    if dt_train.notna().sum() > 0.5 * len(dt_train):
                        dt_test = pd.to_datetime(X_test[col], errors='coerce')
                        X_train[f'{col}_year'] = dt_train.dt.year
                        X_train[f'{col}_month'] = dt_train.dt.month
                        X_train[f'{col}_day'] = dt_train.dt.day
                        X_train[f'{col}_dayofweek'] = dt_train.dt.dayofweek
                        X_test[f'{col}_year'] = dt_test.dt.year
                        X_test[f'{col}_month'] = dt_test.dt.month
                        X_test[f'{col}_day'] = dt_test.dt.day
                        X_test[f'{col}_dayofweek'] = dt_test.dt.dayofweek
                        X_train = X_train.drop(columns=[col])
                        X_test = X_test.drop(columns=[col])
                except Exception:
                    pass

cat_cols = []
for col in X_train.columns:
    if X_train[col].dtype == 'bool':
        X_train[col] = X_train[col].astype(int)
        X_test[col] = X_test[col].astype(int)
    elif X_train[col].dtype == 'object' or str(X_train[col].dtype) in ['string', 'category']:
        X_train[col] = X_train[col].fillna('missing').astype(str)
        X_test[col] = X_test[col].fillna('missing').astype(str)
        cat_cols.append(col)
    else:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

clean_cols = []
seen = {}
for c in X_train.columns:
    c_clean = re.sub(r'[\[\]<>\":,]', '_', str(c))
    if c_clean in seen:
        seen[c_clean] += 1
        clean_cols.append(f"{c_clean}_{seen[c_clean]}")
    else:
        seen[c_clean] = 0
        clean_cols.append(c_clean)

cat_cols = [clean_cols[i] for i, col in enumerate(X_train.columns) if col in cat_cols]
X_train.columns = clean_cols
X_test.columns = clean_cols

X_train_cat = X_train.copy()
X_test_cat = X_test.copy()

for col in cat_cols:
    categories = pd.concat([X_train[col], X_test[col]]).unique()
    X_train_cat[col] = pd.Categorical(X_train[col], categories=categories)
    X_test_cat[col] = pd.Categorical(X_test[col], categories=categories)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))

lgb_ok = True
cat_ok = True
xgb_ok = True

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    y_tr, y_va = y_train[train_idx], y_train[val_idx]

    if lgb_ok:
        try:
            model_lgb = lgb.LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.03,
                num_leaves=31,
                random_state=42 + fold,
                verbosity=-1,
                n_jobs=-1
            )
            model_lgb.fit(
                X_train_cat.iloc[train_idx], y_tr,
                eval_set=[(X_train_cat.iloc[val_idx], y_va)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            lgb_preds += model_lgb.predict(X_test_cat) / kf.n_splits
        except Exception:
            lgb_ok = False

    if cat_ok:
        try:
            model_cb = cb.CatBoostRegressor(
                iterations=1000,
                learning_rate=0.05,
                cat_features=cat_cols if cat_cols else None,
                random_seed=42 + fold,
                verbose=0
            )
            model_cb.fit(
                X_train.iloc[