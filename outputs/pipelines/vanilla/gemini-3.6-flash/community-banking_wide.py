import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import KFold

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/community-banking_wide/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/community-banking_wide/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = 'LOANTOASSET'
y = train[target_col].values.astype(np.float64)

X = train.drop(columns=[target_col])
X_test = test.copy()

if target_col in X_test.columns:
    X_test = X_test.drop(columns=[target_col])

for col in list(X.columns):
    is_date = False
    if pd.api.types.is_datetime64_any_dtype(X[col]):
        is_date = True
    elif X[col].dtype == 'object':
        sample = X[col].dropna().astype(str).head(30)
        if len(sample) > 0:
            try:
                parsed = pd.to_datetime(sample, errors='coerce')
                if parsed.notna().mean() > 0.8:
                    X[col] = pd.to_datetime(X[col], errors='coerce')
                    X_test[col] = pd.to_datetime(X_test[col], errors='coerce')
                    is_date = True
            except Exception:
                pass

    if is_date:
        X[col + '_year'] = X[col].dt.year
        X[col + '_month'] = X[col].dt.month
        X[col + '_quarter'] = X[col].dt.quarter
        X[col + '_day'] = X[col].dt.day

        X_test[col + '_year'] = X_test[col].dt.year
        X_test[col + '_month'] = X_test[col].dt.month
        X_test[col + '_quarter'] = X_test[col].dt.quarter
        X_test[col + '_day'] = X_test[col].dt.day

        X = X.drop(columns=[col])
        X_test = X_test.drop(columns=[col])

drop_cols = [col for col in X.columns if X[col].nunique(dropna=False) <= 1]
X = X.drop(columns=drop_cols)
X_test = X_test.drop(columns=drop_cols)

cat_cols = []
for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype.name == 'category':
        cat_cols.append(col)

for col in cat_cols:
    X[col] = X[col].fillna('__MISSING__').astype(str)
    X_test[col] = X_test[col].fillna('__MISSING__').astype(str)

    categories = sorted(list(set(X[col].unique()).union(set(X_test[col].unique()))))
    cat_type = pd.CategoricalDtype(categories=categories)
    X[col] = X[col].astype(cat_type)
    X_test[col] = X_test[col].astype(cat_type)

num_cols = [col for col in X.columns if col not in cat_cols]
for col in num_cols:
    X[col] = X[col].replace([np.inf, -np.inf], np.nan)
    X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))

X_cb = X.copy()
X_test_cb = X_test.copy()
for col in cat_cols:
    X_cb[col] = X_cb[col].astype(str)
    X_test_cb[col] = X_test_cb[col].astype(str)

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    model_lgb = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbose=-1,
        n_jobs=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_preds += model_lgb.predict(X_test) / kf.n_splits

    X_tr_cb, X_va_cb = X_cb.iloc[train_idx], X_cb.iloc[val_idx]
    model_cb = cb.CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0,
        thread_count=-1
    )
    model_cb.fit(
        X_tr_cb, y_tr,
        eval_set=(X_va_cb, y_va),
        cat_features=cat_cols,
        early_stopping_rounds=50,
        verbose=False
    )
    cb_preds += model_cb.predict(X_test_cb) / kf.n_splits

final_preds = 0.5 * lgb_preds + 0.5 * cb_preds

sub = pd.DataFrame({
    'id': np.arange(len(test)),
    'LOANTOASSET': final_preds
})

sub.to_csv('predictions.csv', index=False)