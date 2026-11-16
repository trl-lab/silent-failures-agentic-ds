import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping
from sklearn.model_selection import KFold

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/gainful-employment/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/gainful-employment/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = 'Debt-to-Earnings Annual Rate'

def clean_numeric_series(s):
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    s_clean = s.astype(str).str.strip()
    missing_vals = ['Privacy Suppressed', 'NULL', 'N/A', 'nan', 'NaN', 'None', '', '-', 'null', 'PRIVACY_SUPPRESSED']
    s_clean = s_clean.replace(missing_vals, np.nan)
    s_clean = s_clean.str.replace('$', '', regex=False).str.replace('%', '', regex=False).str.replace(',', '', regex=False)
    return pd.to_numeric(s_clean, errors='coerce')

train[target_col] = clean_numeric_series(train[target_col])
valid_mask = train[target_col].notna()
train = train[valid_mask].reset_index(drop=True)

y = train[target_col].values

X_train = train.drop(columns=[target_col])
X_test = test.drop(columns=[target_col] if target_col in test.columns else [])

combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)

missing_str_values = ['Privacy Suppressed', 'NULL', 'N/A', 'nan', 'NaN', 'None', '', '-', 'null', 'PRIVACY_SUPPRESSED']

for col in combined.columns:
    if pd.api.types.is_numeric_dtype(combined[col]):
        combined[col] = combined[col].replace([np.inf, -np.inf], np.nan)
    else:
        s_str = combined[col].astype(str).str.strip()
        s_clean = s_str.replace(missing_str_values, np.nan)
        s_num = s_clean.str.replace('$', '', regex=False).str.replace('%', '', regex=False).str.replace(',', '', regex=False)
        s_numeric = pd.to_numeric(s_num, errors='coerce')
        
        non_null_orig = s_clean.notna().sum()
        non_null_num = s_numeric.notna().sum()
        
        if non_null_orig > 0 and (non_null_num / non_null_orig) > 0.5:
            combined[col] = s_numeric
        else:
            combined[col] = s_clean.fillna('Missing').astype(str)

X_train_proc = combined.iloc[:len(train)].copy()
X_test_proc = combined.iloc[len(train):].copy()

cat_cols = [col for col in X_train_proc.columns if X_train_proc[col].dtype == object]

X_train_lgb = X_train_proc.copy()
X_test_lgb = X_test_proc.copy()
for c in cat_cols:
    X_train_lgb[c] = X_train_lgb[c].astype('category')
    X_test_lgb[c] = X_test_lgb[c].astype('category')

X_train_cb = X_train_proc.copy()
X_test_cb = X_test_proc.copy()
for c in cat_cols:
    X_train_cb[c] = X_train_cb[c].astype(str)
    X_test_cb[c] = X_test_cb[c].astype(str)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test_proc))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_proc, y)):
    y_tr, y_va = y[train_idx], y[val_idx]
    
    X_tr_lgb, X_va_lgb = X_train_lgb.iloc[train_idx], X_train_lgb.iloc[val_idx]
    lgb = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    lgb.fit(
        X_tr_lgb, y_tr,
        eval_set=[(X_va_lgb, y_va)],
        callbacks=[early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_pred = lgb.predict(X_test_lgb)
    
    X_tr_cb, X_va_cb = X_train_cb.iloc[train_idx], X_train_cb.iloc[val_idx]
    cb = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    cb.fit(
        X_tr_cb, y_tr,
        eval_set=(X_va_cb, y_va),
        cat_features=cat_cols if len(cat_cols) > 0 else None,
        early_stopping_rounds=50
    )
    cb_pred = cb.predict(X_test_cb)
    
    fold_pred = (lgb_pred + cb_pred) / 2.0
    test_preds += fold_pred / kf.n_splits

test_preds = np.nan_to_num(test_preds, nan=np.nanmean(y))

sub = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'Debt-to-Earnings Annual Rate': test_preds
})
sub.to_csv('predictions.csv', index=False)