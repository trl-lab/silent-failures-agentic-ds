import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')

train_path = "/path/to/user/workspace/ds_failures/datasets/strable/disbursements-in-trust-funds/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/disbursements-in-trust-funds/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Cash Disbursements (US$)"

train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)

y = train_df[target_col].values
X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.copy()

if target_col in X_test_raw.columns:
    X_test_raw = X_test_raw.drop(columns=[target_col])

n_train = len(X_train_raw)
df_all = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

cols_to_check = list(df_all.columns)
for col in cols_to_check:
    if pd.api.types.is_datetime64_any_dtype(df_all[col]):
        dt_series = df_all[col]
        df_all[f'{col}_year'] = dt_series.dt.year
        df_all[f'{col}_month'] = dt_series.dt.month
        df_all[f'{col}_day'] = dt_series.dt.day
        df_all[f'{col}_dayofweek'] = dt_series.dt.dayofweek
        df_all[f'{col}_quarter'] = dt_series.dt.quarter
        df_all.drop(columns=[col], inplace=True)
    elif df_all[col].dtype == 'object':
        try:
            parsed = pd.to_datetime(df_all[col].dropna().iloc[:50], errors='raise')
            dt_series = pd.to_datetime(df_all[col], errors='coerce')
            df_all[f'{col}_year'] = dt_series.dt.year
            df_all[f'{col}_month'] = dt_series.dt.month
            df_all[f'{col}_day'] = dt_series.dt.day
            df_all[f'{col}_dayofweek'] = dt_series.dt.dayofweek
            df_all[f'{col}_quarter'] = dt_series.dt.quarter
            df_all.drop(columns=[col], inplace=True)
        except Exception:
            pass

for col in list(df_all.columns):
    if df_all[col].nunique(dropna=False) <= 1:
        df_all.drop(columns=[col], inplace=True)

cat_cols = []
num_cols = []
for col in df_all.columns:
    if pd.api.types.is_numeric_dtype(df_all[col]):
        num_cols.append(col)
    else:
        cat_cols.append(col)

for col in cat_cols:
    freq = df_all[col].value_counts(dropna=False).to_dict()
    df_all[f'{col}_freq'] = df_all[col].map(freq).fillna(0)

for col in cat_cols:
    df_all[col] = df_all[col].fillna('missing').astype(str).astype('category')

X_train = df_all.iloc[:n_train].copy()
X_test = df_all.iloc[n_train:].copy()

X_train_num = X_train.copy()
X_test_num = X_test.copy()
for col in cat_cols:
    X_train_num[col] = X_train_num[col].cat.codes
    X_test_num[col] = X_test_num[col].cat.codes

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]
    
    lgb_model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        n_estimators=1000,
        learning_rate=0.04,
        num_leaves=63,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_preds += lgb_model.predict(X_test) / kf.n_splits

    X_tr_cat = X_tr.copy()
    X_va_cat = X_va.copy()
    X_te_cat = X_test.copy()
    for col in cat_cols:
        X_tr_cat[col] = X_tr_cat[col].astype(str)
        X_va_cat[col] = X_va_cat[col].astype(str)
        X_te_cat[col] = X_te_cat[col].astype(str)

    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric='RMSE',
        random_seed=42 + fold,
        verbose=0,
        cat_features=cat_cols if len(cat_cols) > 0 else None
    )
    cat_model.fit(
        X_tr_cat, y_tr,
        eval_set=(X_va_cat, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cat_preds += cat_model.predict(X_te_cat) / kf.n_splits

    X_tr_xgb = X_train_num.iloc[train_idx]
    X_va_xgb = X_train_num.iloc[val_idx]

    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.04,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=50
    )
    xgb_model.fit(
        X_tr_xgb, y_tr,
        eval_set=[(X_va_xgb, y_va)],
        verbose=False
    )
    xgb_preds += xgb_model.predict(X_test_num) / kf.n_splits

final_preds = 0.4 * lgb_preds + 0.4 * cat_preds + 0.2 * xgb_preds

sub = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'Cash Disbursements (US$)': final_preds
})
sub.to_csv('predictions.csv', index=False)