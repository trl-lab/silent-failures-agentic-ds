import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/first-time-nadac-rates/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/first-time-nadac-rates/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = 'NADAC Rate'
y = train[target_col].values.astype(np.float32)

X_train = train.drop(columns=[target_col]).copy()
X_test = test.copy()

for col in list(X_train.columns):
    if pd.api.types.is_datetime64_any_dtype(X_train[col]):
        for df in [X_train, X_test]:
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df[f'{col}_dayofyear'] = df[col].dt.dayofyear
            df[f'{col}_timestamp'] = df[col].astype('int64') // 10**9
        X_train.drop(columns=[col], inplace=True)
        X_test.drop(columns=[col], inplace=True)
    elif X_train[col].dtype == 'object':
        try:
            sample = X_train[col].dropna().head(100)
            if len(sample) > 0 and pd.to_datetime(sample, errors='coerce').notna().mean() > 0.8:
                tr_dt = pd.to_datetime(X_train[col], errors='coerce')
                te_dt = pd.to_datetime(X_test[col], errors='coerce')
                for df, dt_s in [(X_train, tr_dt), (X_test, te_dt)]:
                    df[f'{col}_year'] = dt_s.dt.year
                    df[f'{col}_month'] = dt_s.dt.month
                    df[f'{col}_day'] = dt_s.dt.day
                    df[f'{col}_dayofweek'] = dt_s.dt.dayofweek
                    df[f'{col}_dayofyear'] = dt_s.dt.dayofyear
                    df[f'{col}_timestamp'] = dt_s.astype('int64') // 10**9
                X_train.drop(columns=[col], inplace=True)
                X_test.drop(columns=[col], inplace=True)
        except Exception:
            pass

cat_cols = []
for col in X_train.columns:
    if X_train[col].dtype == 'object' or isinstance(X_train[col].dtype, pd.CategoricalDtype):
        cat_cols.append(col)

for col in cat_cols:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)
    categories = pd.concat([X_train[col], X_test[col]]).unique()
    X_train[col] = pd.Categorical(X_train[col], categories=categories)
    X_test[col] = pd.Categorical(X_test[col], categories=categories)

for col in X_train.columns:
    if col not in cat_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

X_train_cb = X_train.copy()
X_test_cb = X_test.copy()
for col in cat_cols:
    X_train_cb[col] = X_train_cb[col].astype(str).fillna('missing')
    X_test_cb[col] = X_test_cb[col].astype(str).fillna('missing')

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    model_lgb = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        n_estimators=1500,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )

    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    lgb_preds += model_lgb.predict(X_test) / n_splits

cat_features_cb = cat_cols if len(cat_cols) > 0 else None

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_cb, y)):
    X_tr, y_tr = X_train_cb.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train_cb.iloc[val_idx], y[val_idx]

    tr_pool = Pool(X_tr, y_tr, cat_features=cat_features_cb)
    va_pool = Pool(X_va, y_va, cat_features=cat_features_cb)

    model_cb = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        random_seed=42 + fold,
        verbose=0
    )

    model_cb.fit(
        tr_pool,
        eval_set=va_pool,
        early_stopping_rounds=50,
        verbose=False
    )

    test_pool = Pool(X_test_cb, cat_features=cat_features_cb)
    cb_preds += model_cb.predict(test_pool) / n_splits

final_preds = 0.5 * lgb_preds + 0.5 * cb_preds
final_preds = np.clip(final_preds, a_min=0, a_max=None)

sub = pd.DataFrame({
    'id': np.arange(len(test)),
    'NADAC Rate': final_preds
})

sub.to_csv('predictions.csv', index=False)