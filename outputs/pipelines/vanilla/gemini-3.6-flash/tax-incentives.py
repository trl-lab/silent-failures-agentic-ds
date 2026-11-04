import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/tax-incentives/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/tax-incentives/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'credit_amount'
y = train_df[target_col].values.astype(float)

X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

drop_cols = []
for col in X_train.columns:
    if col.lower() in ['id', 'index', 'row_id', 'unnamed: 0']:
        drop_cols.append(col)

X_train = X_train.drop(columns=drop_cols)
X_test = X_test.drop(columns=drop_cols)

datetime_cols = []
for col in X_train.columns:
    if pd.api.types.is_datetime64_any_dtype(X_train[col]):
        datetime_cols.append(col)
    elif X_train[col].dtype == 'object':
        try:
            parsed = pd.to_datetime(X_train[col], errors='coerce')
            if parsed.notna().sum() > 0.5 * len(X_train):
                X_train[col] = parsed
                X_test[col] = pd.to_datetime(X_test[col], errors='coerce')
                datetime_cols.append(col)
        except Exception:
            pass

for col in datetime_cols:
    X_train[f'{col}_year'] = X_train[col].dt.year
    X_train[f'{col}_month'] = X_train[col].dt.month
    X_train[f'{col}_day'] = X_train[col].dt.day
    X_train[f'{col}_dayofweek'] = X_train[col].dt.dayofweek

    X_test[f'{col}_year'] = X_test[col].dt.year
    X_test[f'{col}_month'] = X_test[col].dt.month
    X_test[f'{col}_day'] = X_test[col].dt.day
    X_test[f'{col}_dayofweek'] = X_test[col].dt.dayofweek

X_train = X_train.drop(columns=datetime_cols)
X_test = X_test.drop(columns=datetime_cols)

cat_cols = []
num_cols = []

for col in X_train.columns:
    if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category':
        cat_cols.append(col)
    else:
        num_cols.append(col)

for col in cat_cols:
    X_train[col] = X_train[col].astype(str).fillna('missing')
    X_test[col] = X_test[col].astype(str).fillna('missing')

for col in num_cols:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

X_train_lgb = X_train.copy()
X_test_lgb = X_test.copy()

for col in cat_cols:
    all_categories = pd.concat([X_train[col], X_test[col]]).unique()
    X_train_lgb[col] = pd.Categorical(X_train[col], categories=all_categories)
    X_test_lgb[col] = pd.Categorical(X_test[col], categories=all_categories)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

preds_lgb = np.zeros(len(X_test))
preds_cb = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr_lgb, y_tr = X_train_lgb.iloc[train_idx], y[train_idx]
    X_va_lgb, y_va = X_train_lgb.iloc[val_idx], y[val_idx]

    model_lgb = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbose=-1,
        n_jobs=-1
    )
    model_lgb.fit(
        X_tr_lgb, y_tr,
        eval_set=[(X_va_lgb, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    preds_lgb += model_lgb.predict(X_test_lgb) / kf.n_splits

    X_tr_cb, y_tr_cb = X_train.iloc[train_idx], y[train_idx]
    X_va_cb, y_va_cb = X_train.iloc[val_idx], y[val_idx]

    model_cb = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.04,
        depth=6,
        cat_features=cat_cols if len(cat_cols) > 0 else None,
        random_seed=42 + fold,
        verbose=0
    )
    model_cb.fit(
        X_tr_cb, y_tr_cb,
        eval_set=(X_va_cb, y_va_cb),
        early_stopping_rounds=50
    )
    preds_cb += model_cb.predict(X_test) / kf.n_splits

final_preds = 0.5 * preds_lgb + 0.5 * preds_cb

if (y >= 0).all():
    final_preds = np.clip(final_preds, 0, None)

submission = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'credit_amount': final_preds
})

submission.to_csv('predictions.csv', index=False)