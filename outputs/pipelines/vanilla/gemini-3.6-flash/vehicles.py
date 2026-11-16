import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

train_path = "/path/to/user/workspace/ds_failures/datasets/strable/vehicles/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/vehicles/data/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = "youSaveSpend"
y_train = train[target_col].values
X_train = train.drop(columns=[target_col]).copy()
X_test = test.copy()

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

for col in list(X_train.columns):
    if col not in X_test.columns:
        X_train.drop(columns=[col], inplace=True)
        continue
    
    if pd.api.types.is_datetime64_any_dtype(X_train[col]):
        for df in [X_train, X_test]:
            df[col + '_year'] = df[col].dt.year
            df[col + '_month'] = df[col].dt.month
            df[col + '_day'] = df[col].dt.day
            df[col + '_dayofweek'] = df[col].dt.dayofweek
        X_train.drop(columns=[col], inplace=True)
        X_test.drop(columns=[col], inplace=True)
        continue

    if X_train[col].dtype == 'bool':
        X_train[col] = X_train[col].astype(int)
        X_test[col] = X_test[col].astype(int)
        continue

for col in list(X_train.columns):
    if X_train[col].nunique(dropna=False) <= 1:
        X_train.drop(columns=[col], inplace=True)
        X_test.drop(columns=[col], inplace=True)

cat_cols = []
for col in X_train.columns:
    if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category':
        cat_cols.append(col)

X_train_lgb = X_train.copy()
X_test_lgb = X_test.copy()
for col in cat_cols:
    X_train_lgb[col] = X_train_lgb[col].astype(str).fillna('missing').astype('category')
    X_test_lgb[col] = X_test_lgb[col].astype(str).fillna('missing').astype('category')

X_train_cb = X_train.copy()
X_test_cb = X_test.copy()
for col in cat_cols:
    X_train_cb[col] = X_train_cb[col].astype(str).fillna('missing')
    X_test_cb[col] = X_test_cb[col].astype(str).fillna('missing')

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))

for train_idx, val_idx in kf.split(X_train_lgb):
    X_tr_lgb, y_tr = X_train_lgb.iloc[train_idx], y_train[train_idx]
    X_va_lgb, y_va = X_train_lgb.iloc[val_idx], y_train[val_idx]
    
    lgb_model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )
    lgb_model.fit(X_tr_lgb, y_tr)
    lgb_preds += lgb_model.predict(X_test_lgb) / 5.0

    X_tr_cb = X_train_cb.iloc[train_idx]
    X_va_cb = X_train_cb.iloc[val_idx]
    
    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        cat_features=cat_cols,
        random_seed=42,
        verbose=0
    )
    cb_model.fit(X_tr_cb, y_tr, eval_set=(X_va_cb, y_va), early_stopping_rounds=50, verbose=False)
    cb_preds += cb_model.predict(X_test_cb) / 5.0

final_preds = 0.5 * lgb_preds + 0.5 * cb_preds

df_out = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'youSaveSpend': final_preds
})

df_out.to_csv('predictions.csv', index=False)