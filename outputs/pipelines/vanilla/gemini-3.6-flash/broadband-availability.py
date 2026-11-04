import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

train_path = "/path/to/user/workspace/ds_failures/datasets/strable/broadband-availability/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/broadband-availability/data/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = "Population for whom broadband available, 2019 (%)"

y = train[target_col].values
X = train.drop(columns=[target_col])
X_test = test.copy()
if target_col in X_test.columns:
    X_test = X_test.drop(columns=[target_col])

n_train = len(X)
df_all = pd.concat([X, X_test], axis=0, ignore_index=True)

cat_cols = []
num_cols = []
for col in df_all.columns:
    if df_all[col].dtype == 'object' or df_all[col].dtype.name == 'category':
        cat_cols.append(col)
        df_all[col] = df_all[col].fillna("missing").astype(str)
    else:
        num_cols.append(col)

X_all_lgb = df_all.copy()
for col in cat_cols:
    X_all_lgb[col] = X_all_lgb[col].astype("category")

X_lgb = X_all_lgb.iloc[:n_train].copy()
X_test_lgb = X_all_lgb.iloc[n_train:].copy()

X_cb = df_all.iloc[:n_train].copy()
X_test_cb = df_all.iloc[n_train:].copy()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_lgb, y)):
    X_tr_lgb, y_tr = X_lgb.iloc[train_idx], y[train_idx]
    X_va_lgb, y_va = X_lgb.iloc[val_idx], y[val_idx]
    
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
    lgb.fit(X_tr_lgb, y_tr)
    lgb_preds += lgb.predict(X_test_lgb) / kf.n_splits
    
    X_tr_cb = X_cb.iloc[train_idx]
    X_va_cb = X_cb.iloc[val_idx]
    
    cb = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.04,
        depth=6,
        cat_features=cat_cols,
        random_seed=42 + fold,
        verbose=0
    )
    cb.fit(X_tr_cb, y_tr, eval_set=(X_va_cb, y_va), early_stopping_rounds=100)
    cb_preds += cb.predict(X_test_cb) / kf.n_splits

final_preds = 0.5 * lgb_preds + 0.5 * cb_preds

min_y, max_y = np.min(y), np.max(y)
if min_y >= 0 and max_y <= 100:
    final_preds = np.clip(final_preds, min_y, max_y)

submission = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    target_col: final_preds
})

submission.to_csv('predictions.csv', index=False)