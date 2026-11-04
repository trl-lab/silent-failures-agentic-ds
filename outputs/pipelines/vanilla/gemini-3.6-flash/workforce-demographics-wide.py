import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/workforce-demographics-wide/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/workforce-demographics-wide/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = 'Female (%)'
y = train[target_col].values
X = train.drop(columns=[target_col])
X_test = test.drop(columns=[target_col], errors='ignore')

common_cols = [c for c in X.columns if c in X_test.columns]
X = X[common_cols].copy()
X_test = X_test[common_cols].copy()

cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X.select_dtypes(include=['number']).columns.tolist()

for c in cat_cols:
    X[c] = X[c].astype(str)
    X_test[c] = X_test[c].astype(str)
    
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X[c] = oe.fit_transform(X[[c]])
    X_test[c] = oe.transform(X_test[[c]])

for c in num_cols:
    X[c] = pd.to_numeric(X[c], errors='coerce')
    X_test[c] = pd.to_numeric(X_test[c], errors='coerce')

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))

min_val = np.nanmin(y)
max_val = np.nanmax(y)

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]
    
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        random_state=42 + fold,
        verbosity=-1,
        n_jobs=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_preds += model_lgb.predict(X_test) / kf.n_splits
    
    model_cb = cb.CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        random_seed=42 + fold,
        verbose=0
    )
    model_cb.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50
    )
    cb_preds += model_cb.predict(X_test) / kf.n_splits
    
    model_xgb = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=50
    )
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    xgb_preds += model_xgb.predict(X_test) / kf.n_splits

final_preds = 0.4 * lgb_preds + 0.3 * cb_preds + 0.3 * xgb_preds

if not np.isnan(min_val) and not np.isnan(max_val):
    final_preds = np.clip(final_preds, min_val, max_val)

preds_df = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'Female (%)': final_preds
})

preds_df.to_csv('predictions.csv', index=False)