import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

train_path = "/path/to/user/workspace/ds_failures/datasets/strable/china-overseas-finance-inventory/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/china-overseas-finance-inventory/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'installed_capacity'
train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)
y = train_df[target_col].values.astype(float)
X = train_df.drop(columns=[target_col])
X_test = test_df.drop(columns=[target_col], errors='ignore')

drop_cols = []
for col in X.columns:
    if col.lower() in ['id', 'index', 'unnamed: 0']:
        drop_cols.append(col)
if drop_cols:
    X = X.drop(columns=drop_cols)
    X_test = X_test.drop(columns=drop_cols, errors='ignore')

for col in list(X.columns):
    if pd.api.types.is_datetime64_any_dtype(X[col]):
        X[col + '_year'] = X[col].dt.year
        X[col + '_month'] = X[col].dt.month
        X[col + '_day'] = X[col].dt.day
        X[col + '_dayofweek'] = X[col].dt.dayofweek
        X = X.drop(columns=[col])
        
        if col in X_test.columns:
            X_test[col + '_year'] = X_test[col].dt.year
            X_test[col + '_month'] = X_test[col].dt.month
            X_test[col + '_day'] = X_test[col].dt.day
            X_test[col + '_dayofweek'] = X_test[col].dt.dayofweek
            X_test = X_test.drop(columns=[col], errors='ignore')

for col in list(X.columns):
    if X[col].dtype == 'object' or X[col].dtype.name == 'category' or X[col].dtype == 'bool':
        X[col] = X[col].astype(str)
        if col in X_test.columns:
            X_test[col] = X_test[col].astype(str)
        else:
            X_test[col] = 'missing'
            
        all_vals = pd.concat([X[col], X_test[col]], axis=0).dropna().unique()
        val_map = {val: i for i, val in enumerate(all_vals)}
        
        X[col] = X[col].map(val_map).fillna(-1).astype(int)
        X_test[col] = X_test[col].map(val_map).fillna(-1).astype(int)

X_test = X_test.reindex(columns=X.columns, fill_value=np.nan)

X = X.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))

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
    lgb_preds += lgb_model.predict(X_test) / 5.0
    
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=50
    )
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    xgb_preds += xgb_model.predict(X_test) / 5.0
    
    cb_model = CatBoostRegressor(
        loss_function='RMSE',
        eval_metric='RMSE',
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=50
    )
    cb_model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        verbose=False
    )
    cb_preds += cb_model.predict(X_test) / 5.0

final_preds = (lgb_preds + xgb_preds + cb_preds) / 3.0
final_preds = np.clip(final_preds, 0, None)

sub = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'installed_capacity': final_preds
})
sub.to_csv('predictions.csv', index=False)