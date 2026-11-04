import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/conflict-events_wide/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/conflict-events_wide/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

train = train.dropna(subset=['fatalities']).reset_index(drop=True)

y = train['fatalities'].values
X = train.drop(columns=['fatalities'])
X_test = test.copy()

for col in ['id', 'ID', 'Id']:
    if col in X.columns:
        X = X.drop(columns=[col])
        X_test = X_test.drop(columns=[col])

for col in list(X.columns):
    if X[col].dtype == 'object' or isinstance(X[col].dtype, pd.CategoricalDtype):
        try:
            sample = X[col].dropna()
            if len(sample) > 0 and pd.to_datetime(sample.iloc[:100], errors='coerce').notna().mean() > 0.8:
                dt_tr = pd.to_datetime(X[col], errors='coerce')
                dt_te = pd.to_datetime(X_test[col], errors='coerce')
                
                X[f'{col}_year'] = dt_tr.dt.year
                X[f'{col}_month'] = dt_tr.dt.month
                X[f'{col}_day'] = dt_tr.dt.day
                X[f'{col}_dayofweek'] = dt_tr.dt.dayofweek
                
                X_test[f'{col}_year'] = dt_te.dt.year
                X_test[f'{col}_month'] = dt_te.dt.month
                X_test[f'{col}_day'] = dt_te.dt.day
                X_test[f'{col}_dayofweek'] = dt_te.dt.dayofweek
                
                X = X.drop(columns=[col])
                X_test = X_test.drop(columns=[col])
        except Exception:
            pass

cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    X[col] = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)
    
    unique_cats = pd.concat([X[col], X_test[col]]).unique()
    cat_map = {cat: i for i, cat in enumerate(unique_cats)}
    
    X[col] = X[col].map(cat_map).fillna(-1).astype(int)
    X_test[col] = X_test[col].map(cat_map).fillna(-1).astype(int)

for col in X.columns:
    if not np.issubdtype(X[col].dtype, np.number):
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]
    
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        random_state=42 + fold,
        n_jobs=-1,
        verbosity=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_preds += lgb_model.predict(X_test) / kf.n_splits
    
    cb_model = cb.CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=50
    )
    cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    cb_preds += cb_model.predict(X_test) / kf.n_splits
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=50,
        tree_method='hist'
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    xgb_preds += xgb_model.predict(X_test) / kf.n_splits

final_preds = 0.4 * lgb_preds + 0.3 * cb_preds + 0.3 * xgb_preds
final_preds = np.clip(final_preds, 0, None)

submission = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'fatalities': final_preds
})
submission.to_csv('predictions.csv', index=False)