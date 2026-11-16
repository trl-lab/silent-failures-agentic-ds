import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from scipy.optimize import minimize

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/aca-federal-upper-limits-wide/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/aca-federal-upper-limits-wide/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = 'ACA FUL'

train = train.dropna(subset=[target_col]).reset_index(drop=True)
y = train[target_col].values.astype(np.float64)

n_test = len(test)

X_train = train.drop(columns=[target_col])
X_test = test.drop(columns=[target_col], errors='ignore')

for col in list(X_train.columns):
    if col.lower() in ['id', 'index']:
        if col in X_train.columns:
            X_train = X_train.drop(columns=[col])
        if col in X_test.columns:
            X_test = X_test.drop(columns=[col])

feature_cols = [c for c in X_train.columns if c in X_test.columns]
X_train = X_train[feature_cols].copy()
X_test = X_test[feature_cols].copy()

for col in feature_cols:
    if pd.api.types.is_datetime64_any_dtype(X_train[col]):
        X_train[col + '_year'] = X_train[col].dt.year
        X_train[col + '_month'] = X_train[col].dt.month
        X_train[col + '_day'] = X_train[col].dt.day
        X_train[col + '_dayofweek'] = X_train[col].dt.dayofweek
        X_train = X_train.drop(columns=[col])

        X_test[col + '_year'] = X_test[col].dt.year
        X_test[col + '_month'] = X_test[col].dt.month
        X_test[col + '_day'] = X_test[col].dt.day
        X_test[col + '_dayofweek'] = X_test[col].dt.dayofweek
        X_test = X_test.drop(columns=[col])

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

if cat_cols:
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train[cat_cols] = oe.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = oe.transform(X_test[cat_cols])

for col in num_cols:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

clean_cols = [f"f_{i}" for i in range(X_train.shape[1])]
X_train.columns = clean_cols
X_test.columns = clean_cols

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_oof = np.zeros(len(X_train))
cat_oof = np.zeros(len(X_train))
xgb_oof = np.zeros(len(X_train))

lgb_preds = np.zeros(n_test)
cat_preds = np.zeros(n_test)
xgb_preds = np.zeros(n_test)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]
    
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
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
    lgb_oof[val_idx] = model_lgb.predict(X_va)
    lgb_preds += model_lgb.predict(X_test) / n_splits
    
    model_cat = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.04,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    model_cat.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50
    )
    cat_oof[val_idx] = model_cat.predict(X_va)
    cat_preds += model_cat.predict(X_test) / n_splits
    
    model_xgb = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        random_state=42 + fold,
        early_stopping_rounds=50,
        n_jobs=-1
    )
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    xgb_oof[val_idx] = model_xgb.predict(X_va)
    xgb_preds += model_xgb.predict(X_test) / n_splits

def loss_func(weights):
    w1, w2, w3 = weights
    pred = w1 * lgb_oof + w2 * cat_oof + w3 * xgb_oof
    return np.sqrt(mean_squared_error(y, pred))

res = minimize(loss_func, [1/3, 1/3, 1/3], method='Nelder-Mead')
w = res.x
if np.sum(w) != 0:
    w = w / np.sum(w)
else:
    w = np.array([1/3, 1/3, 1/3])

final_preds = w[0] * lgb_preds + w[1] * cat_preds + w[2] * xgb_preds

if y.min() >= 0:
    final_preds = np.clip(final_preds, a_min=0, a_max=None)

submission = pd.DataFrame({
    'id': np.arange(n_test),
    'ACA FUL': final_preds
})

submission.to_csv('predictions.csv', index=False)