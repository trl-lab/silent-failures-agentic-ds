import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

train_path = "/path/to/user/workspace/ds_failures/datasets/strable/schools/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/schools/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Enrollment_FT_Teachers_Ratio"

train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)

y_train = train_df[target_col].values.astype(np.float32)
X_train = train_df.drop(columns=[target_col])

if target_col in test_df.columns:
    X_test = test_df.drop(columns=[target_col])
else:
    X_test = test_df.copy()

cols_to_drop = []
for col in X_train.columns:
    if col.lower() in ['id', 'index', 'objectid', 'fid']:
        cols_to_drop.append(col)

X_train = X_train.drop(columns=cols_to_drop, errors='ignore')
X_test = X_test.drop(columns=cols_to_drop, errors='ignore')

for col in list(X_train.columns):
    if X_train[col].dtype == 'object' or pd.api.types.is_datetime64_any_dtype(X_train[col]):
        try:
            sample = X_train[col].dropna().astype(str).head(100)
            if sample.str.contains('^[0-9]{4}-[0-9]{2}-[0-9]{2}').all() or sample.str.contains('^[0-9]{2}/[0-9]{2}/[0-9]{4}').all():
                dt_train = pd.to_datetime(X_train[col], errors='coerce')
                dt_test = pd.to_datetime(X_test[col], errors='coerce')
                
                X_train[f'{col}_year'] = dt_train.dt.year
                X_train[f'{col}_month'] = dt_train.dt.month
                X_train[f'{col}_day'] = dt_train.dt.day
                X_train[f'{col}_dayofweek'] = dt_train.dt.dayofweek
                
                X_test[f'{col}_year'] = dt_test.dt.year
                X_test[f'{col}_month'] = dt_test.dt.month
                X_test[f'{col}_day'] = dt_test.dt.day
                X_test[f'{col}_dayofweek'] = dt_test.dt.dayofweek
                
                X_train = X_train.drop(columns=[col])
                X_test = X_test.drop(columns=[col])
        except Exception:
            pass

cat_cols = []
num_cols = []

for col in X_train.columns:
    if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category':
        cat_cols.append(col)
    else:
        num_cols.append(col)

for col in cat_cols:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)
    
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train[col] = oe.fit_transform(X_train[[col]])
    X_test[col] = oe.transform(X_test[[col]])

for col in num_cols:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_oof = np.zeros(len(train_df))
lgb_preds = np.zeros(len(test_df))

cat_oof = np.zeros(len(train_df))
cat_preds = np.zeros(len(test_df))

xgb_oof = np.zeros(len(train_df))
xgb_preds = np.zeros(len(test_df))

min_target = y_train.min()

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.04,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_oof[val_idx] = lgb_model.predict(X_va)
    lgb_preds += lgb_model.predict(X_test) / kf.n_splits

    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.04,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    cat_model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cat_oof[val_idx] = cat_model.predict(X_va)
    cat_preds += cat_model.predict(X_test) / kf.n_splits

    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.04,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        tree_method='hist',
        early_stopping_rounds=50
    )
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    xgb_oof[val_idx] = xgb_model.predict(X_va)
    xgb_preds += xgb_model.predict(X_test) / kf.n_splits

def objective(weights):
    w1, w2, w3 = weights
    pred = w1 * lgb_oof + w2 * cat_oof + w3 * xgb_oof
    return np.sqrt(mean_squared_error(y_train, pred))

res = minimize(objective, [1/3, 1/3, 1/3], method='Nelder-Mead')
w1, w2, w3 = res.x
total = w1 + w2 + w3
if total != 0:
    w1, w2, w3 = w1/total, w2/total, w3/total

final_preds = w1 * lgb_preds + w2 * cat_preds + w3 * xgb_preds

if min_target >= 0:
    final_preds = np.clip(final_preds, a_min=0, a_max=None)

sub = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'Enrollment_FT_Teachers_Ratio': final_preds
})

sub.to_csv('predictions.csv', index=False)