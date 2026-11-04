import warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/california-houses/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/california-houses/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = 'Total interior livable area'

if target_col in test.columns:
    test = test.drop(columns=[target_col])

y = train[target_col].values
train = train.drop(columns=[target_col])

n_train = len(train)
df_all = pd.concat([train, test], axis=0, ignore_index=True)

for col in df_all.columns:
    if df_all[col].dtype == 'object':
        df_all[col] = df_all[col].apply(lambda x: str(x) if x is not None else np.nan)

for col in df_all.columns:
    if pd.api.types.is_datetime64_any_dtype(df_all[col]):
        df_all[f'{col}_year'] = df_all[col].dt.year
        df_all[f'{col}_month'] = df_all[col].dt.month
        df_all[f'{col}_day'] = df_all[col].dt.day
        df_all[f'{col}_dayofweek'] = df_all[col].dt.dayofweek
        df_all = df_all.drop(columns=[col])

cat_cols = df_all.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
for col in cat_cols:
    freq = df_all[col].value_counts().to_dict()
    df_all[f'{col}_freq'] = df_all[col].map(freq).fillna(0)
    df_all[col] = df_all[col].fillna('missing').astype(str).astype('category')

X_train = df_all.iloc[:n_train].reset_index(drop=True)
X_test = df_all.iloc[n_train:].reset_index(drop=True)

valid_mask = ~np.isnan(y)
if not np.all(valid_mask):
    X_train = X_train.iloc[valid_mask].reset_index(drop=True)
    y = y[valid_mask]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

test_preds_lgb = np.zeros(len(X_test))
test_preds_cat = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))

cat_feature_names = [col for col in X_train.columns if X_train[col].dtype.name == 'category']

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]
    
    model_lgb = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    test_preds_lgb += model_lgb.predict(X_test) / 5.0
    
    X_tr_cb = X_tr.copy()
    X_va_cb = X_va.copy()
    X_te_cb = X_test.copy()
    for col in cat_feature_names:
        X_tr_cb[col] = X_tr_cb[col].astype(str)
        X_va_cb[col] = X_va_cb[col].astype(str)
        X_te_cb[col] = X_te_cb[col].astype(str)
        
    model_cat = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42 + fold,
        cat_features=cat_feature_names,
        verbose=0
    )
    model_cat.fit(
        X_tr_cb, y_tr,
        eval_set=(X_va_cb, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    test_preds_cat += model_cat.predict(X_te_cb) / 5.0
    
    model_xgb = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        tree_method='hist',
        enable_categorical=True,
        early_stopping_rounds=50,
        random_state=42 + fold,
        n_jobs=-1
    )
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    test_preds_xgb += model_xgb.predict(X_test) / 5.0

final_preds = 0.4 * test_preds_lgb + 0.4 * test_preds_cat + 0.2 * test_preds_xgb
final_preds = np.clip(final_preds, a_min=0, a_max=None)

sub = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'Total interior livable area': final_preds
})
sub.to_csv('predictions.csv', index=False)