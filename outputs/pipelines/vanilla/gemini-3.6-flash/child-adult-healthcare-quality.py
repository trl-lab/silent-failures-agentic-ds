import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/child-adult-healthcare-quality/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/child-adult-healthcare-quality/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = 'State Rate'

if train[target_col].dtype == object or str(train[target_col].dtype) == 'string':
    train[target_col] = train[target_col].astype(str).str.rstrip('%').str.replace(',', '').str.strip()
    train[target_col] = pd.to_numeric(train[target_col], errors='coerce')

train = train.dropna(subset=[target_col]).reset_index(drop=True)
y_train = train[target_col].values

feature_cols = [c for c in train.columns if c != target_col and c.lower() != 'id']

def clean_col(s):
    if s.dtype == 'object' or str(s.dtype) == 'string':
        cleaned = s.astype(str).str.rstrip('%').str.replace(',', '').str.strip()
        converted = pd.to_numeric(cleaned, errors='coerce')
        non_null_orig = s.dropna().count()
        if non_null_orig > 0 and (converted.dropna().count() / non_null_orig) > 0.6:
            return converted
    return s

for col in feature_cols:
    train[col] = clean_col(train[col])
    test[col] = clean_col(test[col])

cat_cols = []
num_cols = []

for col in feature_cols:
    if train[col].dtype == 'object' or str(train[col].dtype) == 'string' or str(train[col].dtype) == 'category':
        cat_cols.append(col)
    else:
        num_cols.append(col)

for col in cat_cols:
    train[col] = train[col].fillna('Missing').astype(str)
    test[col] = test[col].fillna('Missing').astype(str)

for col in cat_cols:
    combined = pd.concat([train[col], test[col]])
    freq = combined.value_counts().to_dict()
    train[f'{col}_freq'] = train[col].map(freq).astype(float)
    test[f'{col}_freq'] = test[col].map(freq).astype(float)

for col in cat_cols:
    categories = sorted(list(set(train[col].unique()).union(set(test[col].unique()))))
    cat_type = pd.CategoricalDtype(categories=categories, ordered=True)
    train[col] = train[col].astype(cat_type).cat.codes
    test[col] = test[col].astype(cat_type).cat.codes

for cat in cat_cols[:5]:
    for num in num_cols:
        if train[num].isnull().sum() < len(train) * 0.9:
            combined = pd.concat([train[[cat, num]], test[[cat, num]]])
            grp_mean = combined.groupby(cat)[num].mean().to_dict()
            grp_std = combined.groupby(cat)[num].std().to_dict()
            train[f'{cat}_{num}_mean'] = train[cat].map(grp_mean)
            test[f'{cat}_{num}_mean'] = test[cat].map(grp_mean)
            train[f'{cat}_{num}_std'] = train[cat].map(grp_std)
            test[f'{cat}_{num}_std'] = test[cat].map(grp_std)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
global_mean = y_train.mean()

for col in cat_cols:
    te_col = f'{col}_te'
    train[te_col] = np.nan
    
    for tr_idx, val_idx in kf.split(train):
        tr_df = train.iloc[tr_idx]
        stats = tr_df.groupby(col)[target_col].agg(['count', 'mean'])
        smooth = (stats['count'] * stats['mean'] + 10 * global_mean) / (stats['count'] + 10)
        train.iloc[val_idx, train.columns.get_loc(te_col)] = train.iloc[val_idx][col].map(smooth).fillna(global_mean)
        
    stats = train.groupby(col)[target_col].agg(['count', 'mean'])
    smooth = (stats['count'] * stats['mean'] + 10 * global_mean) / (stats['count'] + 10)
    test[te_col] = test[col].map(smooth).fillna(global_mean)

X = train.drop(columns=[target_col]).copy()
if 'id' in [c.lower() for c in X.columns]:
    drop_col = [c for c in X.columns if c.lower() == 'id'][0]
    X = X.drop(columns=[drop_col])

X_test = test.copy()
if 'id' in [c.lower() for c in X_test.columns]:
    drop_col = [c for c in X_test.columns if c.lower() == 'id'][0]
    X_test = X_test.drop(columns=[drop_col])

X_test = X_test[X.columns]

test_preds_lgb = np.zeros(len(test))
test_preds_cat = np.zeros(len(test))
test_preds_xgb = np.zeros(len(test))

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'random_state': 42,
    'verbose': -1,
    'n_estimators': 1500
}

xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.03,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_estimators': 1500,
    'early_stopping_rounds': 50
}

cat_params = {
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'learning_rate': 0.04,
    'depth': 6,
    'random_seed': 42,
    'verbose': 0,
    'iterations': 1500
}

for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y_train)):
    X_tr, y_tr = X.iloc[tr_idx], y_train[tr_idx]
    X_val, y_val = X.iloc[val_idx], y_train[val_idx]
    
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    test_preds_lgb += model_lgb.predict(X_test) / kf.n_splits
    
    model_cat = CatBoostRegressor(**cat_params)
    model_cat.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50
    )
    test_preds_cat += model_cat.predict(X_test) / kf.n_splits
    
    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    test_preds_xgb += model_xgb.predict(X_test) / kf.n_splits

final_preds = (test_preds_lgb * 0.4) + (test_preds_cat * 0.4) + (test_preds_xgb * 0.2)

min_val = y_train.min()
max_val = y_train.max()
final_preds = np.clip(final_preds, min_val, max_val)

output = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'State Rate': final_preds
})
output.to_csv('predictions.csv', index=False)