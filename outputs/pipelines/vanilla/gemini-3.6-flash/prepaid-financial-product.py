import pandas as pd
import numpy as np
import warnings
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/prepaid-financial-product/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/prepaid-financial-product/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = 'current_status'
y_raw = train[target_col]

if y_raw.dtype == 'object' or y_raw.dtype.name == 'category' or y_raw.dtype == 'bool':
    le_target = LabelEncoder()
    y = le_target.fit_transform(y_raw.astype(str))
else:
    unique_vals = np.unique(y_raw.dropna())
    if len(unique_vals) == 2 and not np.array_equal(sorted(unique_vals), [0, 1]):
        le_target = LabelEncoder()
        y = le_target.fit_transform(y_raw)
    else:
        y = y_raw.astype(int).values

X = train.drop(columns=[target_col])
X_test = test.copy()

ignore_cols = []
for c in X.columns:
    if c.lower() in ['id', 'index', 'row_id', 'unnamed: 0']:
        ignore_cols.append(c)

X = X.drop(columns=ignore_cols, errors='ignore')
X_test = X_test.drop(columns=ignore_cols, errors='ignore')

datetime_cols = []
for c in X.columns:
    if pd.api.types.is_datetime64_any_dtype(X[c]):
        datetime_cols.append(c)
    elif X[c].dtype == 'object':
        try:
            sample = X[c].dropna().iloc[:100]
            if len(sample) > 0:
                parsed = pd.to_datetime(sample, errors='coerce')
                if parsed.notna().mean() > 0.8:
                    datetime_cols.append(c)
        except Exception:
            pass

for c in datetime_cols:
    X[c] = pd.to_datetime(X[c], errors='coerce')
    X_test[c] = pd.to_datetime(X_test[c], errors='coerce')
    
    for prefix, attr in [('year', 'year'), ('month', 'month'), ('day', 'day'), ('dayofweek', 'dayofweek'), ('hour', 'hour')]:
        X[f'{c}_{prefix}'] = getattr(X[c].dt, attr)
        X_test[f'{c}_{prefix}'] = getattr(X_test[c].dt, attr)
    
    X.drop(columns=[c], inplace=True)
    X_test.drop(columns=[c], inplace=True)

cat_cols = []
num_cols = []

for c in X.columns:
    if X[c].dtype == 'object' or X[c].dtype.name == 'category' or X[c].dtype == 'bool':
        cat_cols.append(c)
    else:
        num_cols.append(c)

for c in cat_cols:
    X[c] = X[c].astype(str).fillna('missing')
    X_test[c] = X_test[c].astype(str).fillna('missing')
    
    le = LabelEncoder()
    combined = pd.concat([X[c], X_test[c]], axis=0).astype(str)
    le.fit(combined)
    X[c] = le.transform(X[c].astype(str))
    X_test[c] = le.transform(X_test[c].astype(str))

for c in num_cols:
    X[c] = pd.to_numeric(X[c], errors='coerce')
    X_test[c] = pd.to_numeric(X_test[c], errors='coerce')

X_lgb = X.copy()
X_test_lgb = X_test.copy()
for c in cat_cols:
    X_lgb[c] = X_lgb[c].astype('category')
    X_test_lgb[c] = X_test_lgb[c].astype('category')

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'n_estimators': 1500,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'max_depth': 6
}

cat_params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'learning_rate': 0.04,
    'iterations': 1500,
    'random_seed': 42,
    'verbose': False,
    'thread_count': -1
}

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.03,
    'n_estimators': 1500,
    'random_state': 42,
    'n_jobs': -1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'max_depth': 6,
    'tree_method': 'hist',
    'early_stopping_rounds': 50
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr_lgb, y_tr = X_lgb.iloc[train_idx], y[train_idx]
    X_va_lgb, y_va = X_lgb.iloc[val_idx], y[val_idx]
    
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(
        X_tr_lgb, y_tr,
        eval_set=[(X_va_lgb, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_preds += model_lgb.predict_proba(X_test_lgb)[:, 1] / n_splits
    
    X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
    
    model_cat = CatBoostClassifier(**cat_params)
    model_cat.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cat_preds += model_cat.predict_proba(X_test)[:, 1] / n_splits
    
    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    xgb_preds += model_xgb.predict_proba(X_test)[:, 1] / n_splits

final_preds = 0.45 * lgb_preds + 0.35 * cat_preds + 0.20 * xgb_preds

sub = pd.DataFrame({
    'id': np.arange(len(test)),
    'current_status': final_preds
})

sub.to_csv('predictions.csv', index=False)