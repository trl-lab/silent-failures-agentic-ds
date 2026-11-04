import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/device-covid19serology/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/device-covid19serology/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = 'antibody_truth'

y_raw = train[target_col]
if y_raw.dtype == 'object' or str(y_raw.dtype) in ['category', 'bool']:
    val_map = {}
    unique_vals = set(y_raw.dropna().unique())
    for val in unique_vals:
        s_val = str(val).strip().lower()
        if s_val in ['1', 'true', 'positive', 'pos', '1.0', 'p']:
            val_map[val] = 1
        elif s_val in ['0', 'false', 'negative', 'neg', '0.0', 'n']:
            val_map[val] = 0
    if len(val_map) == len(unique_vals) and len(set(val_map.values())) == 2:
        y = y_raw.map(val_map).astype(int)
    else:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y_raw.astype(str)), index=train.index)
else:
    y = y_raw.astype(int)

X = train.drop(columns=[target_col])
X_test = test.copy()
if target_col in X_test.columns:
    X_test = X_test.drop(columns=[target_col])

common_cols = [c for c in X.columns if c in X_test.columns]
X = X[common_cols]
X_test = X_test[common_cols]

drop_cols = []
for c in X.columns:
    if X[c].nunique(dropna=False) <= 1:
        drop_cols.append(c)
    elif c.lower() in ['id', 'row_id', 'index', 'sample_id', 'patient_id', 'unnamed: 0']:
        drop_cols.append(c)

X = X.drop(columns=drop_cols)
X_test = X_test.drop(columns=drop_cols)

cat_cols = []
for c in X.columns:
    if X[c].dtype == 'object' or str(X[c].dtype) in ['category', 'bool']:
        cat_cols.append(c)

for c in cat_cols:
    le = LabelEncoder()
    s_train = X[c].astype(str).fillna('missing')
    s_test = X_test[c].astype(str).fillna('missing')
    combined = pd.concat([s_train, s_test], axis=0)
    le.fit(combined)
    X[c] = le.transform(s_train)
    X_test[c] = le.transform(s_test)

for c in X.columns:
    if c not in cat_cols:
        X[c] = pd.to_numeric(X[c], errors='coerce')
        X_test[c] = pd.to_numeric(X_test[c], errors='coerce')

X = X.astype(np.float32)
X_test = X_test.astype(np.float32)

min_samples = y.value_counts().min() if y.nunique() > 1 else 5
n_splits = max(2, min(5, min_samples))

if y.nunique() > 1 and min_samples >= 2:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(cv.split(X, y))
else:
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(cv.split(X))

test_preds_lgb = np.zeros(len(X_test))
test_preds_cb = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(splits):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    model_lgb = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    test_preds_lgb += model_lgb.predict_proba(X_test)[:, 1] / len(splits)
    
    model_cb = CatBoostClassifier(
        iterations=500,
        learning_rate=0.03,
        random_seed=42,
        verbose=0,
        early_stopping_rounds=50
    )
    model_cb.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va)
    )
    test_preds_cb += model_cb.predict_proba(X_test)[:, 1] / len(splits)

    model_xgb = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='auc',
        early_stopping_rounds=50,
        verbosity=0
    )
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    test_preds_xgb += model_xgb.predict_proba(X_test)[:, 1] / len(splits)

final_preds = 0.4 * test_preds_lgb + 0.3 * test_preds_cb + 0.3 * test_preds_xgb

submission = pd.DataFrame({
    'id': np.arange(len(test_preds_lgb)),
    'antibody_truth': final_preds
})

submission.to_csv('predictions.csv', index=False)