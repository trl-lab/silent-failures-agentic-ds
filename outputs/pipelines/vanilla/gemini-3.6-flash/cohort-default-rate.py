import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/cohort-default-rate/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/cohort-default-rate/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Average or Greater than 30'

y_raw = train_df[target_col]
if y_raw.dtype == 'bool':
    y = y_raw.astype(int).values
else:
    uniques = y_raw.dropna().unique()
    if len(uniques) == 2:
        pos_found = False
        for u in uniques:
            s = str(u).lower()
            if any(k in s for k in ['1', 'true', 'yes', 'greater', 'average', 'above', 'high', '>=30', '>30', '30']):
                y = (y_raw == u).astype(int).values
                pos_found = True
                break
        if not pos_found:
            y = (y_raw == uniques[1]).astype(int).values
    else:
        y = y_raw.astype(int).values

X_train = train_df.drop(columns=[target_col])
X_test = test_df.drop(columns=[target_col], errors='ignore')

cols_to_drop = []
for col in X_train.columns:
    if col.lower() in ['id', 'row_id', 'index', 'unnamed: 0']:
        if np.array_equal(X_train[col].values, np.arange(len(X_train))):
            cols_to_drop.append(col)

if cols_to_drop:
    X_train = X_train.drop(columns=cols_to_drop)
    X_test = X_test.drop(columns=cols_to_drop, errors='ignore')

cat_cols = []
num_cols = []

for col in X_train.columns:
    if pd.api.types.is_datetime64_any_dtype(X_train[col]):
        X_train[col] = X_train[col].astype('int64') // 10**9
        X_test[col] = X_test[col].astype('int64') // 10**9
        num_cols.append(col)
    elif X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category' or X_train[col].dtype == 'bool':
        cat_cols.append(col)
    else:
        num_cols.append(col)

X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()

for col in cat_cols:
    X_train_encoded[col] = X_train_encoded[col].astype(str)
    X_test_encoded[col] = X_test_encoded[col].astype(str)

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
if len(cat_cols) > 0:
    X_train_encoded[cat_cols] = encoder.fit_transform(X_train_encoded[cat_cols])
    X_test_encoded[cat_cols] = encoder.transform(X_test_encoded[cat_cols])

for col in X_train_encoded.columns:
    X_train_encoded[col] = pd.to_numeric(X_train_encoded[col], errors='coerce')
    X_test_encoded[col] = pd.to_numeric(X_test_encoded[col], errors='coerce')

X_train_cat = X_train.copy()
X_test_cat = X_test.copy()
for col in X_train.columns:
    if col in cat_cols:
        X_train_cat[col] = X_train_cat[col].astype(str)
        X_test_cat[col] = X_test_cat[col].astype(str)
    else:
        X_train_cat[col] = pd.to_numeric(X_train_cat[col], errors='coerce')
        X_test_cat[col] = pd.to_numeric(X_test_cat[col], errors='coerce')

cat_features_idx = [X_train_cat.columns.get_loc(c) for c in cat_cols]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_encoded, y)):
    X_tr, y_tr = X_train_encoded.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train_encoded.iloc[val_idx], y[val_idx]

    model_lgb = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbosity=-1,
        n_jobs=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_preds += model_lgb.predict_proba(X_test_encoded)[:, 1] / skf.n_splits

    model_xgb = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        random_state=42 + fold,
        tree_method='hist',
        early_stopping_rounds=50,
        n_jobs=-1
    )
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    xgb_preds += model_xgb.predict_proba(X_test_encoded)[:, 1] / skf.n_splits

    X_tr_c = X_train_cat.iloc[train_idx]
    X_va_c = X_train_cat.iloc[val_idx]

    model_cat = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=False,
        thread_count=-1
    )
    model_cat.fit(
        X_tr_c, y_tr,
        eval_set=(X_va_c, y_va),
        cat_features=cat_features_idx if len(cat_features_idx) > 0 else None,
        early_stopping_rounds=50
    )
    cat_preds += model_cat.predict_proba(X_test_cat)[:, 1] / skf.n_splits

final_preds = (lgb_preds + xgb_preds + cat_preds) / 3.0

sub = pd.DataFrame({
    'id': np.arange(len(test_df)),
    target_col: final_preds
})

sub.to_csv('predictions.csv', index=False)