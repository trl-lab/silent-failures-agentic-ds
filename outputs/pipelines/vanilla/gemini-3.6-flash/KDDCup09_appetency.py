import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

train_path = '/path/to/user/workspace/ds_failures/datasets/openml/KDDCup09_appetency/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/KDDCup09_appetency/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = 'APPETENCY'
y_raw = train[target_col]

val_counts = y_raw.value_counts()
unique_vals = set(val_counts.index)

if 1 in unique_vals and -1 in unique_vals:
    y = (y_raw == 1).astype(int)
elif '1' in unique_vals and '-1' in unique_vals:
    y = (y_raw == '1').astype(int)
elif 1 in unique_vals and 0 in unique_vals:
    y = (y_raw == 1).astype(int)
elif '1' in unique_vals and '0' in unique_vals:
    y = (y_raw == '1').astype(int)
elif True in unique_vals and False in unique_vals:
    y = (y_raw == True).astype(int)
else:
    pos_val = val_counts.idxmin()
    y = (y_raw == pos_val).astype(int)

X_train = train.drop(columns=[target_col]).copy()
X_test = test.copy()
if target_col in X_test.columns:
    X_test = X_test.drop(columns=[target_col])

X_train['null_count'] = X_train.isnull().sum(axis=1)
X_test['null_count'] = X_test.isnull().sum(axis=1)

cols_to_drop = []
for c in X_train.columns:
    if c == 'null_count':
        continue
    tr_u = X_train[c].dropna().nunique()
    te_u = X_test[c].dropna().nunique()
    if tr_u == 0 or (tr_u == 1 and te_u <= 1):
        cols_to_drop.append(c)

X_train = X_train.drop(columns=cols_to_drop)
X_test = X_test.drop(columns=cols_to_drop)

features = X_train.columns.tolist()
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = [c for c in features if c not in cat_cols]

for c in num_cols:
    X_train[c] = pd.to_numeric(X_train[c], errors='coerce')
    X_test[c] = pd.to_numeric(X_test[c], errors='coerce')

for c in cat_cols:
    X_train[c] = X_train[c].astype(str).fillna('MISSING')
    X_test[c] = X_test[c].astype(str).fillna('MISSING')

if len(cat_cols) > 0:
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.float32)
    X_train[cat_cols] = oe.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = oe.transform(X_test[cat_cols])

X_tr = X_train.astype(np.float32)
X_te = X_test.astype(np.float32)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_te))
xgb_preds = np.zeros(len(X_te))
cb_preds = np.zeros(len(X_te))

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_tr, y)):
    X_tr_fold, y_tr_fold = X_tr.iloc[trn_idx], y.iloc[trn_idx]
    X_va_fold, y_va_fold = X_tr.iloc[val_idx], y.iloc[val_idx]

    model_lgb = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        colsample_bytree=0.7,
        subsample=0.8,
        min_child_samples=50,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    model_lgb.fit(
        X_tr_fold, y_tr_fold,
        eval_set=[(X_va_fold, y_va_fold)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_preds += model_lgb.predict_proba(X_te)[:, 1] / skf.n_splits

    model_xgb = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        colsample_bytree=0.7,
        subsample=0.8,
        min_child_weight=10,
        tree_method='hist',
        random_state=42 + fold,
        n_jobs=-1,
        eval_metric='auc',
        early_stopping_rounds=50
    )
    model_xgb.fit(
        X_tr_fold, y_tr_fold,
        eval_set=[(X_va_fold, y_va_fold)],
        verbose=False
    )
    xgb_preds += model_xgb.predict_proba(X_te)[:, 1] / skf.n_splits

    model_cb = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.04,
        depth=6,
        eval_metric='AUC',
        random_seed=42 + fold,
        verbose=False,
        early_stopping_rounds=50
    )
    model_cb.fit(
        X_tr_fold, y_tr_fold,
        eval_set=(X_va_fold, y_va_fold)
    )
    cb_preds += model_cb.predict_proba(X_te)[:, 1] / skf.n_splits

final_preds = 0.4 * lgb_preds + 0.3 * xgb_preds + 0.3 * cb_preds

predictions = pd.DataFrame({
    'id': np.arange(len(X_te)),
    'APPETENCY': final_preds
})

predictions.to_csv('predictions.csv', index=False)