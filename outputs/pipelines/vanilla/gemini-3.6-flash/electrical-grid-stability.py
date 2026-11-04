import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

train_path = '/path/to/user/workspace/ds_failures/datasets/uci/electrical-grid-stability/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/electrical-grid-stability/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'stabf'
y_raw = train_df[target_col]

if y_raw.dtype == 'object' or isinstance(y_raw.dtype, pd.CategoricalDtype):
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))
else:
    y = y_raw.values.astype(int)

drop_cols = [target_col, 'stab', 'id']
features = [c for c in train_df.columns if c in test_df.columns and c not in drop_cols]

X = train_df[features].copy()
X_test = test_df[features].copy()

for col in features:
    if X[col].isnull().any() or X_test[col].isnull().any():
        med = X[col].median()
        X[col] = X[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)

test_preds = np.zeros(len(test_df))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    
    model_lgb = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    model_lgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    model_xgb = xgb.XGBClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        random_state=42 + fold,
        n_jobs=-1,
        eval_metric='logloss',
        early_stopping_rounds=50
    )
    model_xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    model_cb = cb.CatBoostClassifier(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=50
    )
    model_cb.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=False
    )
    
    p_lgb = model_lgb.predict_proba(X_test)[:, 1]
    p_xgb = model_xgb.predict_proba(X_test)[:, 1]
    p_cb = model_cb.predict_proba(X_test)[:, 1]
    
    fold_pred = (p_lgb + p_xgb + p_cb) / 3.0
    test_preds += fold_pred / skf.n_splits

submission = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'stabf': test_preds
})

submission.to_csv('predictions.csv', index=False)