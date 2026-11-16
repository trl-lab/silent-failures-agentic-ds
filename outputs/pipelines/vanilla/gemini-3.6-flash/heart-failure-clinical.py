import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

train_path = "/path/to/user/workspace/ds_failures/datasets/uci/heart-failure-clinical/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/uci/heart-failure-clinical/data/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = [c for c in train.columns if c.upper() == 'DEATH_EVENT'][0]

X_train = train.drop(columns=[target_col]).copy()
y_train = train[target_col].values

feature_cols = [c for c in X_train.columns if c in test.columns]
X_train = X_train[feature_cols]
X_test = test[feature_cols].copy()

for col in feature_cols:
    if X_train[col].dtype == 'object' or str(X_train[col].dtype) == 'category':
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

X_train_num = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
X_test_num = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train_num.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train_num.iloc[val_idx], y_train[val_idx]
    
    lgb = LGBMClassifier(
        objective='binary',
        metric='auc',
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42 + fold,
        verbose=-1,
        n_estimators=100
    )
    lgb.fit(X_tr, y_tr)
    
    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42 + fold,
        n_estimators=100
    )
    xgb.fit(X_tr, y_tr)
    
    cb = CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='AUC',
        depth=4,
        learning_rate=0.03,
        l2_leaf_reg=3.0,
        random_seed=42 + fold,
        verbose=False,
        iterations=150
    )
    cb.fit(X_tr, y_tr)
    
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, random_state=42 + fold))
    lr.fit(X_tr, y_tr)
    
    et = ExtraTreesClassifier(n_estimators=100, max_depth=5, min_samples_split=5, random_state=42 + fold)
    et.fit(X_tr, y_tr)
    
    pred_lgb = lgb.predict_proba(X_test_num)[:, 1]
    pred_xgb = xgb.predict_proba(X_test_num)[:, 1]
    pred_cb = cb.predict_proba(X_test_num)[:, 1]
    pred_lr = lr.predict_proba(X_test_num)[:, 1]
    pred_et = et.predict_proba(X_test_num)[:, 1]
    
    fold_pred = (pred_lgb * 0.25 + pred_xgb * 0.25 + pred_cb * 0.25 + pred_lr * 0.15 + pred_et * 0.10)
    test_preds += fold_pred / skf.n_splits

sub = pd.DataFrame({
    'id': np.arange(len(test)),
    'DEATH_EVENT': test_preds
})

sub.to_csv('predictions.csv', index=False)