import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

train_path = "/path/to/user/workspace/ds_failures/datasets/uci/breast-cancer-wisconsin/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/uci/breast-cancer-wisconsin/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Diagnosis'
id_cols = ['ID', 'id', 'Id', 'iD']

drop_cols = [c for c in id_cols + [target_col] if c in train_df.columns]
feature_cols = [c for c in train_df.columns if c not in drop_cols]

X = train_df[feature_cols].copy()
y_raw = train_df[target_col]

test_drop = [c for c in id_cols + [target_col] if c in test_df.columns]
test_feature_cols = [c for c in test_df.columns if c not in test_drop]
X_test = test_df[test_feature_cols].copy()
X_test = X_test[X.columns]

if y_raw.dtype == object or str(y_raw.dtype) == 'category':
    classes = sorted(y_raw.unique())
    class_map = {c: i for i, c in enumerate(classes)}
    y = y_raw.map(class_map).values
else:
    classes = sorted(y_raw.unique())
    class_map = {c: i for i, c in enumerate(classes)}
    y = y_raw.map(class_map).values

imputer = SimpleImputer(strategy='median')
X_imp = imputer.fit_transform(X)
X_test_imp = imputer.transform(X_test)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)
X_test_scaled = scaler.transform(X_test_imp)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_imp, y)):
    X_tr, y_tr = X_imp[train_idx], y[train_idx]
    X_tr_sc = X_scaled[train_idx]
    
    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    lr.fit(X_tr_sc, y_tr)
    pred_lr = lr.predict_proba(X_test_scaled)[:, 1]
    
    et = ExtraTreesClassifier(n_estimators=200, max_depth=6, random_state=42)
    et.fit(X_tr, y_tr)
    pred_et = et.predict_proba(X_test_imp)[:, 1]
    
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    rf.fit(X_tr, y_tr)
    pred_rf = rf.predict_proba(X_test_imp)[:, 1]
    
    lgb = LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    lgb.fit(X_tr, y_tr)
    pred_lgb = lgb.predict_proba(X_test_imp)[:, 1]
    
    xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss')
    xgb.fit(X_tr, y_tr)
    pred_xgb = xgb.predict_proba(X_test_imp)[:, 1]
    
    cb = CatBoostClassifier(iterations=200, depth=4, learning_rate=0.03, verbose=0, random_seed=42)
    cb.fit(X_tr, y_tr)
    pred_cb = cb.predict_proba(X_test_imp)[:, 1]
    
    fold_pred = (pred_lr + pred_et + pred_rf + pred_lgb + pred_xgb + pred_cb) / 6.0
    test_preds += fold_pred / skf.n_splits

submission = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'Diagnosis': test_preds
})

submission.to_csv('predictions.csv', index=False)