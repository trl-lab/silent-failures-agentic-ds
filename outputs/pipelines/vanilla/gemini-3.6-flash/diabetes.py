import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

train_path = '/path/to/user/workspace/ds_failures/datasets/openml/diabetes/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/diabetes/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

y = (train['class'].astype(str).str.lower().str.contains('pos|1|true')).astype(int)
X = train.drop(columns=['class'])
X_test = test.drop(columns=['class'], errors='ignore')

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    col_lower = str(col).lower()
    if not any(k in col_lower for k in ['preg', 'age', 'ped', 'class']):
        if (X[col] == 0).sum() > 0:
            X.loc[X[col] == 0, col] = np.nan
            X_test.loc[X_test[col] == 0, col] = np.nan

combined = pd.concat([X, X_test], axis=0)
combined = pd.get_dummies(combined, drop_first=True)
X_proc = combined.iloc[:len(train)].copy()
X_test_proc = combined.iloc[len(train):].copy()

n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds = np.zeros(len(X_test_proc))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_proc, y)):
    X_tr, y_tr = X_proc.iloc[train_idx], y.iloc[train_idx]
    
    m1 = LGBMClassifier(n_estimators=200, learning_rate=0.03, max_depth=4, num_leaves=15, 
                        subsample=0.8, colsample_bytree=0.8, random_state=42 + fold, verbose=-1)
    m1.fit(X_tr, y_tr)
    p1 = m1.predict_proba(X_test_proc)[:, 1]
    
    m2 = XGBClassifier(n_estimators=200, learning_rate=0.03, max_depth=3, subsample=0.8, 
                       colsample_bytree=0.8, random_state=42 + fold, eval_metric='logloss')
    m2.fit(X_tr, y_tr)
    p2 = m2.predict_proba(X_test_proc)[:, 1]
    
    m3 = CatBoostClassifier(iterations=300, learning_rate=0.03, depth=4, random_seed=42 + fold, verbose=0)
    m3.fit(X_tr, y_tr)
    p3 = m3.predict_proba(X_test_proc)[:, 1]
    
    m4 = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.03, max_depth=4, l2_regularization=1.0, random_state=42 + fold)
    m4.fit(X_tr, y_tr)
    p4 = m4.predict_proba(X_test_proc)[:, 1]
    
    m5 = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=0.5, random_state=42 + fold))
    m5.fit(X_tr, y_tr)
    p5 = m5.predict_proba(X_test_proc)[:, 1]
    
    m6 = make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42 + fold))
    m6.fit(X_tr, y_tr)
    p6 = m6.predict_proba(X_test_proc)[:, 1]

    fold_pred = (p1 * 0.2 + p2 * 0.2 + p3 * 0.2 + p4 * 0.2 + p5 * 0.1 + p6 * 0.1)
    test_preds += fold_pred / n_splits

sub = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'class': test_preds
})
sub.to_csv('predictions.csv', index=False)