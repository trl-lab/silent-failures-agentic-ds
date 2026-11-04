import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

train_path = '/path/to/user/workspace/ds_failures/datasets/openml/ilpd/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/ilpd/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Class'
X = train_df.drop(columns=[target_col]).copy()
y = train_df[target_col].copy()
X_test = test_df.copy()

le = LabelEncoder()
y_encoded = le.fit_transform(y.astype(str))

cat_cols = ['V2']
for col in cat_cols:
    X[col] = X[col].astype('category').cat.codes
    X_test[col] = X_test[col].astype('category').cat.codes

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    X_tr, y_tr = X.iloc[train_idx], y_encoded[train_idx]
    X_va, y_va = X.iloc[val_idx], y_encoded[val_idx]
    
    # Model 1: LightGBM
    lgb = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        random_state=42 + fold,
        verbose=-1
    )
    lgb.fit(X_tr, y_tr)
    pred_lgb = lgb.predict_proba(X_test)[:, 1]
    
    # Model 2: CatBoost
    cb = CatBoostClassifier(
        iterations=300,
        learning_rate=0.03,
        depth=4,
        random_seed=42 + fold,
        verbose=False
    )
    cb.fit(X_tr, y_tr)
    pred_cb = cb.predict_proba(X_test)[:, 1]
    
    # Model 3: XGBoost
    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        random_state=42 + fold,
        eval_metric='logloss'
    )
    xgb.fit(X_tr, y_tr)
    pred_xgb = xgb.predict_proba(X_test)[:, 1]
    
    fold_pred = (pred_lgb + pred_cb + pred_xgb) / 3.0
    test_preds += fold_pred / n_splits

submission = pd.DataFrame({
    'id': range(len(test_preds)),
    'Class': test_preds
})

submission.to_csv('predictions.csv', index=False)