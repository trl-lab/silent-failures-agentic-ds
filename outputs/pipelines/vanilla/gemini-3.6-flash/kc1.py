import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier

train_path = "/path/to/user/workspace/ds_failures/datasets/openml/kc1/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/openml/kc1/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "defects"
if target_col not in train_df.columns:
    if "problems" in train_df.columns:
        target_col = "problems"
    else:
        target_col = train_df.columns[-1]

y_raw = train_df[target_col]
if pd.api.types.is_numeric_dtype(y_raw):
    y = y_raw.astype(int)
else:
    y = y_raw.astype(str).str.strip().str.lower().isin(['true', '1', 't', 'yes']).astype(int)

X = train_df.drop(columns=[target_col], errors='ignore')
X_test = test_df.drop(columns=[target_col], errors='ignore')

for col in ['id', 'ID', 'Id']:
    if col in X.columns:
        X = X.drop(columns=[col])
    if col in X_test.columns:
        X_test = X_test.drop(columns=[col])

clean_cols = [re.sub(r'[^\w]', '_', str(c)) for c in X.columns]
X.columns = clean_cols
X_test.columns = clean_cols

for c in clean_cols:
    X[c] = pd.to_numeric(X[c], errors='coerce')
    X_test[c] = pd.to_numeric(X_test[c], errors='coerce')

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbosity=-1
    )
    lgb_model.fit(X_tr, y_tr)
    p_lgb = lgb_model.predict_proba(X_test)[:, 1]
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        eval_metric='auc'
    )
    xgb_model.fit(X_tr, y_tr)
    p_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    cat_model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.03,
        depth=4,
        random_seed=42 + fold,
        verbose=0
    )
    cat_model.fit(X_tr, y_tr)
    p_cat = cat_model.predict_proba(X_test)[:, 1]
    
    hgb_model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.03,
        max_depth=4,
        random_state=42 + fold
    )
    hgb_model.fit(X_tr, y_tr)
    p_hgb = hgb_model.predict_proba(X_test)[:, 1]
    
    fold_pred = (p_lgb * 0.3 + p_xgb * 0.3 + p_cat * 0.3 + p_hgb * 0.1)
    test_preds += fold_pred / skf.n_splits

output = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'defects': test_preds
})

output.to_csv('predictions.csv', index=False)