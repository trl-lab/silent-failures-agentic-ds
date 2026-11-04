import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold

train_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/predict-who-is-more-influential-in-a-social-network/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/predict-who-is-more-influential-in-a-social-network/data/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

y = train['Choice'].values
X = train.drop(columns=['Choice'])
X_test = test.copy()

if 'id' in X.columns:
    X = X.drop(columns=['id'])
if 'id' in X_test.columns:
    X_test = X_test.drop(columns=['id'])

def swap_raw_ab(df):
    df_swapped = df.copy()
    a_cols = [c for c in df.columns if c.startswith('A_')]
    rename_dict = {}
    for a_col in a_cols:
        b_col = 'B_' + a_col[2:]
        if b_col in df.columns:
            rename_dict[a_col] = b_col
            rename_dict[b_col] = a_col
    return df_swapped.rename(columns=rename_dict)

def engineer_features(df):
    df_feat = df.copy()
    a_cols = [c for c in df.columns if c.startswith('A_')]
    for a_col in a_cols:
        b_col = 'B_' + a_col[2:]
        if b_col in df.columns:
            a_val = df_feat[a_col].astype(float)
            b_val = df_feat[b_col].astype(float)
            
            df_feat[f'diff_{a_col[2:]}'] = a_val - b_val
            
            log_a = np.log1p(np.clip(a_val.fillna(0), 0, None))
            log_b = np.log1p(np.clip(b_val.fillna(0), 0, None))
            df_feat[f'log_diff_{a_col[2:]}'] = log_a - log_b
            
            df_feat[f'ratio_{a_col[2:]}'] = (a_val.fillna(0) + 1.0) / (b_val.fillna(0) + 1.0)

    for col in df_feat.columns:
        if df_feat[col].dtype == 'object' or df_feat[col].dtype.name == 'category':
            df_feat[col] = df_feat[col].astype('category').cat.codes
            
    return df_feat

X_test_swapped = swap_raw_ab(X_test)
X_test_eng = engineer_features(X_test)
X_test_swapped_eng = engineer_features(X_test_swapped)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr_raw, y_tr = X.iloc[train_idx], y[train_idx]
    X_va_raw, y_va = X.iloc[val_idx], y[val_idx]
    
    X_tr_swapped = swap_raw_ab(X_tr_raw)
    X_tr_aug = pd.concat([X_tr_raw, X_tr_swapped], ignore_index=True)
    y_tr_aug = np.concatenate([y_tr, 1 - y_tr])
    
    X_tr_eng = engineer_features(X_tr_aug)
    X_va_eng = engineer_features(X_va_raw)
    
    model_lgb = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbosity=-1
    )
    model_lgb.fit(
        X_tr_eng, y_tr_aug,
        eval_set=[(X_va_eng, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    model_cat = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    model_cat.fit(
        X_tr_eng, y_tr_aug,
        eval_set=(X_va_eng, y_va),
        early_stopping_rounds=50
    )
    
    model_xgb = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        random_state=42 + fold,
        eval_metric='logloss',
        early_stopping_rounds=50
    )
    model_xgb.fit(
        X_tr_eng, y_tr_aug,
        eval_set=[(X_va_eng, y_va)],
        verbose=False
    )
    
    p1_lgb = model_lgb.predict_proba(X_test_eng)[:, 1]
    p2_lgb = model_lgb.predict_proba(X_test_swapped_eng)[:, 1]
    p_lgb = (p1_lgb + (1.0 - p2_lgb)) / 2.0