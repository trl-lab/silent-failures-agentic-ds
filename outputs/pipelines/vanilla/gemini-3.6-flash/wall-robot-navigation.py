import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

train_path = "/path/to/user/workspace/ds_failures/datasets/openml/wall-robot-navigation/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/openml/wall-robot-navigation/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "class"
X_train_raw = train_df.drop(columns=[target_col])
y_train_raw = train_df[target_col]
X_test_raw = test_df.drop(columns=[target_col], errors='ignore')

def extract_features(df):
    df_feat = df.copy()
    s_cols = list(df.columns)
    if len(s_cols) >= 2:
        vals = df[s_cols].values
        
        df_feat['s_mean'] = np.mean(vals, axis=1)
        df_feat['s_std'] = np.std(vals, axis=1)
        df_feat['s_min'] = np.min(vals, axis=1)
        df_feat['s_max'] = np.max(vals, axis=1)
        df_feat['s_median'] = np.median(vals, axis=1)
        df_feat['s_ptp'] = df_feat['s_max'] - df_feat['s_min']
        
        for i in range(len(s_cols)):
            next_i = (i + 1) % len(s_cols)
            df_feat[f'diff_{i}_{next_i}'] = df[s_cols[i]] - df[s_cols[next_i]]
            
        if len(s_cols) == 24:
            for i in range(12):
                df_feat[f'opp_diff_{i}'] = df[s_cols[i]] - df[s_cols[i+12]]
                
    return df_feat

X_train = extract_features(X_train_raw)
X_test = extract_features(X_test_raw)

X_train.columns = [str(c) for c in X_train.columns]
X_test.columns = [str(c) for c in X_test.columns]

le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)

num_classes = len(le.classes_)
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds = np.zeros((len(X_test), num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    
    lgb = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    lgb.fit(X_tr, y_tr)
    test_preds += lgb.predict_proba(X_test) / (n_splits * 3)
    
    cb = CatBoostClassifier(
        iterations=500,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    cb.fit(X_tr, y_tr)
    test_preds += cb.predict_proba(X_test) / (n_splits * 3)
    
    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        random_state=42 + fold,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    xgb.fit(X_tr, y_tr)
    test_preds += xgb.predict_proba(X_test) / (n_splits * 3)

final_preds_idx = np.argmax(test_preds, axis=1)
final_preds = le.inverse_transform(final_preds_idx)

submission = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'class': final_preds
})
submission.to_csv('predictions.csv', index=False)