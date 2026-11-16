import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/bioresponse/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/bioresponse/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Activity'
feature_cols = [c for c in train_df.columns if c != target_col and c in test_df.columns]

X = train_df[feature_cols].copy()
y = train_df[target_col].values
X_test = test_df[feature_cols].copy()

# Fill missing values if any
X = X.fillna(X.median(numeric_only=True))
X_test = X_test.fillna(X.median(numeric_only=True))

# 2. Cross-validation and ensembling
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds_lgb = np.zeros(len(X_test))
test_preds_cat = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]
    
    # LightGBM
    lgb = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    lgb.fit(X_tr, y_tr)
    test_preds_lgb += lgb.predict_proba(X_test)[:, 1] / n_splits
    
    # CatBoost
    cat = CatBoostClassifier(
        iterations=500,
        learning_rate=0.04,
        depth=6,
        random_seed=42 + fold,
        verbose=0,
        thread_count=-1
    )
    cat.fit(X_tr, y_tr)
    test_preds_cat += cat.predict_proba(X_test)[:, 1] / n_splits
    
    # XGBoost
    xgb = XGBClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.7,
        eval_metric='logloss',
        random_state=42 + fold,
        n_jobs=-1
    )
    xgb.fit(X_tr, y_tr)
    test_preds_xgb += xgb.predict_proba(X_test)[:, 1] / n_splits

# Blend predictions
final_preds = (test_preds_lgb * 0.35) + (test_preds_cat * 0.40) + (test_preds_xgb * 0.25)
final_preds = np.clip(final_preds, 1e-15, 1 - 1e-15)

# 3. Save predictions
submission = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'Activity': final_preds
})

submission.to_csv('predictions.csv', index=False)