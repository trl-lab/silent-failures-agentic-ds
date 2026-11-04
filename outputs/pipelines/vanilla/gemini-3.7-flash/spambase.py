import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/openml/spambase/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/spambase/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Separate features and target
target_col = 'class'
X = train_df.drop(columns=[target_col]).copy()
y = train_df[target_col].astype(int).values
X_test = test_df.copy()

# Ensure all feature columns are numeric
for col in X.columns:
    if X[col].dtype.name == 'category' or X[col].dtype == 'object':
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

# Fill any potential missing values with column medians
for col in X.columns:
    median_val = X[col].median()
    X[col] = X[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

# 2. Train models with Stratified K-Fold Cross-Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds_lgb = np.zeros(len(X_test))
test_preds_cat = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]
    
    # LightGBM
    lgb = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbose=-1
    )
    lgb.fit(X_tr, y_tr)
    test_preds_lgb += lgb.predict_proba(X_test)[:, 1] / n_splits
    
    # CatBoost
    cat = CatBoostClassifier(
        iterations=600,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    cat.fit(X_tr, y_tr)
    test_preds_cat += cat.predict_proba(X_test)[:, 1] / n_splits
    
    # XGBoost
    xgb = XGBClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42 + fold
    )
    xgb.fit(X_tr, y_tr)
    test_preds_xgb += xgb.predict_proba(X_test)[:, 1] / n_splits

# 3. Ensemble predictions
final_preds = (test_preds_lgb + test_preds_cat + test_preds_xgb) / 3.0

# 4. Save predictions
submission_df = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'class': final_preds
})
submission_df.to_csv('predictions.csv', index=False)