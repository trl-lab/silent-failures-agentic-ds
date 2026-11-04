import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/uci/default-of-credit-card-clients/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/default-of-credit-card-clients/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'default payment next month'
feature_cols = [c for c in train_df.columns if c != target_col]

X = train_df[feature_cols].copy()
y = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

# Feature engineering
def add_features(df):
    df = df.copy()
    # Bill amounts sum and mean
    bill_cols = [f'X{i}' for i in range(12, 18)]
    pay_amt_cols = [f'X{i}' for i in range(18, 24)]
    pay_status_cols = [f'X{i}' for i in range(6, 12)]
    
    df['bill_sum'] = df[bill_cols].sum(axis=1)
    df['pay_amt_sum'] = df[pay_amt_cols].sum(axis=1)
    df['pay_status_max'] = df[pay_status_cols].max(axis=1)
    df['pay_status_sum'] = df[pay_status_cols].sum(axis=1)
    
    # Utilization ratio
    df['utilization'] = df['bill_sum'] / (df['X1'] * 6 + 1e-5)
    return df

X_feat = add_features(X)
X_test_feat = add_features(X_test)

# Stratified K-Fold Ensemble
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_test_preds = np.zeros(len(X_test_feat))
xgb_test_preds = np.zeros(len(X_test_feat))
cat_test_preds = np.zeros(len(X_test_feat))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_feat, y)):
    X_tr, y_tr = X_feat.iloc[train_idx], y.iloc[train_idx]
    X_va, y_val = X_feat.iloc[val_idx], y.iloc[val_idx]
    
    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_test_preds += lgb_model.predict_proba(X_test_feat)[:, 1] / n_splits
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='auc',
        random_state=42 + fold,
        early_stopping_rounds=50,
        n_jobs=-1
    )
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_val)],
        verbose=False
    )
    xgb_test_preds += xgb_model.predict_proba(X_test_feat)[:, 1] / n_splits
    
    # CatBoost
    cat_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.04,
        depth=5,
        eval_metric='AUC',
        random_seed=42 + fold,
        verbose=False
    )
    cat_model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_val),
        early_stopping_rounds=50,
        verbose=False
    )
    cat_test_preds += cat_model.predict_proba(X_test_feat)[:, 1] / n_splits

# Ensemble average
final_preds = (lgb_test_preds + xgb_test_preds + cat_test_preds) / 3.0

# Create submission dataframe
submission_df = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'default payment next month': final_preds
})

submission_df.to_csv('predictions.csv', index=False)