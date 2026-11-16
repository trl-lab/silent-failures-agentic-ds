import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/uci/electrical-grid-stability/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/electrical-grid-stability/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'stabf'
feature_cols = [c for c in train_df.columns if c != target_col and c != 'stab']

X = train_df[feature_cols].copy()
y_raw = train_df[target_col]

le = LabelEncoder()
y = le.fit_transform(y_raw)

X_test = test_df[feature_cols].copy()

# Stratified K-Fold Cross-Validation & Ensembling
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_f, y_train_f = X.iloc[train_idx], y[train_idx]
    X_val_f, y_val_f = X.iloc[val_idx], y[val_idx]
    
    # LightGBM
    model_lgb = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbose=-1
    )
    model_lgb.fit(
        X_train_f, y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    # CatBoost
    model_cb = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    model_cb.fit(
        X_train_f, y_train_f,
        eval_set=(X_val_f, y_val_f),
        early_stopping_rounds=50,
        verbose=False
    )
    
    # XGBoost
    model_xgb = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        eval_metric='auc',
        random_state=42 + fold,
        early_stopping_rounds=50
    )
    model_xgb.fit(
        X_train_f, y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        verbose=False
    )
    
    val_pred = (
        model_lgb.predict_proba(X_val_f)[:, 1] +
        model_cb.predict_proba(X_val_f)[:, 1] +
        model_xgb.predict_proba(X_val_f)[:, 1]
    ) / 3.0
    
    oof_preds[val_idx] = val_pred
    
    fold_test_pred = (
        model_lgb.predict_proba(X_test)[:, 1] +
        model_cb.predict_proba(X_test)[:, 1] +
        model_xgb.predict_proba(X_test)[:, 1]
    ) / 3.0
    
    test_preds += fold_test_pred / n_splits

cv_auc = roc_auc_score(y, oof_preds)
print(f"OOF ROC AUC Score: {cv_auc:.5f}")

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'stabf': test_preds
})
output_df.to_csv('predictions.csv', index=False)