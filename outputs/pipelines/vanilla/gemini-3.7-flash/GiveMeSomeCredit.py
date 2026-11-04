import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/GiveMeSomeCredit/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/GiveMeSomeCredit/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'SeriousDlqin2yrs'
drop_cols = ['Unnamed: 0', target_col]

feature_cols = [col for col in train_df.columns if col not in drop_cols]

X = train_df[feature_cols].copy()
y = train_df[target_col].copy()
X_test = test_df[[col for col in feature_cols if col in test_df.columns]].copy()

# 2. Cross-validation and modeling
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    # Model 1: LightGBM
    lgb_model = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_tr, y_tr)
    
    # Model 2: CatBoost
    cat_model = CatBoostClassifier(
        iterations=600,
        learning_rate=0.04,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    cat_model.fit(X_tr, y_tr)
    
    # Predict on test set
    lgb_test_pred = lgb_model.predict_proba(X_test)[:, 1]
    cat_test_pred = cat_model.predict_proba(X_test)[:, 1]
    
    fold_pred = 0.5 * lgb_test_pred + 0.5 * cat_test_pred
    test_preds += fold_pred / n_splits

# 3. Save predictions
submission = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'SeriousDlqin2yrs': test_preds
})

submission.to_csv('predictions.csv', index=False)