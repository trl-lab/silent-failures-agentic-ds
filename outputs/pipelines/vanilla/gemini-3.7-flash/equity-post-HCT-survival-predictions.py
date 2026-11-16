import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/equity-post-HCT-survival-predictions/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/equity-post-HCT-survival-predictions/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

drop_cols = ['ID', 'efs', 'efs_time']
feature_cols = [col for col in train_df.columns if col not in drop_cols]

X = train_df[feature_cols].copy()
y = train_df['efs'].astype(int).values
X_test = test_df[[col for col in feature_cols if col in test_df.columns]].copy()

cat_cols = []
num_cols = []
for col in feature_cols:
    if X[col].dtype == 'object' or str(X[col].dtype) == 'category':
        cat_cols.append(col)
    else:
        num_cols.append(col)

for col in cat_cols:
    X[col] = X[col].fillna('Missing').astype(str)
    X_test[col] = X_test[col].fillna('Missing').astype(str)
    
    categories = sorted(list(set(X[col].unique()).union(set(X_test[col].unique()))))
    X[col] = pd.Categorical(X[col], categories=categories)
    X_test[col] = pd.Categorical(X_test[col], categories=categories)

for col in num_cols:
    median_val = X[col].median()
    X[col] = X[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))

X_cat_str = X.copy()
X_test_cat_str = X_test.copy()
for col in cat_cols:
    X_cat_str[col] = X_cat_str[col].astype(str)
    X_test_cat_str[col] = X_test_cat_str[col].astype(str)

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
    lgb_preds += lgb.predict_proba(X_test)[:, 1] / n_splits
    
    # CatBoost
    X_tr_c = X_cat_str.iloc[train_idx]
    cb = CatBoostClassifier(
        iterations=600,
        learning_rate=0.03,
        depth=6,
        cat_features=cat_cols,
        random_seed=42 + fold,
        verbose=0
    )
    cb.fit(X_tr_c, y_tr)
    cat_preds += cb.predict_proba(X_test_cat_str)[:, 1] / n_splits

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        random_state=42 + fold,
        eval_metric='logloss'
    )
    xgb.fit(X_tr, y_tr)
    xgb_preds += xgb.predict_proba(X_test)[:, 1] / n_splits

final_preds = (lgb_preds + cat_preds + xgb_preds) / 3.0

submission = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'efs': final_preds
})

submission.to_csv('predictions.csv', index=False)