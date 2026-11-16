import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

train_path = '/path/to/user/workspace/ds_failures/datasets/openml/churn/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/churn/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'class'
y = train_df[target_col].astype(int).values
X = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

categorical_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()

for col in categorical_cols:
    X[col] = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)
    le = LabelEncoder()
    le.fit(list(X[col]) + list(X_test[col]))
    X[col] = le.transform(X[col])
    X_test[col] = le.transform(X_test[col])

for col in X.columns:
    if X[col].isnull().any() or X_test[col].isnull().any():
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    lgb = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbosity=-1
    )
    lgb.fit(X_tr, y_tr)
    lgb_pred = lgb.predict_proba(X_test)[:, 1]

    cat = CatBoostClassifier(
        iterations=500,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=False
    )
    cat.fit(X_tr, y_tr)
    cat_pred = cat.predict_proba(X_test)[:, 1]

    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        random_state=42 + fold,
        eval_metric='logloss'
    )
    xgb.fit(X_tr, y_tr)
    xgb_pred = xgb.predict_proba(X_test)[:, 1]

    fold_pred = (lgb_pred + cat_pred + xgb_pred) / 3.0
    test_preds += fold_pred / n_splits

output_df = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'class': test_preds
})

output_df.to_csv('predictions.csv', index=False)