import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/uci/magic-gamma-telescope/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/magic-gamma-telescope/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Separate features and target
target_col = 'class'
X = train_df.drop(columns=[target_col]).copy()
y_raw = train_df[target_col]
X_test = test_df.copy()

if target_col in X_test.columns:
    X_test = X_test.drop(columns=[target_col])

# Encode target
le = LabelEncoder()
y = le.fit_transform(y_raw)

# Preprocessing: Handle missing values for numerical features
for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype.name == 'category':
        col_le = LabelEncoder()
        X[col] = col_le.fit_transform(X[col].astype(str))
        X_test[col] = col_le.transform(X_test[col].astype(str))
    else:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

# K-Fold Ensemble
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    # LightGBM
    lgb_model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[]
    )
    test_preds += lgb_model.predict_proba(X_test)[:, 1] / (3 * n_splits)

    # CatBoost
    cb_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False)
    test_preds += cb_model.predict_proba(X_test)[:, 1] / (3 * n_splits)

    # XGBoost
    xgb_model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        random_state=42 + fold,
        eval_metric='logloss',
        early_stopping_rounds=50
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    test_preds += xgb_model.predict_proba(X_test)[:, 1] / (3 * n_splits)

# Output predictions
output_df = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'class': test_preds
})

output_df.to_csv('predictions.csv', index=False)