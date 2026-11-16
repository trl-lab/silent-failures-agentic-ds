import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

train_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/porto-seguro-safe-driver-prediction/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/porto-seguro-safe-driver-prediction/data/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

ignore_cols = ['id', 'target']
base_features = [c for c in train.columns if c not in ignore_cols]
y = train['target'].values

train['missing_count'] = (train[base_features] == -1).sum(axis=1)
test['missing_count'] = (test[base_features] == -1).sum(axis=1)

cat_cols = [c for c in base_features if c.endswith('_cat')]
for col in cat_cols:
    freq = pd.concat([train[col], test[col]]).value_counts(normalize=True)
    train[f'{col}_freq'] = train[col].map(freq)
    test[f'{col}_freq'] = test[col].map(freq)

num_