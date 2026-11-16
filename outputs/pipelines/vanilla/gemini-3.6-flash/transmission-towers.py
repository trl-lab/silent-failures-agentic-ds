import re
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/transmission-towers/data/train.parquet'
test_path  = '/path/to/user/workspace/ds_failures/datasets/strable/transmission-towers/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = 'SupStruc'

train = train.dropna(subset=[target_col]).reset_index(drop=True)
y = train[target_col].astype(float).values
valid_mask = np.isfinite(y)
train = train.iloc[valid_mask].reset_index(drop=True)
y = y[valid_mask]

X = train.drop(columns=[target_col])
X_test = test.drop(columns=[target_col], errors='ignore')

for col in list(X.columns):
    if pd.api.types.is_datetime64_any_dtype(X[col]) or 'date' in str(col).lower() or 'time' in str(col).lower():
        try:
            d_