import re
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.model_selection import KFold
import xgboost as xgb

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/global-dams-database/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/global-dams-database/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Res_capacityM3'

train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)
train_df = train_df[train_df[target_col] >= 0].reset_index(drop=True)

y_train = train_df[target_col].values.astype(np.float64)

X_train = train_df.drop(columns=[target_col])
X_test = test_df.copy()

common_cols = [c for c in X_train.columns if c in X_test.columns]
X_train = X_train[common_cols]
X_test = X_test[common_cols]

clean_cols = [re.sub(r'[\[\]<>" :]', '_', str(col)) for col in X_train.columns]
X_train.columns = clean_cols
X_test.columns = clean_cols

for col in list(X_train.columns):
    if pd.api.types.is_datetime64_any_dtype(X_train[col]):
        X_train[col + '_year'] = X_train[col].dt.year
        X_train[col + '_month'] = X_train[col].dt.month
        X_train[col + '_day'] = X_train[col].dt.day
        X_train = X_train.drop(columns=[col])

        X_test[col + '_year'] = X_test[col].dt.year
        X_test[col + '_month'] = X_test[col].dt.month