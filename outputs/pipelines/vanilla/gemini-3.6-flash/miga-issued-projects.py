import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold

import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb

warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/miga-issued-projects/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/miga-issued-projects/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Gross Issued Amount (US$)'

y_train = train_df[target_col].values.astype(float)
train_features = train_df.drop(columns=[target_col], errors='ignore')
test_features = test_df.drop(columns=[target_col], errors='ignore')

n_train = len(train_features)
n_test = len(test_features)

df_all = pd.concat([train_features, test_features], axis=0, ignore_index=True)

for col in list(df_all.columns):
    if pd.api.types.is_datetime64_any_dtype