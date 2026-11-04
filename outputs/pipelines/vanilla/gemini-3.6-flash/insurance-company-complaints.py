import os
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/insurance-company-complaints/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/insurance-company-complaints/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Recovery'
if target_col not in train_df.columns:
    raise KeyError(f"Target column {target_col} not found in train data.")

y_train_series = pd.to_numeric(train_df[target_col], errors='coerce')
valid_mask = y_train_series.notna()
train_df = train_df[valid_mask].reset_index(drop=True)
y_train = y_train_series[valid_mask].values.astype(float)

X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.drop(columns=[target_col], errors='ignore')

n_train = len(X_train_raw)
n_test = len(X_test_raw)

combined = pd.concat([X_train_raw, X_test_raw