import ast
import json
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
import xgboost as xgb

warnings.filterwarnings('ignore')

train_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/tmdb-box-office-prediction/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/tmdb-box-office-prediction/data/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

train_len = len(train)
test_len = len(test)

y_train_raw = train['revenue'].values
y_train_log = np.log1p(np.maximum(0, y_train_raw))

df_all = pd.concat(
    [train.drop(columns=['revenue'], errors='ignore'), test], axis=0
).reset_index(drop=True)


def safe_eval(val):
    if pd.isna(val) or val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, dict):
        return [