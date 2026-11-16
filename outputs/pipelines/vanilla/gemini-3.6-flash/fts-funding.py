import os
import gc
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

warnings.filterwarnings('ignore')

train_path = "/path/to/user/workspace/ds_failures/datasets/strable/fts-funding/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/fts-funding/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'amountUSD'

train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)
y_train = train_df[target_col].values.astype(np.float64)

X_train = train_df.drop(columns=[target_col])
X_test = test_df.copy()
if target_col in X_test.columns:
    X_test = X_test.drop(columns=[target_col])

n_train = len(X_train)
df_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)

cols_to_process = list(df_all.columns)
for col in cols_to_process:
    if pd.api.types.is_datetime64_any_dtype(df_all[col]):
        dt = df_all[col]
        df_all[f"{col}_year"] = dt.dt.year
        df_all[f"{col}_month"] = dt.dt.month
        df_all[f"{col}_day"] = dt.dt.day
        df_all[f"{col}_dayofweek"] = dt.dt.dayofweek
        df_all[f"{col}_dayofyear"] = dt.dt.dayofyear
        df_all.drop(columns=[col], inplace=True)
    elif df_all[col].dtype == 'object' or str(df_all[col].dtype) == 'category':
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                dt = pd.to_datetime(df_all[col], errors='coerce')
                if dt.notnull().sum() > 0.3 *