import re
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

train_path = "/path/to/user/workspace/ds_failures/datasets/uci/online-news-popularity/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/uci/online-news-popularity/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

def extract_url_features(df):
    if 'url' not in df.columns:
        return pd.DataFrame(index=df.index)
    urls = df['url'].astype(str)
    pattern = r'(\d{4})/(\d{2})/(\d{2})/([^/]+)'
    extracted = urls.str.extract(pattern)
    year = pd.to_numeric(extracted[0], errors='coerce').fillna(-1)
    month = pd.to_numeric(extracted[1], errors='coerce').fillna(-1)
    day = pd.to_numeric(extracted[2], errors='coerce