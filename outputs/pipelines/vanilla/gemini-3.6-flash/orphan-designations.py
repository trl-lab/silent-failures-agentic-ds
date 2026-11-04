import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

def main():
    train_path = '/path/to/user/workspace/ds_failures/datasets/strable/orphan-designations/data/train.parquet'
    test_path = '/path/to/user/workspace/ds_failures/datasets/strable/orphan-designations/data/test.parquet'

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    target_col = 'Status'
    y_raw = train[target_col]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    feature_cols = [c for c in train.columns if c in test.columns and c.lower() not in ['status', 'id']]

    X_tr = pd.DataFrame(index=train.index)
    X_te = pd.DataFrame(index=test.index)

    for c in feature_cols:
        s_tr = train[c]
        s_te = test[c]

        is_date = False
        if pd.api.types.is_datetime64_any_dtype(s_tr) or 'date