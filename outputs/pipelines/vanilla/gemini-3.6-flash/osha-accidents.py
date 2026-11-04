```python
import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/osha-accidents/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/osha-accidents/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Task Assigned'

y_raw = train_df[target_col]
if y_raw.dtype == 'bool':
    y = y_raw.astype(int).values
elif pd.api.types.is_numeric_dtype(y_raw):
    y = y_raw.astype(int).values
else:
    str_map = {'true': 1, 'false': 0, 'yes': 1, 'no': 0, 'y': 1, 'n':