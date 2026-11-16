import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/animalandveterinary-event/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/animalandveterinary-event/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = 'serious_ae'

def parse_target(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in ['1', '1.0', 'true', 'yes', 'y', 't']:
        return 1
    if s in ['0', '0.0', 'false', 'no', 'n', 'f']:
        return 0
    try:
        return int(float(s))
    except Exception:
        return np.nan

y = train[target_col].apply(parse_target)
valid_mask = y.notna()
train = train[valid_mask].reset_index(drop=True)
y = y[valid_mask].astype(int)

X_train = train.drop(columns=[target_col], errors='ignore')
X_test = test.drop