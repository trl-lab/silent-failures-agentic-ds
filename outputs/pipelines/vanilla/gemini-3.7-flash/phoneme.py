import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
import warnings

warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/openml/phoneme/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/phoneme/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)