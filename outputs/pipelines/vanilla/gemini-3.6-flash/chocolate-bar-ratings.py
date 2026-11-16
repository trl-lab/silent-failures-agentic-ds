KF = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)`
- `LabelEncoder` fit and `inverse_transform`
- All imports are standard or allowed packages.

Everything looks completely solid and clean. Ready to write output code.```python
import pandas as pd
import numpy as np
import re
import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

def main():
    train_path = "/path/to/user/workspace/ds_failures/datasets/strable/chocolate-bar-ratings/data/train.parquet"
    test_path = "/path/to/user/workspace/ds_failures/datasets/strable/chocolate-bar-ratings/data/test.parquet"

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    target_col = 'Rating'
    y_raw = train[target_col]
    X_train_raw = train.drop(columns=[target_col])
    X_test_raw = test.copy()

    if target_col in X_test_raw.columns