import re
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import StratifiedKFold, KFold
import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

def main():
    train_path = '/path/to/user/workspace/ds_failures/datasets/strable/transmission-lines/data/train.parquet'
    test_path = '/path/to/user/workspace/ds_failures/datasets/strable/transmission-lines/data/test.parquet'

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    target_col = 'VOLT_CLASS'

    train = train.dropna(subset=[target_col]).reset_index(drop=True)

    y_raw = train[target_col].astype(str)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)

    X_train = train.drop(columns=[target_col])
    X_test = test.drop(columns=[target_col], errors='ignore')

    def extract_geo_features(df):
        if 'geometry' in df.columns:
            try:
                import shapely.wkt
                import shapely.wkb