```python
import os
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from lightgbm import LGBMRegressor, early_stopping
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')

def main():
    train_path = "/path/to/user/workspace/ds_failures/datasets/strable/electric-generating-plants/data/train.parquet"
    test_path = "/path/to/user/workspace/ds_failures/datasets/strable/electric-generating-plants/data/test.parquet"

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    # Drop missing target rows
    train = train.dropna(subset=['NET_GEN']).reset_index(drop=True)

    y = train['NET_GEN'].values.astype(np.float64)
    train_df = train.drop(columns=['NET_GEN'])
    test_df = test.copy()

    n_train = len(train_df)
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    # Process datetime columns
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[f'{