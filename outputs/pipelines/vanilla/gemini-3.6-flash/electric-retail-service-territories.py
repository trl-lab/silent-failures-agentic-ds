`, `warnings`, `sklearn.model_selection`, `sklearn.preprocessing`, `sklearn.ensemble`, `lightgbm as lgb`, `xgboost as xgb`, `catboost.CatBoostRegressor`.
No external network access or pip install required.

Everything looks solid and robust. Ready to produce python code.import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

train_path = "/path/to/user/workspace/ds_failures/datasets/strable/electric-retail-service-territories/data/train.parquet"
test_path  = "/path/to/user/workspace/ds_failures/datasets/strable/electric-retail-service-territories/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df  = pd.read_parquet(test_path)

target_col = "TOTAL_MWH"
y = train_df[target_col].values.astype(np.float64)

X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.copy()

common_cols = [c for c in X_train_raw.columns if c in X