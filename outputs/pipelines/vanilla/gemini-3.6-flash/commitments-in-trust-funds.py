import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/commitments-in-trust-funds/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/commitments-in-trust-funds/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = 'New Commitments (US$)'

y_train = train[target_col].values.astype(np.float64)
min_y = np.min(y_train)

X_train = train.drop(columns=[target_col]).copy()
X_test = test.copy()

train_len = len(X_train)
df_all = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)

for col in list(df_all.columns):
    if df_all[col].dtype == 'object' or df_all[col].dtype == 'category' or pd.api.types.is_datetime64_any_dtype(df_all[col