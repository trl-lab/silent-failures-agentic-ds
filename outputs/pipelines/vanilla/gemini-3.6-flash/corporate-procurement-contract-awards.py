import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
import lightgbm as lgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/corporate-procurement-contract-awards/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/corporate-procurement-contract-awards/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Contract Award Amount (USD)'
y_train = train_df[target_col].values

X_train_raw = train_df.drop(columns=[target_col], errors='ignore')
X_test_raw = test_df.drop(columns=[target_col], errors='ignore')

valid_y_mask = ~np.isnan(y_train)
X_train_raw = X_train_raw[valid_y_mask].reset_index(drop=True)
y_train = y_train[valid_y_mask]
y_train = np.maximum(0, y_train)

n_train = len(X_train_raw)
df_all = pd.concat([X_train_raw, X_test_raw], ignore_index=True)

date_cols = []
cat_cols = []
num_cols = []
text_cols = []

for col in df_all.columns:
    col_str = str(col).lower()
    if 'date' in col_str or df_all[col].dtype == 'datetime64[ns]':
        date_cols.append(col)
    elif df_all[col].dtype == 'object' or df_all[col].dtype.name == 'category':
        sample = df