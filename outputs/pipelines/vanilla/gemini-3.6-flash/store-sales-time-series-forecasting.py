import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/store-sales-time-series-forecasting/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/store-sales-time-series-forecasting/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = 'sales'
y = np.log1p(np.maximum(0, train[target_col].values.astype(np.float32)))

n_train = len(train)
n_test = len(test)

df_train = train.drop(columns=[target_col])
df_test = test.copy()

date_cols = [c for c in df_train.columns if 'date' in c.lower() or pd.api.types.is_datetime64_any_dtype(df_train[c])]

for c in date_cols:
    df_train[c] = pd.to_datetime(df_train[c])
    df_test[c] = pd.to_datetime(df_test[c])
    
    for df in [df_train, df_test]:
        df[f'{c}_year'] = df[c].dt.year
        df[f'{c}_month'] = df[c].dt.month
        df[f'{c}_day'] = df[c].dt.day
        df[f'{c}_dayofweek'] = df[c].dt.dayofweek
        df[f'{c}_dayofyear'] = df[c].dt.dayofyear
        df[f'{c}_week'] = df[c].dt.isocalendar().week.astype(int)
        df[f'{c}_is_weekend'] = df[c].dt.dayofweek.isin([5, 6]).astype(int)
        
    df_train = df_train.drop(columns=[c])
    df_test = df_test.drop(columns=[c])

feat_cols = [c for c in df_train.columns if c.lower() != 'id']

X = df_train[feat_cols].copy()
X_test = df_test[feat_cols].copy()

bool_cols = X.select_dtypes(include=['bool']).columns.tolist()
for c in bool_cols:
    X[c] = X[c].astype(int)
    X_test[c] = X_test[c].astype(int)

cat_cols = X.select_dtypes(include=['object', '