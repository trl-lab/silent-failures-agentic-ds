import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb

train_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/airbnb-recruiting-new-user-bookings/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/airbnb-recruiting-new-user-bookings/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

y_train_raw = train_df['country_destination']
label_enc = LabelEncoder()
y_train = label_enc.fit_transform(y_train_raw)

n_train = len(train_df)
n_test = len(test_df)

if 'country_destination' in train_df.columns:
    train_df = train_df.drop(columns=['country_destination'])
if 'country_destination' in test_df.columns:
    test_df = test_df.drop(columns=['country_destination'])

df_all = pd.concat([train_df, test_df], axis=0, ignore_index=True)

if 'date_first_booking' in df_all.columns:
    df_all = df_all.drop(columns=['date_first_booking'])

if 'date_account_created' in df_