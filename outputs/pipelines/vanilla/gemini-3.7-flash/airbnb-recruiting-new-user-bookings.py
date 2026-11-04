import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/airbnb-recruiting-new-user-bookings/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/airbnb-recruiting-new-user-bookings/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Separate target
y_train_raw = train_df['country_destination']
le_target = LabelEncoder()
y_train = le_target.fit_transform(y_train_raw)

# Combine for preprocessing
train_len = len(train_df)
df = pd.concat([train_df.drop(columns=['country_destination']), test_df], ignore_index=True)

# Process date_account_created
dac = pd.to_datetime(df['date_account_created'], format='%Y-%m-%d', errors='coerce')
df['dac_year'] = dac.dt.year
df['dac_month'] = dac.dt.month
df['dac_day'] = dac.dt.day
df['dac_dayofweek'] = dac.dt.dayofweek
df['dac_dayofyear'] = dac.dt.dayofyear

# Process timestamp_first_active
tfa = pd.to_datetime(df['timestamp_first_active'].astype(str), format='%Y%m%d%H%M%S', errors='coerce')
df['tfa_year'] = tfa.dt.year
df['tfa_month'] = tfa.dt.month
df['tfa_day'] = tfa.dt.day
df['tfa_hour'] = tfa.dt.hour
df['tfa_dayofweek'] = tfa.dt.dayofweek
df['tfa_dayofyear'] = tfa.dt.dayofyear