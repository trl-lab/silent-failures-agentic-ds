import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

train_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/nyc-taxi-trip-duration/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/nyc-taxi-trip-duration/data/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

if 'trip_duration' in train.columns:
    train = train[(train['trip_duration'] >= 10) & (train['trip_duration'] <= 86400)].reset_index(drop=True)

n_train = len(train)
y_train = np.log1p(train['trip_duration'].values)

df = pd.concat([
    train.drop(columns=['trip_duration', 'dropoff_datetime'], errors='ignore'),
    test.drop(columns=['dropoff_datetime'], errors='ignore')
], ignore_index=True)

df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['hour'] = df['pickup_datetime'].dt.hour
df['weekday'] = df['pickup_datetime'].dt.dayofweek
df['month'] = df['pickup_datetime'].dt.month
df['day'] = df['pickup_datetime'].dt.day
df['minute'] = df['pickup_datetime'].dt.minute
df['dayofyear'] = df['pickup_datetime'].dt.dayofyear
df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
df['minute_of_day'] = df['hour'] * 60 + df['minute']

lat1 = df['pickup_latitude'].values
lng1 = df['pickup_longitude'].values
lat2 = df['dropoff_latitude'].values
lng2 = df['dropoff_longitude'].values

def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371.0
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    return 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(np.clip(d, 0, 1)))

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def bearing_array(lat1, lng1, lat2, lng2):
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1_r, lng1_r, lat2_r, lng2_r = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2_r)
    x = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

df['haversine_dist'] = haversine_array(lat1, lng1, lat2, lng2)
df['manhattan_dist'] = dummy_manhattan_distance(lat1, lng1, lat2, lng2)
df['bearing'] = bearing_array(lat1, lng1, lat2, lng2)
df['lat_diff'] = np.abs(lat2 - lat1)
df['lng_diff'] = np.abs(lng2 - lng1)
df['center_lat'] = (lat1 + lat2) / 2.0
df['center_lng'] = (lng1 + lng2) / 2.0

if 'store_and_fwd_flag' in df.columns:
    df['store_and_fwd_flag'] = (df['store_and_fwd_flag'] == 'Y').astype(int)

df['hour_dist_mean'] = df.groupby('hour')['haversine_dist'].transform('mean')
df['weekday_dist_mean'] = df.groupby('weekday')['haversine_dist'].transform('mean')

features_to_drop = ['id', 'pickup_datetime']
features = [c for c in df.columns if c not in features_to_drop]

X_train = df.iloc[:n_train][features]
X_test = df.iloc[n_train:][features]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 127,
    'max_depth': -1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'min_child_samples': 20,
    'n_jobs': -1,
    'random_state': 42,
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]
    
    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=trn_data)
    
    model = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=1200,
        valid_sets=[trn_data, val_data],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits

final_preds = np.expm1(test_preds)
final_preds = np.clip(final_preds, a_min=1.0, a_max=None)

sub = pd.DataFrame({
    'id': np.arange(len(X_test)),
    'trip_duration': final_preds
})

sub.to_csv('predictions.csv', index=False)