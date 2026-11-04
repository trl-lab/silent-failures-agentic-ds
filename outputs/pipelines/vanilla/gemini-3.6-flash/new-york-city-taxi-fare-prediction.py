import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold

train_path = "/path/to/user/workspace/ds_failures/datasets/mlebench/new-york-city-taxi-fare-prediction/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/mlebench/new-york-city-taxi-fare-prediction/data/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

train = train.dropna()
train = train[(train['fare_amount'] > 0) & (train['fare_amount'] < 500)]
train = train[(train['pickup_longitude'] > -80) & (train['pickup_longitude'] < -70)]
train = train[(train['dropoff_longitude'] > -80) & (train['dropoff_longitude'] < -70)]
train = train[(train['pickup_latitude'] > 35) & (train['pickup_latitude'] < 45)]
train = train[(train['dropoff_latitude'] > 35) & (train['dropoff_latitude'] < 45)]

def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    a = np.clip(a, 0.0, 1.0)
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

def dummy_manhattan_distance(lat1, pd_lon1, lat2, pd_lon2):
    a = haversine_np(pd_lon1, lat1, pd_lon1, lat2)
    b = haversine_np(pd_lon1, lat1, pd_lon2, lat1)
    return a + b

def bearing_array(lat1, lng1, lat2, lng2):
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

NYC_LANDMARKS = {
    'jfk': (-73.7781, 40.6413),
    'lga': (-73.8740, 40.7769),
    'ewr': (-74.1745, 40.6895),
    'nyc': (-73.9855, 40.7580),
}

def add_features(df):
    dt = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = dt.dt.hour
    df['year'] = dt.dt.year
    df['month'] = dt.dt.month
    df['day'] = dt.dt.day
    df['dayofweek'] = dt.dt.dayofweek
    df['hour_float'] = df['hour'] + dt.dt.minute / 60.0
    
    dlon = df['dropoff_longitude'] - df['pickup_longitude']
    dlat = df['dropoff_latitude'] - df['pickup_latitude']
    df['dlon'] = dlon
    df['dlat'] = dlat
    df['euclidean'] = np.sqrt(dlon**2 + dlat**2)
    
    df['haversine'] = haversine_np(
        df['pickup_longitude'], df['pickup_latitude'],
        df['dropoff_longitude'], df['dropoff_latitude']
    )
    df['manhattan'] = dummy_manhattan_distance(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    df['bearing'] = bearing_array(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    
    for name, (lon, lat) in NYC_LANDMARKS.items():
        df[f'p_dist_{name}'] = haversine_np(df['pickup_longitude'], df['pickup_latitude'], lon, lat)
        df[f'd_dist_{name}'] = haversine_np(df['dropoff_longitude'], df['dropoff_latitude'], lon, lat)
        
    df['pickup_center_lat'] = df['pickup_latitude'] - 40.75
    df['pickup_center_lon'] = df['pickup_longitude'] - (-73.98)
    df['dropoff_center_lat'] = df['dropoff_latitude'] - 40.75
    df['dropoff_center_lon'] = df['dropoff_longitude'] - (-73.98)
    
    return df

train = add_features(train)
test = add_features(test)

features = [c for c in train.columns if c not in ['key', 'pickup_datetime', 'fare_amount']]
X = train[features]
y = train['fare_amount']
X_test = test[features]

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'n_estimators': 1500,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    test_preds += model.predict(X_test) / kf.n_splits

test_preds = np.clip(test_preds, 2.5, None)

submission = pd.DataFrame({
    'id': np.arange(len(test)),
    'fare_amount': test_preds
})
submission.to_csv('predictions.csv', index=False)