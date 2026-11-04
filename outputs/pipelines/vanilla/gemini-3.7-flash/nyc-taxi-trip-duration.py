import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/nyc-taxi-trip-duration/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/nyc-taxi-trip-duration/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    km = 6371.0 * c
    return km

def manhattan_distance(lat1, lon1, lat2, lon2):
    lat_dist = haversine_distance(lat1, lon1, lat2, lon1)
    lon_dist = haversine_distance(lat1, lon1, lat1, lon2)
    return lat_dist + lon_dist

def bearing_array(lat1, lon1, lat2, lon2):
    lon_delta_rad = np.radians(lon2 - lon1)
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    y = np.sin(lon_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon_delta_rad)
    return np.degrees(np.arctan2(y, x))

def extract_features(df):
    features = pd.DataFrame(index=df.index)
    
    # Datetime features
    pickup_dt = pd.to_datetime(df['pickup_datetime'])
    features['pickup_hour'] = pickup_dt.dt.hour
    features['pickup_dayofweek'] = pickup_dt.dt.dayofweek
    features['pickup_month'] = pickup_dt.dt.month
    features['pickup_day'] = pickup_dt.dt.day
    features['pickup_minute'] = pickup_dt.dt.minute
    features['is_weekend'] = (features['pickup_dayofweek'] >= 5).astype(int)
    
    # Categorical / basic numeric features
    features['vendor_id'] = df['vendor_id'].astype(int)
    features['passenger_count'] = df['passenger_count'].fillna(1).astype(int)
    features['store_and_fwd_flag'] = (df['store_and_fwd_flag'] == 'Y').astype(int)
    
    # Coordinates
    lat1 = df['pickup_latitude'].values
    lon1 = df['pickup_longitude'].values
    lat2 = df['dropoff_latitude'].values
    lon2 = df['dropoff_longitude'].values
    
    features['pickup_latitude'] = lat1
    features['pickup_longitude'] = lon1
    features['dropoff_latitude'] = lat2
    features['dropoff_longitude'] = lon2
    
    # Coordinate differences & distances
    features['lat_diff'] = np.abs(lat2 - lat1)
    features['lon_diff'] = np.abs(lon2 - lon1)
    features['haversine_dist'] = haversine_distance(lat1, lon1, lat2, lon2)
    features['manhattan_dist'] = manhattan_distance(lat1, lon1, lat2, lon2)
    features['bearing'] = bearing_array(lat1, lon1, lat2, lon2)
    
    # Center coordinates
    features['center_latitude'] = (lat1 + lat2) / 2.0
    features['center_longitude'] = (lon1 + lon2) / 2.0
    
    return features

# Prepare feature matrices
X_train = extract_features(train_df)
X_test = extract_features(test_df)
y_train_log = np.log1p(train_df['trip_duration'].values)

# LightGBM training with K-Fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(test_df))

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'n_estimators': 1500,
    'learning_rate': 0.08,
    'num_leaves': 127,
    'max_depth': -1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train_log[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train_log[val_idx]
    
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    test_preds += model.predict(X_test) / kf.n_splits

# Convert predictions back to original scale
final_predictions = np.expm1(test_preds)
final_predictions = np.clip(final_predictions, 1, None)

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(test_df)),
    'trip_duration': final_predictions
})
output_df.to_csv('predictions.csv', index=False)