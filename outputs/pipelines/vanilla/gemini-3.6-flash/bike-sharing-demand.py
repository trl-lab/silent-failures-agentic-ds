import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/bike-sharing-demand/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/bike-sharing-demand/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

if 'datetime' in train.columns:
    train['datetime'] = pd.to_datetime(train['datetime'])
    test['datetime'] = pd.to_datetime(test['datetime'])

def extract_features(df):
    df = df.copy()
    if 'datetime' in df.columns:
        dt = df['datetime']
        df['year'] = dt.dt.year
        df['month'] = dt.dt.month
        df['day'] = dt.dt.day
        df['hour'] = dt.dt.hour
        df['dayofweek'] = dt.dt.dayofweek
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        if 'workingday' in df.columns:
            df['peak_working'] = ((df['hour'].isin([7, 8, 9, 17, 18, 19])) & (df['workingday'] == 1)).astype(int)
            df['peak_casual'] = ((df['hour'].isin([11, 12, 13, 14, 15, 16])) & (df['workingday'] == 0)).astype(int)
            
    if 'temp' in df.columns and 'atemp' in df.columns:
        df['temp_diff'] = df['temp'] - df['atemp']
        
    return df

df_train = extract_features(train)
df_test = extract_features(test)

drop_cols = ['datetime', 'casual', 'registered', 'count']
feature_cols = [c for c in df_train.columns if c not in drop_cols]

X_train = df_train[feature_cols]
X_test = df_test[feature_cols]

if 'casual' in train.columns and 'registered' in train.columns:
    y_casual = np.log1p(train['casual'])
    y_registered = np.log1p(train['registered'])
    y_count = np.log1p(train['count'])
    
    model_lgb_c = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, random_state=42, verbose=-1)
    model_xgb_c = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.03, random_state=42, verbosity=0)
    model_cb_c = CatBoostRegressor(n_estimators=1000, learning_rate=0.03, random_state=42, verbose=0)
    
    model_lgb_c.fit(X_train, y_casual)
    model_xgb_c.fit(X_train, y_casual)
    model_cb_c.fit(X_train, y_casual)
    
    pred_casual = np.expm1(
        (model_lgb_c.predict(X_test) + model_xgb_c.predict(X_test) + model_cb_c.predict(X_test)) / 3.0
    )
    pred_casual = np.clip(pred_casual, 0, None)
    
    model_lgb_r = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, random_state=42, verbose=-1)
    model_xgb_r = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.03, random_state=42, verbosity=0)
    model_cb_r = CatBoostRegressor(n_estimators=1000, learning_rate=0.03, random_state=42, verbose=0)
    
    model_lgb_r.fit(X_train, y_registered)
    model_xgb_r.fit(X_train, y_registered)
    model_cb_r.fit(X_train, y_registered)
    
    pred_registered = np.expm1(
        (model_lgb_r.predict(X_test) + model_xgb_r.predict(X_test) + model_cb_r.predict(X_test)) / 3.0
    )
    pred_registered = np.clip(pred_registered, 0, None)
    
    model_lgb_tot = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, random_state=42, verbose=-1)
    model_xgb_tot = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.03, random_state=42, verbosity=0)
    model_cb_tot = CatBoostRegressor(n_estimators=1000, learning_rate=0.03, random_state=42, verbose=0)
    
    model_lgb_tot.fit(X_train, y_count)
    model_xgb_tot.fit(X_train, y_count)
    model_cb_tot.fit(X_train, y_count)
    
    pred_tot = np.expm1(
        (model_lgb_tot.predict(X_test) + model_xgb_tot.predict(X_test) + model_cb_tot.predict(X_test)) / 3.0
    )
    pred_tot = np.clip(pred_tot, 0, None)
    
    final_pred = 0.5 * (pred_casual + pred_registered) + 0.5 * pred_tot
else:
    y_count = np.log1p(train['count'])
    
    model_lgb = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, random_state=42, verbose=-1)
    model_xgb = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.03, random_state=42, verbosity=0)
    model_cb = CatBoostRegressor(n_estimators=1000, learning_rate=0.03, random_state=42, verbose=0)
    
    model_lgb.fit(X_train, y_count)
    model_xgb.fit(X_train, y_count)
    model_cb.fit(X_train, y_count)
    
    final_pred = np.expm1(
        (model_lgb.predict(X_test) + model_xgb.predict(X_test) + model_cb.predict(X_test)) / 3.0
    )

final_pred = np.clip(final_pred, 0, None)

sub = pd.DataFrame({
    'id': np.arange(len(test)),
    'count': final_pred
})

sub.to_csv('predictions.csv', index=False)