import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

train_path = '/path/to/user/workspace/ds_failures/datasets/uci/seoul-bike-sharing-demand/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/seoul-bike-sharing-demand/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Rented Bike Count'
y = train_df[target_col].values.astype(np.float32)

X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.copy()
if target_col in X_test_raw.columns:
    X_test_raw = X_test_raw.drop(columns=[target_col])

non_functioning_mask = None
if 'Functioning Day' in test_df.columns:
    non_functioning_mask = test_df['Functioning Day'].astype(str).str.lower().isin(['no', '0', 'false', 'non functioning day', 'n'])

df_all = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

if 'Date' in df_all.columns:
    dates = pd.to_datetime(df_all['Date'], dayfirst=True, errors='coerce')
    df_all['year'] = dates.dt.year.fillna(-1).astype(int)
    df_all['month'] = dates.dt.month.fillna(-1).astype(int)
    df_all['day'] = dates.dt.day.fillna(-1).astype(int)
    df_all['dayofweek'] = dates.dt.dayofweek.fillna(-1).astype(int)
    df_all['dayofyear'] = dates.dt.dayofyear.fillna(-1).astype(int)
    df_all['is_weekend'] = dates.dt.dayofweek.isin([5, 6]).astype(int)
    df_all = df_all.drop(columns=['Date'])

if 'Hour' in df_all.columns:
    df_all['hour_sin'] = np.sin(2 * np.pi * df_all['Hour'] / 24.0)
    df_all['hour_cos'] = np.cos(2 * np.pi * df_all['Hour'] / 24.0)

if 'month' in df_all.columns:
    df_all['month_sin'] = np.sin(2 * np.pi * df_all['month'] / 12.0)
    df_all['month_cos'] = np.cos(2 * np.pi * df_all['month'] / 12.0)

if 'dayofweek' in df_all.columns:
    df_all['dow_sin'] = np.sin(2 * np.pi * df_all['dayofweek'] / 7.0)
    df_all['dow_cos'] = np.cos(2 * np.pi * df_all['dayofweek'] / 7.0)

if 'Temperature' in df_all.columns and 'Dew point temperature' in df_all.columns:
    df_all['temp_dew_diff'] = df_all['Temperature'] - df_all['Dew point temperature']

if 'Temperature' in df_all.columns and 'Humidity' in df_all.columns:
    df_all['temp_humidity'] = df_all['Temperature'] * df_all['Humidity']

if 'Solar Radiation' in df_all.columns and 'Temperature' in df_all.columns:
    df_all['solar_temp'] = df_all['Solar Radiation'] * df_all['Temperature']

if 'Rainfall' in df_all.columns:
    df_all['is_raining'] = (df_all['Rainfall'] > 0).astype(int)

if 'Snowfall' in df_all.columns:
    df_all['is_snowing'] = (df_all['Snowfall'] > 0).astype(int)

cat_cols = ['Seasons', 'Holiday', 'Functioning Day']
for c in cat_cols:
    if c in df_all.columns:
        df_all[c] = df_all[c].astype(str).astype('category').cat.codes

X_train = df_all.iloc[:len(train_df)].copy()
X_test = df_all.iloc[len(train_df):].copy()

kf = KFold(n_splits=10, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]
    
    model_lgb = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        learning_rate=0.03,
        num_leaves=63,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        n_estimators=2000,
        random_state=42 + fold,
        verbose=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    lgb_preds += model_lgb.predict(X_test) / kf.n_splits
    
    model_xgb = xgb.XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=2000,
        random_state=42 + fold,
        tree_method='hist',
        early_stopping_rounds=100
    )
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    xgb_preds += model_xgb.predict(X_test) / kf.n_splits
    
    model_cb = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=7,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42 + fold,
        early_stopping_rounds=100,
        verbose=False
    )
    model_cb.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        verbose=False
    )
    cb_preds += model_cb.predict(X_test) / kf.n_splits

final_preds = (lgb_preds + xgb_preds + cb_preds) / 3.0
final_preds = np.clip(final_preds, 0, None)

if non_functioning_mask is not None:
    final_preds[non_functioning_mask] = 0.0

sub = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'Rented Bike Count': final_preds
})
sub.to_csv('predictions.csv', index=False)