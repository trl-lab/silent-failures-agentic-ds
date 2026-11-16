We propose a gradient boosting solution using LightGBM tailored to the prosumer energy forecasting task. Separate models are trained for energy consumption (`is_consumption=1`) and energy production (`is_consumption=0`) to capture distinct physical and behavioral drivers. The feature pipeline engineers calendar and cyclical time features, historical target lag and rolling statistics (with a minimum 48-hour lag to prevent leakage), county-aggregated weather forecasts (solar radiation, temperature, cloud cover, and wind), electricity and gas price signals, and client capacity metrics. The models are evaluated using Mean Absolute Error (MAE) on a temporal hold-out validation set representing the most recent data blocks.

```python
import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

def load_data():
    def read_df(name):
        candidates = [
            f"./input/{name}.parquet",
            f"./input/{name}.csv",
            f"./input/{name}/{name}.parquet",
            f"./input/{name}/{name}.csv",
            f"./input/{name}"
        ]
        for path in candidates:
            if os.path.exists(path):
                if path.endswith('.parquet'):
                    return pd.read_parquet(path)
                elif path.endswith('.csv'):
                    return pd.read_csv(path)
        return None

    train = read_df("train")
    test = read_df("test")
    client = read_df("client")
    electricity_prices = read_df("electricity_prices")
    gas_prices = read_df("gas_prices")
    forecast_weather = read_df("forecast_weather")
    weather_mapping = read_df("weather_station_to_county_mapping")
    
    return train, test, client, electricity_prices, gas_prices, forecast_weather, weather_mapping

print("Loading data...")
train_df, test_df, client_df, elec_df, gas_df, f_weather_df, w_mapping_df = load_data()

# Process datetime columns
for df in [train_df, test_df]:
    if df is not None:
        df['datetime'] = pd.to_datetime(df['datetime'])

# Preprocess weather forecast
if f_weather_df is not None and w_mapping_df is not None:
    f_weather_df['forecast_datetime'] = pd.to_datetime(f_weather_df['forecast_datetime'])
    w_mapping_clean = w_mapping_df.dropna(subset=['county', 'latitude', 'longitude'])
    w_mapping_clean = w_mapping_clean.groupby(['latitude', 'longitude'])['county'].first().reset_index()
    f_weather_df = f_weather_df.merge(w_mapping_clean, on=['latitude', 'longitude'], how='inner')
    
    weather_agg = f_weather_df.groupby(['forecast_datetime', 'county'])[[
        'temperature', 'dewpoint', 'cloudcover_total', 'cloudcover_low',
        'cloudcover_mid', 'cloudcover_high', 'direct_solar_radiation',
        'surface_solar_radiation', '10_metre_u_wind_component', '10_metre_v_wind_component'
    ]].mean().reset_index()
    weather_agg.rename(columns={'forecast_datetime': 'datetime'}, inplace=True)
else:
    weather_agg = None

# Preprocess prices
if elec_df is not None:
    elec_df['forecast_date'] = pd.to_datetime(elec_df['forecast_date'])
    elec_df = elec_df.groupby('forecast_date')[['euros_per_mwh']].mean().reset_index()
    elec_df.rename(columns={'forecast_date': 'datetime'}, inplace=True)

if gas_df is not None:
    gas_df['forecast_date'] = pd.to_datetime(gas_df['forecast_date'])
    gas_df['gas_price_mean'] = (gas_df['lowest_price_per_mwh'] + gas_df['highest_price_per_mwh']) / 2.0
    gas_df = gas_df.groupby('forecast_date')[['gas_price_mean']].mean().reset_index()
    gas_df['date'] = gas_df['forecast_date'].dt.date

# Preprocess client data
if client_df is not None:
    client_df['date'] = pd.to_datetime(client_df['date']).dt.date
    client_cols = ['county', 'is_business', 'product_type', 'date', 'eic_count', 'installed_capacity']
    client_df = client_df[client_cols].drop_duplicates()

# Combine train and test to engineer features seamlessly
train_df['is_train'] = 1
test_df['is_train'] = 0
if 'target' not in test_df.columns:
    test_df['target'] = np.nan

combined = pd.concat([train_df, test_df], ignore_index=True)
combined['date'] = combined['datetime'].dt.date

# Merge Client
if client_df is not None:
    combined = combined.merge(client_df, on=['county', 'is_business', 'product_type', 'date'], how='left')
    combined['installed_capacity'] = combined['installed_capacity'].fillna(0)
    combined['eic_count'] = combined['eic_count'].fillna(0)

# Merge Weather
if weather_agg is not None:
    combined = combined.merge(weather_agg, on=['datetime', 'county'], how='left')

# Merge Electricity
if elec_df is not None:
    combined = combined.merge(elec_df, on='datetime', how='left')

# Merge Gas
if gas_df is not None:
    combined = combined.merge(gas_df[['date', 'gas_price_mean']], on='date', how='left')

# Datetime Features
combined['hour'] = combined['datetime'].dt.hour
combined['dayofweek'] = combined['datetime'].dt.dayofweek
combined['dayofyear'] = combined['datetime'].dt.dayofyear
combined['month'] = combined['datetime'].dt.month
combined['sin_hour'] = np.sin(2 * np.pi * combined['hour'] / 24)
combined['cos_hour'] = np.cos(2 * np.pi * combined['hour'] / 24)

# Create Target Lag Features
print("Creating lag features...")
lag_group_cols = ['county', 'is_business', 'product_type', 'is_consumption']
combined.sort_values(by=lag_group_cols + ['datetime'], inplace=True)

# Lags of 48h, 72h, 96h, 168h
for lag in [48, 72, 96, 168]:
    combined[f'target_lag_{lag}h'] = combined.groupby(lag_group_cols)['target'].shift(lag)

# Rolling stats over 48h lag
combined['target_roll_mean_48h'] = combined.groupby(lag_group_cols)['target_lag_48h'].transform(lambda x: x.rolling(24, min_periods=1).mean())
combined['target_roll_std_48h'] = combined.groupby(lag_group_cols)['target_lag_48h'].transform(lambda x: x.rolling(24, min_periods=1).std())

# Split back to train and test
train_data = combined[combined['is_train'] == 1].copy()
test_data = combined[combined['is_train'] == 0].copy()

# Drop rows where target is missing in train
train_data = train_data.dropna(subset=['target']).reset_index(drop=True)
test_data.sort_values(by='datetime', inplace=True)
test_data = test_data.reset_index(drop=True)

# Validation Split based on data_block_id (last 60 blocks for temporal hold-out)
max_block = train_data['data_block_id'].max()
val_cutoff = max_block - 60

features = [
    'county', 'is_business', 'product_type', 'is_consumption',
    'eic_count', 'installed_capacity', 'hour', 'dayofweek', 'dayofyear', 'month',
    'sin_hour', 'cos_hour', 'target_lag_48h', 'target_lag_72h', 'target_lag_96h',
    'target_lag_168h', 'target_roll_mean_48h', 'target_roll_std_48h'
]

weather_cols = [
    'temperature', 'dewpoint', 'cloudcover_total', 'cloudcover_low',
    'cloudcover_mid', 'cloudcover_high', 'direct_solar_radiation',
    'surface_solar_radiation', '10_metre_u_wind_component', '10_metre_v_wind_component',
    'euros_per_mwh', 'gas_price_mean'
]
for col in weather_cols:
    if col in train_data.columns:
        features.append(col)

cat_features = ['county', 'is_business', 'product_type', 'is_consumption']
for c in cat_features:
    train_data[c] = train_data[c].astype('category')
    test_data[c] = test_data[c].astype('category')

lgb_params = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'learning_rate': 0.08,
    'num_leaves': 127,
    'n_estimators': 800,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# Train separate models for consumption (is_consumption == 1) and production (is_consumption == 0)
val_mask = train_data['data_block_id'] > val_cutoff
tr_df = train_data[~val_mask]
va_df = train_data[val_mask]

val_preds = np.zeros(len(va_df))
test_preds = np.zeros(len(test_data))

for is_cons in [0, 1]:
    tr_sub = tr_df[tr_df['is_consumption'] == is_cons]
    va_sub = va_df[va_df['is_consumption'] == is_cons]
    te_sub = test_data[test_data['is_consumption'] == is_cons]
    
    sub_features = [f for f in features if f != 'is_consumption']
    
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        tr_sub[sub_features], tr_sub['target'],
        eval_set=[(va_sub[sub_features], va_sub['target'])],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
    )
    
    val_preds[va_df['is_consumption'] == is_cons] = model.predict(va_sub[sub_features])
    test_preds[test_data['is_consumption'] == is_cons] = model.predict(te_sub