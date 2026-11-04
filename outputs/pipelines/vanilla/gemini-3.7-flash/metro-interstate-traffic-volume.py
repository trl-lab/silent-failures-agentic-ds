import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/uci/metro-interstate-traffic-volume/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/metro-interstate-traffic-volume/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Preprocessing and feature engineering
def preprocess(df):
    df = df.copy()
    
    # DateTime features
    dt = pd.to_datetime(df['date_time'])
    df['hour'] = dt.dt.hour
    df['dayofweek'] = dt.dt.dayofweek
    df['month'] = dt.dt.month
    df['year'] = dt.dt.year
    df['day'] = dt.dt.day
    df['dayofyear'] = dt.dt.dayofyear
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # Cyclical hour encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    
    # Holiday processing
    df['is_holiday'] = ((df['holiday'].notna()) & (df['holiday'] != 'None')).astype(int)
    df['holiday'] = df['holiday'].fillna('None').astype(str)
    
    # Weather text normalization
    df['weather_main'] = df['weather_main'].fillna('Unknown').astype(str).str.lower()
    df['weather_description'] = df['weather_description'].fillna('Unknown').astype(str).str.lower()
    
    # Drop raw date_time column
    df = df.drop(columns=['date_time'])
    
    return df

# Target separation
y_train = train_df['traffic_volume'].values
X_train = preprocess(train_df.drop(columns=['traffic_volume']))
X_test = preprocess(test_df)

# Categorical columns
categorical_cols = ['holiday', 'weather_main', 'weather_description']
for col in categorical_cols:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# K-Fold Cross-Validation & LightGBM Model
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'n_estimators': 2000,
    'learning_rate': 0.03,
    'num_leaves': 127,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    test_preds += model.predict(X_test) / kf.n_splits

# Ensure non-negative predictions
test_preds = np.clip(test_preds, 0, None)

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'traffic_volume': test_preds
})
output_df.to_csv('predictions.csv', index=False)