import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold

train_path = '/path/to/user/workspace/ds_failures/datasets/uci/occupancy-detection/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/occupancy-detection/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

def extract_features(df):
    df = df.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['hour'] = df['date'].dt.hour
        df['minute'] = df['date'].dt.minute
        df['dayofweek'] = df['date'].dt.dayofweek
        df['second'] = df['date'].dt.second
        
        seconds_in_day = (df['hour'] * 3600 + df['minute'] * 60 + df['second']) / 86400.0
        df['sin_time'] = np.sin(2 * np.pi * seconds_in_day)
        df['cos_time'] = np.cos(2 * np.pi * seconds_in_day)

    sensor_cols = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
    for col in sensor_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[f'{col}_diff1'] = df[col].diff(1).fillna(0)
            df[f'{col}_diff5'] = df[col].diff(5).fillna(0)
            df[f'{col}_roll5_mean'] = df[col].rolling(5, min_periods=1).mean().fillna(0)
            df[f'{col}_roll15_mean'] = df[col].rolling(15, min_periods=1).mean().fillna(0)
            df[f'{col}_roll5_std'] = df[col].rolling(5, min_periods=1).std().fillna(0)

    if 'Light' in df.columns:
        df['light_gt_0'] = (df['Light'] > 0).astype(int)
        df['light_gt_100'] = (df['Light'] > 100).astype(int)
        
    if 'CO2' in df.columns and 'Temperature' in df.columns:
        df['co2_temp_ratio'] = np.where(df['Temperature'] != 0, df['CO2'] / df['Temperature'], 0)
        
    if 'Humidity' in df.columns and 'Temperature' in df.columns:
        df['hum_temp_ratio'] = np.where(df['Temperature'] != 0, df['Humidity'] / df['Temperature'], 0)
        
    return df

train_feat = extract_features(train_df)
test_feat = extract_features(test_df)

drop_cols = ['id', 'date', 'Occupancy']
feature_cols = [c for c in train_feat.columns if c not in drop_cols]

X = train_feat[feature_cols].values
y = train_df['Occupancy'].values
X_test = test_feat[feature_cols].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[val_idx], y[val_idx]
    
    lgb = LGBMClassifier(n_estimators=500, learning_rate=0.03, random_state=42, verbose=-1, max_depth=6)
    lgb.fit(X_tr, y_tr)
    
    xgb = XGBClassifier(n_estimators=500, learning_rate=0.03, random_state=42, max_depth=6, eval_metric='auc')
    xgb.fit(X_tr, y_tr)
    
    cat = CatBoostClassifier(iterations=500, learning_rate=0.03, random_seed=42, verbose=0, depth=6)
    cat.fit(X_tr, y_tr)
    
    fold_preds = (
        lgb.predict_proba(X_test)[:, 1] + 
        xgb.predict_proba(X_test)[:, 1] + 
        cat.predict_proba(X_test)[:, 1]
    ) / 3.0
    
    test_preds += fold_preds / 5.0

submission = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'Occupancy': test_preds
})

submission.to_csv('predictions.csv', index=False)