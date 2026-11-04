import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

train_path = '/path/to/user/workspace/ds_failures/datasets/uci/metro-interstate-traffic-volume/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/metro-interstate-traffic-volume/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

def preprocess(train_df, test_df):
    df_all = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    dt = pd.to_datetime(df_all['date_time'])
    df_all['hour'] = dt.dt.hour
    df_all['dayofweek'] = dt.dt.dayofweek
    df_all['month'] = dt.dt.month
    df_all['year'] = dt.dt.year
    df_all['dayofyear'] = dt.dt.dayofyear
    df_all['quarter'] = dt.dt.quarter
    df_all['is_weekend'] = (df_all['dayofweek'] >= 5).astype(int)

    df_all['hour_sin'] = np.sin(2 * np.pi * df_all['hour'] / 24.0)
    df_all['hour_cos'] = np.cos(2 * np.pi * df_all['hour'] / 24.0)
    df_all['month_sin'] = np.sin(2 * np.pi * df_all['month'] / 12.0)
    df_all['month_cos'] = np.cos(2 * np.pi * df_all['month'] / 12.0)
    df_all['dow_sin'] = np.sin(2 * np.pi * df_all['dayofweek'] / 7.0)
    df_all['dow_cos'] = np.cos(2 * np.pi * df_all['dayofweek'] / 7.0)

    df_all['is_rush_hour'] = ((df_all['dayofweek'] < 5) & (
        ((df_all['hour'] >= 7) & (df_all['hour'] <= 9)) |
        ((df_all['hour'] >= 16) & (df_all['hour'] <= 18))
    )).astype(int)

    df_all['temp'] = df_all['temp'].apply(lambda x: np.nan if pd.notna(x) and x < 100 else x)
    df_all['rain_1h'] = df_all['rain_1h'].apply(lambda x: np.nan if pd.notna(x) and x > 100 else x)

    cat_cols = ['holiday', 'weather_main', 'weather_description']
    for col in cat_cols:
        df_all[col] = df_all[col].fillna('Missing').astype(str)
        le = LabelEncoder()
        df_all[col] = le.fit_transform(df_all[col])

    df_all = df_all.drop(columns=['date_time'], errors='ignore')

    n_train = len(train_df)
    train_processed = df_all.iloc[:n_train].copy()
    test_processed = df_all.iloc[n_train:].copy()

    test_processed = test_processed.drop(columns=['traffic_volume'], errors='ignore')

    return train_processed, test_processed

train_p, test_p = preprocess(train, test)

features = [c for c in train_p.columns if c != 'traffic_volume']
X = train_p[features]
y = train_p['traffic_volume']
X_test = test_p[features]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(test_p))
cat_preds = np.zeros(len(test_p))
xgb_preds = np.zeros(len(test_p))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

    model_lgb = LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    model_lgb.fit(X_tr, y_tr)
    lgb_preds += model_lgb.predict(X_test) / 5.0

    model_cat = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=7,
        random_seed=42 + fold,
        verbose=0
    )
    model_cat.fit(X_tr, y_tr)
    cat_preds += model_cat.predict(X_test) / 5.0

    model_xgb = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1
    )
    model_xgb.fit(X_tr, y_tr, verbose=False)
    xgb_preds += model_xgb.predict(X_test) / 5.0

final_preds = (lgb_preds + cat_preds + xgb_preds) / 3.0
final_preds = np.clip(final_preds, 0, None)

sub = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'traffic_volume': final_preds
})
sub.to_csv('predictions.csv', index=False)