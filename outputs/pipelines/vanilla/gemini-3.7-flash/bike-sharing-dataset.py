import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/uci/bike-sharing-dataset/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/bike-sharing-dataset/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Feature engineering
def preprocess(df):
    df = df.copy()
    if 'dteday' in df.columns:
        dt = pd.to_datetime(df['dteday'])
        df['day'] = dt.dt.day
        df['dayofyear'] = dt.dt.dayofyear
        df['quarter'] = dt.dt.quarter
        df['days_since_start'] = (dt - pd.to_datetime('2011-01-01')).dt.days
        df = df.drop(columns=['dteday'])
    
    # Interaction features
    df['temp_hum'] = df['temp'] * df['hum']
    df['temp_windspeed'] = df['temp'] * df['windspeed']
    df['is_rush_hour'] = df['hr'].isin([7, 8, 9, 17, 18, 19]).astype(int) * df['workingday']
    df['is_weekend_daytime'] = (~df['workingday'].astype(bool)).astype(int) * df['hr'].between(10, 16).astype(int)
    
    return df

X_train = preprocess(train_df.drop(columns=['cnt']))
y_train = train_df['cnt'].values
X_test = preprocess(test_df)

categorical_cols = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
for col in categorical_cols:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

# K-Fold Cross-Validation Ensemble
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))

# XGBoost requires numerical data
X_train_num = X_train.copy()
X_test_num = X_test.copy()
for col in categorical_cols:
    if col in X_train_num.columns:
        X_train_num[col] = X_train_num[col].astype(int)
        X_test_num[col] = X_test_num[col].astype(int)

for train_idx, val_idx in kf.split(X_train):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_tr_num = X_train_num.iloc[train_idx]
    
    # LightGBM
    lgb = LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=63,
        random_state=42,
        verbosity=-1
    )
    lgb.fit(X_tr, y_tr)
    lgb_preds += lgb.predict(X_test) / n_splits

    # CatBoost
    cat_feature_names = [c for c in categorical_cols if c in X_train.columns]
    cat = CatBoostRegressor(
        iterations=800,
        learning_rate=0.04,
        depth=7,
        cat_features=cat_feature_names,
        verbose=0,
        random_seed=42
    )
    cat.fit(X_tr, y_tr)
    cat_preds += cat.predict(X_test) / n_splits

    # XGBoost
    xgb = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        random_state=42
    )
    xgb.fit(X_tr_num, y_tr)
    xgb_preds += xgb.predict(X_test_num) / n_splits

# Blend predictions
final_predictions = 0.45 * lgb_preds + 0.35 * cat_preds + 0.20 * xgb_preds
final_predictions = np.clip(final_predictions, 0, None)

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    'cnt': final_predictions
})
output_df.to_csv('predictions.csv', index=False)