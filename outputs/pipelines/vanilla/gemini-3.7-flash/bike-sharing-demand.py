import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/bike-sharing-demand/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/bike-sharing-demand/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

def extract_features(df):
    df = df.copy()
    dt = pd.to_datetime(df['datetime'])
    df['hour'] = dt.dt.hour
    df['day'] = dt.dt.day
    df['month'] = dt.dt.month
    df['year'] = dt.dt.year
    df['dayofweek'] = dt.dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['peak_hour'] = df['hour'].isin([8, 9, 17, 18, 19]).astype(int)
    
    # Drop columns not needed or not present in test
    cols_to_drop = ['datetime', 'casual', 'registered', 'count']
    features = [col for col in df.columns if col not in cols_to_drop]
    return df[features]

X_train = extract_features(train_df)
X_test = extract_features(test_df)

# Target is log-transformed for RMSLE optimization
y_train = np.log1p(train_df['count'].values)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))

categorical_features = ['season', 'holiday', 'workingday', 'weather', 'year', 'month', 'dayofweek', 'hour']

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_val = X_train.iloc[val_idx], y_train[val_idx]
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_preds += lgb_model.predict(X_test) / kf.n_splits
    
    # CatBoost
    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    cb_model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_val),
        early_stopping_rounds=50,
        verbose=False
    )
    cb_preds += cb_model.predict(X_test) / kf.n_splits
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        random_state=42 + fold,
        early_stopping_rounds=50,
        verbosity=0
    )
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_val)],
        verbose=False
    )
    xgb_preds += xgb_model.predict(X_test) / kf.n_splits

# Ensemble predictions and inverse log transform
blend_log = 0.4 * lgb_preds + 0.3 * cb_preds + 0.3 * xgb_preds
final_preds = np.expm1(blend_log)
final_preds = np.clip(final_preds, 0, None)

# Save output
submission = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'count': final_preds
})
submission.to_csv('predictions.csv', index=False)