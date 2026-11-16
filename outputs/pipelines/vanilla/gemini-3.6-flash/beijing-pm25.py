import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

def main():
    train_path = "/path/to/user/workspace/ds_failures/datasets/uci/beijing-pm25/data/train.parquet"
    test_path = "/path/to/user/workspace/ds_failures/datasets/uci/beijing-pm25/data/test.parquet"

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    train = train.dropna(subset=['pm2.5']).reset_index(drop=True)

    train['is_train'] = 1
    test['is_train'] = 0
    test['_test_idx'] = np.arange(len(test))

    combined = pd.concat([train, test], ignore_index=True)

    dt = pd.to_datetime(combined[['year', 'month', 'day', 'hour']])
    combined['dayofweek'] = dt.dt.dayofweek
    combined['dayofyear'] = dt.dt.dayofyear
    combined['weekofyear'] = dt.dt.isocalendar().week.astype(int)

    combined['hour_sin'] = np.sin(2 * np.pi * combined['hour'] / 24.0)
    combined['hour_cos'] = np.cos(2 * np.pi * combined['hour'] / 24.0)
    combined['month_sin'] = np.sin(2 * np.pi * combined['month'] / 12.0)
    combined['month_cos'] = np.cos(2 * np.pi * combined['month'] / 12.0)
    combined['dayofweek_sin'] = np.sin(2 * np.pi * combined['dayofweek'] / 7.0)
    combined['dayofweek_cos'] = np.cos(2 * np.pi * combined['dayofweek'] / 7.0)
    combined['dayofyear_sin'] = np.sin(2 * np.pi * combined['dayofyear'] / 365.25)
    combined['dayofyear_cos'] = np.cos(2 * np.pi * combined['dayofyear'] / 365.25)

    if 'TEMP' in combined.columns and 'DEWP' in combined.columns:
        combined['dew_point_spread'] = combined['TEMP'] - combined['DEWP']

    if 'cbwd' in combined.columns:
        cbwd_str = combined['cbwd'].astype(str)
        cbwd_map = {val: idx for idx, val in enumerate(cbwd_str.unique())}
        combined['cbwd_code'] = cbwd_str.map(cbwd_map)
        freq = cbwd_str.value_counts(normalize=True).to_dict()
        combined['cbwd_freq'] = cbwd_str.map(freq)

    if 'No' in combined.columns:
        combined = combined.sort_values('No').reset_index(drop=True)
    else:
        combined = combined.sort_values(['year', 'month', 'day', 'hour']).reset_index(drop=True)

    weather_cols = [c for c in ['DEWP', 'TEMP', 'PRES', 'Iws', 'dew_point_spread', 'Is', 'Ir'] if c in combined.columns]
    for col in weather_cols:
        combined[f'{col}_lag1'] = combined[col].shift(1)
        combined[f'{col}_lag2'] = combined[col].shift(2)
        combined[f'{col}_lead1'] = combined[col].shift(-1)
        combined[f'{col}_diff1'] = combined[col] - combined[f'{col}_lag1']
        combined[f'{col}_roll_mean_6'] = combined[col].rolling(6, min_periods=1).mean()
        combined[f'{col}_roll_std_6'] = combined[col].rolling(6, min_periods=1).std().fillna(0)
        combined[f'{col}_roll_mean_24'] = combined[col].rolling(24, min_periods=1).mean()

    train_df = combined[combined['is_train'] == 1].reset_index(drop=True)
    test_df = combined[combined['is_train'] == 0].sort_values('_test_idx').reset_index(drop=True)

    drop_cols = ['No', 'pm2.5', 'is_train', '_test_idx', 'cbwd']
    features = [c for c in train_df.columns if c not in drop_cols]

    X = train_df[features].copy()
    y = train_df['pm2.5'].copy()
    X_test = test_df[features].copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    lgb_test_preds = np.zeros(len(X_test))
    xgb_test_preds = np.zeros(len(X_test))
    cb_test_preds = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

        lgb_model = lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        lgb_test_preds += lgb_model.predict(X_test) / kf.n_splits

        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            eval_metric='rmse',
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            n_jobs=-1
        )
        xgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            early_stopping_rounds=50,
            verbose=False
        )
        xgb_test_preds += xgb_model.predict(X_test) / kf.n_splits

        cb_model = CatBoostRegressor(
            loss_function='RMSE',
            eval_metric='RMSE',
            iterations=1200,
            learning_rate=0.04,
            depth=7,
            random_seed=42 + fold,
            verbose=0,
            early_stopping_rounds=50
        )
        cb_model.fit(
            X_tr, y_tr,
            eval_set=(X_va, y_va),
            verbose=False
        )
        cb_test_preds += cb_model.predict(X_test) / kf.n_splits

    final_preds = 0.4 * lgb_test_preds + 0.3 * xgb_test_preds + 0.3 * cb_test_preds
    final_preds = np.clip(final_preds, 0, None)

    sub =