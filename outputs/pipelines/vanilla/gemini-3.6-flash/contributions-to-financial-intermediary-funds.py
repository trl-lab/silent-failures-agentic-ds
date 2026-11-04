import os
import warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/contributions-to-financial-intermediary-funds/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/contributions-to-financial-intermediary-funds/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Amount in USD'
train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)
y = train_df[target_col].values

feature_cols = [c for c in train_df.columns if c in test_df.columns and c != target_col]

df_all = pd.concat([train_df[feature_cols], test_df[feature_cols]], axis=0).reset_index(drop=True)

datetime_cols = []
for c in feature_cols:
    if pd.api.types.is_datetime64_any_dtype(df_all[c]):
        parsed = df_all[c]
    elif df_all[c].dtype == 'object':
        try:
            parsed = pd.to_datetime(df_all[c], errors='coerce')
            if parsed.notna().sum() < 0.3 * len(df_all):
                parsed = None
        except Exception:
            parsed = None
    else:
        parsed = None

    if parsed is not None:
        df_all[c + '_year'] = parsed.dt.year
        df_all[c + '_month'] = parsed.dt.month
        df_all[c + '_day'] = parsed.dt.day
        df_all[c + '_dayofweek'] = parsed.dt.dayofweek
        df_all[c + '_timestamp'] = (parsed - pd.Timestamp("1970-01-01")).dt.total_seconds()
        datetime_cols.append(c)

df_all = df_all.drop(columns=datetime_cols)

cat_cols = []
num_cols = []
for c in df_all.columns:
    if df_all[c].dtype == 'object' or df_all[c].dtype.name == 'category':
        cat_cols.append(c)
    else:
        num_cols.append(c)

for c in cat_cols:
    freq = df_all[c].value_counts().to_dict()
    df_all[c + '_freq'] = df_all[c].map(freq)
    le = LabelEncoder()
    df_all[c] = le.fit_transform(df_all[c].astype(str).fillna('missing'))

for c in num_cols:
    df_all[c] = df_all[c].astype(float)

X_train = df_all.iloc[:len(train_df)].copy()
X_test = df_all.iloc[len(train_df):].copy()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
lgb_preds = np.zeros(len(test_df))
cat_preds = np.zeros(len(test_df))
xgb_preds = np.zeros(len(test_df))

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 3000,
    'learning_rate': 0.03,
    'num_leaves': 63,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1
}

cat_params = {
    'iterations': 3000,
    'learning_rate': 0.03,
    'depth': 6,
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'verbose': 0
}

xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'n_estimators': 3000,
    'learning_rate': 0.03,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )
    lgb_preds += model_lgb.predict(X_test) / kf.n_splits

    model_cat = CatBoostRegressor(**cat_params)
    model_cat.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=100
    )
    cat_preds += model_cat.predict(X_test) / kf.n_splits

    model_xgb = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=100)
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    xgb_preds += model_xgb.predict(X_test) / kf.n_splits

final_preds = (lgb_preds + cat_preds + xgb_preds) / 3.0

sub = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'Amount in USD': final_preds
})

sub.to_csv('predictions.csv', index=False)