import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/sberbank-russian-housing-market/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/sberbank-russian-housing-market/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

if 'price_doc' in train.columns:
    y_train = np.log1p(train['price_doc'].values)
    train = train.drop(columns=['price_doc'])

for df in [train, test]:
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)

for df in [train, test]:
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'])
        df['year'] = ts.dt.year
        df['month'] = ts.dt.month
        df['day'] = ts.dt.day
        df['dayofweek'] = ts.dt.dayofweek
        df['dayofyear'] = ts.dt.dayofyear
        df['year_month'] = df['year'] * 12 + df['month']
        min_date = pd.to_datetime('2011-01-01')
        df['time_idx'] = (ts - min_date).dt.days
        df.drop(columns=['timestamp'], inplace=True)

def add_features(df):
    cols = set(df.columns)
    if 'full_sq' in cols and 'life_sq' in cols:
        df['life_sq_ratio'] = df['life_sq'] / (df['full_sq'] + 1.0)
    if 'full_sq' in cols and 'kitch_sq' in cols:
        df['kitch_sq_ratio'] = df['kitch_sq'] / (df['full_sq'] + 1.0)
    if 'floor' in cols and 'max_floor' in cols:
        df['floor_ratio'] = df['floor'] / (df['max_floor'] + 1.0)
    if 'full_sq' in cols and 'num_room' in cols:
        df['room_sq'] = df['full_sq'] / (df['num_room'] + 1.0)
    if 'year' in cols and 'build_year' in cols:
        df['age'] = df['year'] - df['build_year']
        df['age'] = df['age'].apply(lambda x: x if (0 <= x <= 200) else np.nan)
    return df

train = add_features(train)
test = add_features(test)

features = [c for c in train.columns if c in test.columns]
train = train[features]
test = test[features]

cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    train[col] = train[col].astype(str)
    test[col] = test[col].astype(str)
    cats = sorted(list(set(train[col].unique()).union(set(test[col].unique()))))
    cat_map = {c: i for i, c in enumerate(cats)}
    train[col] = train[col].map(cat_map).fillna(-1).astype(int)
    test[col] = test[col].map(cat_map).fillna(-1).astype(int)

for col in features:
    if train[col].dtype == 'float64':
        train[col] = train[col].astype('float32')
        test[col] = test[col].astype('float32')
    elif train[col].dtype == 'int64':
        train[col] = train[col].astype('int32')
        test[col] = test[col].astype('int32')

kf = KFold(n_splits=5, shuffle=True, random_state=42)
lgb_preds = np.zeros(len(test))
xgb_preds = np.zeros(len(test))
cat_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    X_tr, y_tr = train.iloc[train_idx], y_train[train_idx]
    X_va, y_val = train.iloc[val_idx], y_train[val_idx]

    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_preds += lgb_model.predict(test) / kf.n_splits

    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        random_state=42 + fold,
        n_jobs=-1,
        tree_method='hist'
    )
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_val)],
        verbose=False
    )
    xgb_preds += xgb_model.predict(test) / kf.n_splits

    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.04,
        depth=6,
        random_seed=42 + fold,
        verbose=False,
        early_stopping_rounds=50
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_val), verbose=False)
    cat_preds += cat_model.predict(test) / kf.n_splits

final_log_preds = 0.4 * lgb_preds + 0.3 * xgb_preds + 0.3 * cat_preds
final_preds = np.expm1(final_log_preds)
final_preds = np.clip(final_preds, 0, None)

sub = pd.DataFrame({
    'id': np.arange(len(test)),
    'price_doc': final_preds
})
sub.to_csv('predictions.csv', index=False)