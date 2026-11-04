import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

train_path = '/path/to/user/workspace/ds_failures/datasets/openml/house_sales/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/house_sales/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

y_train = train_df['price'].values
X_train = train_df.drop(columns=['price'])
X_test = test_df.copy()

def preprocess_features(df):
    df = df.copy()
    if 'date' in df.columns:
        date_dt = pd.to_datetime(df['date'], format='%Y%m%dT%H%M%S', errors='coerce')
        df['year_sold'] = date_dt.dt.year
        df['month_sold'] = date_dt.dt.month
        df['day_sold'] = date_dt.dt.day
        df['dayofweek_sold'] = date_dt.dt.dayofweek
        df = df.drop(columns=['date'])
    
    if 'zipcode' in df.columns:
        df['zipcode'] = df['zipcode'].astype('category').cat.codes

    if 'yr_built' in df.columns and 'year_sold' in df.columns:
        df['house_age'] = df['year_sold'] - df['yr_built']
    
    if 'yr_renovated' in df.columns and 'year_sold' in df.columns:
        df['is_renovated'] = (df['yr_renovated'] > 0).astype(int)
        df['years_since_renov'] = np.where(df['yr_renovated'] > 0, df['year_sold'] - df['yr_renovated'], df['house_age'])

    df['sqft_living_ratio'] = df['sqft_living'] / (df['sqft_lot'] + 1.0)
    df['sqft_above_ratio'] = df['sqft_above'] / (df['sqft_living'] + 1.0)
    df['sqft_living_diff15'] = df['sqft_living'] - df['sqft_living15']
    df['sqft_lot_diff15'] = df['sqft_lot'] - df['sqft_lot15']
    df['bed_bath_ratio'] = df['bedrooms'] / (df['bathrooms'] + 0.1)

    return df

X_train_proc = preprocess_features(X_train)
X_test_proc = preprocess_features(X_test)

for col in X_train_proc.columns:
    if X_train_proc[col].dtype == 'object' or str(X_train_proc[col].dtype) == 'category':
        X_train_proc[col] = X_train_proc[col].astype('category').cat.codes
        X_test_proc[col] = X_test_proc[col].astype('category').cat.codes

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test_proc))
cat_preds = np.zeros(len(X_test_proc))
xgb_preds = np.zeros(len(X_test_proc))

for train_idx, val_idx in kf.split(X_train_proc, y_train):
    X_tr, y_tr = X_train_proc.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train_proc.iloc[val_idx], y_train[val_idx]
    
    model_lgb = LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    model_lgb.fit(X_tr, y_tr)
    lgb_preds += model_lgb.predict(X_test_proc) / kf.n_splits
    
    model_cat = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.04,
        depth=6,
        random_seed=42,
        verbose=0
    )
    model_cat.fit(X_tr, y_tr)
    cat_preds += model_cat.predict(X_test_proc) / kf.n_splits
    
    model_xgb = XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model_xgb.fit(X_tr, y_tr)
    xgb_preds += model_xgb.predict(X_test_proc) / kf.n_splits

final_preds = (0.45 * lgb_preds) + (0.35 * cat_preds) + (0.20 * xgb_preds)

sub = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'price': final_preds
})

sub.to_csv('predictions.csv', index=False)