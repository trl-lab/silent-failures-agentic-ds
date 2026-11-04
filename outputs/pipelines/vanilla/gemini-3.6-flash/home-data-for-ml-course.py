import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

train_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/home-data-for-ml-course/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/home-data-for-ml-course/data/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

y = np.log1p(train['SalePrice'].values)
train_df = train.drop(columns=['SalePrice'])
test_df = test.copy()

if 'Id' in train_df.columns:
    train_df = train_df.drop(columns=['Id'])
if 'Id' in test_df.columns:
    test_df = test_df.drop(columns=['Id'])

def create_features(df):
    df = df.copy()
    
    df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(0)
    df['1stFlrSF'] = df['1stFlrSF'].fillna(0)
    df['2ndFlrSF'] = df['2ndFlrSF'].fillna(0)
    df['FullBath'] = df['FullBath'].fillna(0)
    df['HalfBath'] = df['HalfBath'].fillna(0)
    df['BsmtFullBath'] = df['BsmtFullBath'].fillna(0)
    df['BsmtHalfBath'] = df['BsmtHalfBath'].fillna(0)
    df['GarageCars'] = df['GarageCars'].fillna(0)
    df['GarageArea'] = df['GarageArea'].fillna(0)
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
    
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['Total_Porch_SF'] = (df['OpenPorchSF'].fillna(0) + 
                            df['3SsnPorch'].fillna(0) + 
                            df['EnclosedPorch'].fillna(0) + 
                            df['ScreenPorch'].fillna(0) + 
                            df['WoodDeckSF'].fillna(0))
    df['TotalBaths'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['IsRemodeled'] = (df['YearBuilt'] != df['YearRemodAdd']).astype(int)
    df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']
    
    df['OverallQual_Cond'] = df['OverallQual'] * df['OverallCond']
    df['Qual_SF'] = df['OverallQual'] * df['TotalSF']
    
    return df

train_df = create_features(train_df)
test_df = create_features(test_df)

cat_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    s_train = train_df[col].astype(str)
    s_test = test_df[col].astype(str)
    le = LabelEncoder()
    le.fit(pd.concat([s_train, s_test]))
    train_df[col] = le.transform(s_train)
    test_df[col] = le.transform(s_test)

num_cols = train_df.columns.tolist()
for col in num_cols:
    median_val = train_df[col].median()
    train_df[col] = train_df[col].fillna(median_val)
    test_df[col] = test_df[col].fillna(median_val)

X = train_df.values
X_test = test_df.values

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_test_preds = np.zeros(len(test_df))
cb_test_preds = np.zeros(len(test_df))
xgb_test_preds = np.zeros(len(test_df))
ridge_test_preds = np.zeros(len(test_df))

for train_idx, val_idx in kf.split(X, y):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    # LightGBM
    model_lgb = lgb.LGBMRegressor(
        objective='regression',
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        n_estimators=1000,
        random_state=42,
        verbose=-1
    )
    model_lgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_test_preds += model_lgb.predict(X_test) / kf.n_splits
    
    # CatBoost
    model_cb = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        loss_function='RMSE',
        random_seed=42,
        verbose=0
    )
    model_cb.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
    cb_test_preds += model_cb.predict(X_test) / kf.n_splits
    
    # XGBoost
    model_xgb = xgb.XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=1000,
        random_state=42,
        early_stopping_rounds=50
    )
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_test_preds += model_xgb.predict(X_test) / kf.n_splits
    
    # Ridge
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model_ridge = Ridge(alpha=10.0)
    model_ridge.fit(X_train_scaled, y_train)
    ridge_test_preds += model_ridge.predict(X_test_scaled) / kf.n_splits

log_preds = (
    0.35 * cb_test_preds + 
    0.35 * lgb_test_preds + 
    0.20 * xgb_test_preds + 
    0.10 * ridge_test_preds
)

final_preds = np.expm1(log_preds)
final_preds = np.clip(final_preds, a_min=0, a_max=None)

submission = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'SalePrice': final_preds
})

submission.to_csv('predictions.csv', index=False)