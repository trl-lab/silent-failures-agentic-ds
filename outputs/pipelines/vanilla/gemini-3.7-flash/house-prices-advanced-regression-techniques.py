import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/house-prices-advanced-regression-techniques/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/house-prices-advanced-regression-techniques/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

y = np.log1p(train_df['SalePrice'].values)
X = train_df.drop(columns=['Id', 'SalePrice'])
X_test = test_df.drop(columns=['Id']) if 'Id' in test_df.columns else test_df.copy()

categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

for col in numeric_cols:
    median_val = X[col].median()
    X[col] = X[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

for col in categorical_cols:
    X[col] = X[col].astype(str).fillna('Missing')
    X_test[col] = X_test[col].astype(str).fillna('Missing')
    all_categories = sorted(list(set(X[col].unique()).union(set(X_test[col].unique()))))
    cat_mapping = {cat: idx for idx, cat in enumerate(all_categories)}
    X[col] = X[col].map(cat_mapping).astype(int)
    X_test[col] = X_test[col].map(cat_mapping).astype(int)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))

for train_idx, val_idx in kf.split(X):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]
    
    lgb = LGBMRegressor(n_estimators=1000, learning_rate=0.03, max_depth=6, num_leaves=31, random_state=42, verbose=-1)
    lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[])
    lgb_preds += lgb.predict(X_test) / kf.n_splits
    
    cat = CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=6, random_state=42, verbose=0)
    cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False)
    cat_preds += cat.predict(X_test) / kf.n_splits
    
    xgb = XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=5, random_state=42, verbosity=0)
    xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    xgb_preds += xgb.predict(X_test) / kf.n_splits

final_log_preds = 0.4 * lgb_preds + 0.4 * cat_preds + 0.2 * xgb_preds
final_preds = np.expm1(final_log_preds)

sub = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'SalePrice': final_preds
})
sub.to_csv('predictions.csv', index=False)