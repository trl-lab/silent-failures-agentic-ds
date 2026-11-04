import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/home-data-for-ml-course/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/home-data-for-ml-course/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Drop ID column if present
features = [c for c in train_df.columns if c not in ['Id', 'SalePrice']]
X = train_df[features].copy()
y = train_df['SalePrice'].copy()
X_test = test_df[features].copy()

# Target transformation
y_log = np.log1p(y)

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Fill missing values
for col in categorical_cols:
    X[col] = X[col].astype(str).fillna('missing')
    X_test[col] = X_test[col].astype(str).fillna('missing')

for col in numeric_cols:
    median_val = X[col].median()
    X[col] = X[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

# Convert categorical columns to category dtype for LightGBM and CatBoost
for col in categorical_cols:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# K-Fold Cross Validation Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))

# LightGBM Parameters
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'n_estimators': 1500,
    'learning_rate': 0.03,
    'num_leaves': 31,
    'random_state': 42,
    'verbose': -1
}

# Train LightGBM
for train_idx, val_idx in kf.split(X, y_log):
    X_tr, y_tr = X.iloc[train_idx], y_log.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y_log.iloc[val_idx]
    
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_preds += model_lgb.predict(X_test) / kf.n_splits

# Train CatBoost
for train_idx, val_idx in kf.split(X, y_log):
    X_tr, y_tr = X.iloc[train_idx].copy(), y_log.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx].copy(), y_log.iloc[val_idx]
    
    # Convert category columns to strings for CatBoost
    for col in categorical_cols:
        X_tr[col] = X_tr[col].astype(str)
        X_va[col] = X_va[col].astype(str)
        
    cb_test = X_test.copy()
    for col in categorical_cols:
        cb_test[col] = cb_test[col].astype(str)
        
    model_cb = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        cat_features=categorical_cols,
        verbose=0,
        random_seed=42
    )
    model_cb.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cb_preds += model_cb.predict(cb_test) / kf.n_splits

# Blend predictions and invert log transformation
final_log_preds = 0.5 * lgb_preds + 0.5 * cb_preds
final_preds = np.expm1(final_log_preds)

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'SalePrice': final_preds
})
output_df.to_csv('predictions.csv', index=False)