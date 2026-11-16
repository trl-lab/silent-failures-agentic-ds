import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/museums/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/museums/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'REVENUE15'
y_train = train_df[target_col].values
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

# Feature processing
drop_cols = ['MID']
for col in drop_cols:
    if col in X_train.columns:
        X_train.drop(columns=[col], inplace=True)
    if col in X_test.columns:
        X_test.drop(columns=[col], inplace=True)

# Process columns: identify potential numeric columns stored as objects
numeric_cols = []
cat_cols = []

for col in X_train.columns:
    # Try converting to numeric
    s_train = pd.to_numeric(X_train[col], errors='coerce')
    s_test = pd.to_numeric(X_test[col], errors='coerce')
    
    train_num_ratio = s_train.notnull().mean()
    test_num_ratio = s_test.notnull().mean()
    
    if train_num_ratio > 0.6 and test_num_ratio > 0.6:
        X_train[col] = s_train.astype(np.float32)
        X_test[col] = s_test.astype(np.float32)
        numeric_cols.append(col)
    else:
        X_train[col] = X_train[col].astype(str).fillna('missing')
        X_test[col] = X_test[col].astype(str).fillna('missing')
        cat_cols.append(col)

# For LightGBM, convert categorical columns to category type
for col in cat_cols:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# K-Fold Cross Validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))

# Prepare for CatBoost
X_train_cb = X_train.copy()
X_test_cb = X_test.copy()
for col in cat_cols:
    X_train_cb[col] = X_train_cb[col].astype(str)
    X_test_cb[col] = X_test_cb[col].astype(str)
for col in numeric_cols:
    fill_val = X_train_cb[col].median()
    if np.isnan(fill_val):
        fill_val = 0
    X_train_cb[col] = X_train_cb[col].fillna(fill_val)
    X_test_cb[col] = X_test_cb[col].fillna(fill_val)

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'feature_fraction': 0.8,
    'random_state': 42,
    'verbosity': -1,
    'n_estimators': 1500
}

cb_params = {
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'iterations': 1200,
    'learning_rate': 0.06,
    'depth': 6,
    'random_seed': 42,
    'verbose': 0
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    # LightGBM
    X_tr_lgb, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va_lgb, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_tr_lgb, y_tr,
        eval_set=[(X_va_lgb, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_preds += lgb_model.predict(X_test) / n_splits
    
    # CatBoost
    X_tr_cb = X_train_cb.iloc[train_idx]
    X_va_cb = X_train_cb.iloc[val_idx]
    
    cb_model = CatBoostRegressor(**cb_params, cat_features=cat_cols)
    cb_model.fit(
        X_tr_cb, y_tr,
        eval_set=(X_va_cb, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cb_preds += cb_model.predict(X_test_cb) / n_splits

# Ensemble predictions
final_preds = 0.6 * lgb_preds + 0.4 * cb_preds

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(final_preds)),
    'REVENUE15': final_preds
})
output_df.to_csv('predictions.csv', index=False)