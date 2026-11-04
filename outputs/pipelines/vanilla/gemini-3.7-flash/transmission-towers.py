import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/transmission-towers/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/transmission-towers/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'SupStruc'
y_train = train_df[target_col].values
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

# Drop identifier/URL columns that don't generalize
drop_cols = ['url', 'FID']
X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])

# Identify categorical and numerical columns
cat_cols = []
num_cols = []

for col in X_train.columns:
    if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category':
        cat_cols.append(col)
        # Fill missing values and convert to string category
        X_train[col] = X_train[col].astype(str).fillna('missing').astype('category')
        X_test[col] = X_test[col].astype(str).fillna('missing').astype('category')
    else:
        num_cols.append(col)
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

# K-Fold Cross Validation Setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_test_preds = np.zeros(len(X_test))

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'n_estimators': 1500,
    'learning_rate': 0.03,
    'num_leaves': 63,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_val = X_train.iloc[val_idx], y_train[val_idx]
    
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    lgb_test_preds += model.predict(X_test) / n_splits

# Generate output submission dataframe
sub = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'SupStruc': lgb_test_preds
})

sub.to_csv('predictions.csv', index=False)