import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/colleges-and-universities/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/colleges-and-universities/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'RATIO_ENROLL_EMP'
y_train = train_df[target_col].values
X_train = train_df.drop(columns=[target_col])
X_test = test_df.copy()

# Feature processing
cols_to_drop = ['OBJECTID', 'IPEDSID']
cols_to_drop = [c for c in cols_to_drop if c in X_train.columns]

X_train = X_train.drop(columns=cols_to_drop)
X_test = X_test.drop(columns=cols_to_drop)

# Identify column types
cat_cols = []
num_cols = []

for col in X_train.columns:
    if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category':
        cat_cols.append(col)
    else:
        num_cols.append(col)

# Fill missing values and convert categoricals
for col in cat_cols:
    X_train[col] = X_train[col].astype(str).fillna('missing')
    X_test[col] = X_test[col].astype(str).fillna('missing')

for col in num_cols:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

# Convert object columns to category for LightGBM
X_train_lgb = X_train.copy()
X_test_lgb = X_test.copy()
for col in cat_cols:
    X_train_lgb[col] = X_train_lgb[col].astype('category')
    X_test_lgb[col] = X_test_lgb[col].astype('category')

# K-Fold Cross Validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr_lgb, y_tr = X_train_lgb.iloc[train_idx], y_train[train_idx]
    X_va_lgb, y_va = X_train_lgb.iloc[val_idx], y_train[val_idx]
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbose=-1,
        n_jobs=-1
    )
    lgb_model.fit(
        X_tr_lgb, y_tr,
        eval_set=[(X_va_lgb, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_preds += lgb_model.predict(X_test_lgb) / n_splits
    
    # CatBoost
    X_tr_cat = X_train.iloc[train_idx].copy()
    X_va_cat = X_train.iloc[val_idx].copy()
    
    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        cat_features=cat_cols,
        random_seed=42 + fold,
        verbose=0
    )
    cat_model.fit(
        X_tr_cat, y_tr,
        eval_set=(X_va_cat, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cat_preds += cat_model.predict(X_test) / n_splits

# Ensemble predictions
final_predictions = 0.5 * lgb_preds + 0.5 * cat_preds

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    'RATIO_ENROLL_EMP': final_predictions
})
output_df.to_csv('predictions.csv', index=False)