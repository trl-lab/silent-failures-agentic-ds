import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings

warnings.filterwarnings('ignore')

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/pol-terminal/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/pol-terminal/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'CAPACITY'
y = train_df[target_col].values
X = train_df.drop(columns=[target_col])
X_test = test_df.copy()

# Drop identifier columns with no predictive value or leaking high cardinality
drop_cols = [col for col in ['OBJECTID', 'GlobalID', 'TERM_ID'] if col in X.columns]
X = X.drop(columns=drop_cols)
X_test = X_test.drop(columns=drop_cols)

# Identify column types
cat_cols = []
num_cols = []

for col in X.columns:
    if X[col].dtype == 'object' or pd.api.types.is_categorical_dtype(X[col]):
        cat_cols.append(col)
        X[col] = X[col].astype(str).fillna('missing')
        X_test[col] = X_test[col].astype(str).fillna('missing')
    else:
        num_cols.append(col)
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

# Convert string columns to categorical for LightGBM
X_lgb = X.copy()
X_test_lgb = X_test.copy()
for col in cat_cols:
    X_lgb[col] = X_lgb[col].astype('category')
    X_test_lgb[col] = X_test_lgb[col].astype('category')

# K-Fold Cross Validation Setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_test_preds = np.zeros(len(X_test))
cb_test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr_lgb, y_tr = X_lgb.iloc[train_idx], y[train_idx]
    X_va_lgb, y_va = X_lgb.iloc[val_idx], y[val_idx]
    
    # LightGBM Model
    lgb_model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        random_state=42 + fold,
        verbosity=-1,
        n_jobs=-1
    )
    lgb_model.fit(
        X_tr_lgb, y_tr,
        eval_set=[(X_va_lgb, y_va)],
        callbacks=[]
    )
    lgb_test_preds += lgb_model.predict(X_test_lgb) / n_splits

    # CatBoost Model
    X_tr_cb, y_tr_cb = X.iloc[train_idx], y[train_idx]
    X_va_cb, y_va_cb = X.iloc[val_idx], y[val_idx]
    
    cb_model = CatBoostRegressor(
        iterations=600,
        learning_rate=0.05,
        cat_features=cat_cols,
        random_seed=42 + fold,
        verbose=0
    )
    cb_model.fit(
        X_tr_cb, y_tr_cb,
        eval_set=(X_va_cb, y_va_cb),
        early_stopping_rounds=50,
        verbose=False
    )
    cb_test_preds += cb_model.predict(X_test) / n_splits

# Ensemble predictions
final_preds = 0.5 * lgb_test_preds + 0.5 * cb_test_preds

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'CAPACITY': final_preds
})

output_df.to_csv('predictions.csv', index=False)