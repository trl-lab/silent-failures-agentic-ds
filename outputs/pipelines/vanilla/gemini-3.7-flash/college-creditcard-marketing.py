import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/college-creditcard-marketing/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/college-creditcard-marketing/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'PAYMENTS_BY_ISSUER'
y_train = train_df[target_col].values

# 2. Preprocess data
def preprocess(df):
    data = df.copy()
    
    # Standardize string representations and handle special missing symbols like '.'
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].astype(str).str.strip().str.lower()
            data[col] = data[col].replace(['.', 'none', 'nan', 'null', ''], np.nan)
            
    # Convert numerical column that was loaded as object
    if 'NEW_ACCOUNTS_OPENED_IN_REPORTING_YEAR' in data.columns:
        data['NEW_ACCOUNTS_OPENED_IN_REPORTING_YEAR'] = pd.to_numeric(
            data['NEW_ACCOUNTS_OPENED_IN_REPORTING_YEAR'], errors='coerce'
        )
        
    # Feature engineering
    if 'TOTAL_OPEN_ACCOUNTS_AS_OF_END_OF_REPORTING_YEAR' in data.columns and 'NEW_ACCOUNTS_OPENED_IN_REPORTING_YEAR' in data.columns:
        data['NEW_TO_TOTAL_RATIO'] = (
            data['NEW_ACCOUNTS_OPENED_IN_REPORTING_YEAR'] / 
            (data['TOTAL_OPEN_ACCOUNTS_AS_OF_END_OF_REPORTING_YEAR'] + 1.0)
        )
    
    return data

X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.copy()

X_train_clean = preprocess(X_train_raw)
X_test_clean = preprocess(X_test_raw)

# Identify column types
cat_cols = []
num_cols = []
for col in X_train_clean.columns:
    if X_train_clean[col].dtype == 'object' or pd.api.types.is_string_dtype(X_train_clean[col]):
        cat_cols.append(col)
        X_train_clean[col] = X_train_clean[col].fillna('__MISSING__').astype('category')
        X_test_clean[col] = X_test_clean[col].fillna('__MISSING__').astype('category')
    else:
        num_cols.append(col)

# 3. Model training with K-Fold CV
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test_clean))
cb_preds = np.zeros(len(X_test_clean))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_clean, y_train)):
    X_tr, y_tr = X_train_clean.iloc[train_idx], y_train[train_idx]
    X_va, y_val = X_train_clean.iloc[val_idx], y_train[val_idx]
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbosity=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_preds += lgb_model.predict(X_test_clean) / n_splits
    
    # CatBoost
    cb_cat_features = cat_cols
    X_tr_cb = X_tr.copy()
    X_va_cb = X_va.copy()
    X_test_cb = X_test_clean.copy()
    for col in cb_cat_features:
        X_tr_cb[col] = X_tr_cb[col].astype(str)
        X_va_cb[col] = X_va_cb[col].astype(str)
        X_test_cb[col] = X_test_cb[col].astype(str)
        
    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.04,
        depth=6,
        cat_features=cb_cat_features,
        random_seed=42 + fold,
        verbose=0
    )
    cb_model.fit(
        X_tr_cb, y_tr,
        eval_set=(X_va_cb, y_val),
        early_stopping_rounds=50,
        verbose=False
    )
    cb_preds += cb_model.predict(X_test_cb) / n_splits

# 4. Ensemble predictions
final_predictions = 0.5 * lgb_preds + 0.5 * cb_preds
final_predictions = np.clip(final_predictions, 0, None)

# 5. Save predictions
submission_df = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    'PAYMENTS_BY_ISSUER': final_predictions
})
submission_df.to_csv('predictions.csv', index=False)