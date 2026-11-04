import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/miga-issued-projects/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/miga-issued-projects/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Gross Issued Amount (US$)'
y = train_df[target_col].values

combined = pd.concat([train_df.drop(columns=[target_col]), test_df], ignore_index=True)

# Feature Engineering
# Dates
for col in ['Project Start Date', 'Project Expiration Date']:
    combined[col] = pd.to_datetime(combined[col], errors='coerce')

combined['start_year'] = combined['Project Start Date'].dt.year
combined['start_month'] = combined['Project Start Date'].dt.month
combined['start_day'] = combined['Project Start Date'].dt.day
combined['start_dayofweek'] = combined['Project Start Date'].dt.dayofweek

combined['exp_year'] = combined['Project Expiration Date'].dt.year
combined['exp_month'] = combined['Project Expiration Date'].dt.month
combined['exp_day'] = combined['Project Expiration Date'].dt.day

combined['project_duration_days'] = (combined['Project Expiration Date'] - combined['Project Start Date']).dt.days

# Project Links
combined['has_link'] = combined['Project Links'].notna().astype(int)
combined['link_node_id'] = combined['Project Links'].str.extract(r'(\d+)').astype(float)

# Project Name text features
combined['name_len'] = combined['Project Name'].fillna('').astype(str).str.len()
combined['name_word_count'] = combined['Project Name'].fillna('').astype(str).str.split().str.len()

# Categorical columns to encode
cat_cols = ['Project Name', 'Host Country', 'Host Country Code', 'Host Region', 'Host Region Code', 'Business Sector', 'Product Line']
for col in cat_cols:
    combined[col] = combined[col].astype('category')

# Drop original datetime & link columns
drop_cols = ['Project Start Date', 'Project Expiration Date', 'Project Links']
combined = combined.drop(columns=drop_cols)

n_train = len(train_df)
X_train = combined.iloc[:n_train].copy()
X_test = combined.iloc[n_train:].copy()

# Categorical handling for GBDT
cat_col_names = [c for c in X_train.columns if X_train[c].dtype.name == 'category']

# K-Fold Cross Validation & Ensembling
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))

# XGBoost requires category dtype or numeric
X_train_xgb = X_train.copy()
X_test_xgb = X_test.copy()
for col in cat_col_names:
    X_train_xgb[col] = X_train_xgb[col].cat.codes.astype('category')
    X_test_xgb[col] = X_test_xgb[col].cat.codes.astype('category')

# CatBoost string categorization
X_train_cb = X_train.copy()
X_test_cb = X_test.copy()
for col in cat_col_names:
    X_train_cb[col] = X_train_cb[col].astype(str).fillna('missing')
    X_test_cb[col] = X_test_cb[col].astype(str).fillna('missing')

for train_idx, val_idx in kf.split(X_train):
    # LightGBM
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]
    
    lgb_model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_preds += lgb_model.predict(X_test) / 5.0

    # CatBoost
    X_tr_cb, y_tr_cb = X_train_cb.iloc[train_idx], y[train_idx]
    X_va_cb, y_va_cb = X_train_cb.iloc[val_idx], y[val_idx]
    
    cb_model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.03,
        depth=6,
        cat_features=cat_col_names,
        random_seed=42,
        verbose=False
    )
    cb_model.fit(
        X_tr_cb, y_tr_cb,
        eval_set=(X_va_cb, y_va_cb),
        early_stopping_rounds=50,
        verbose=False
    )
    cb_preds += cb_model.predict(X_test_cb) / 5.0

    # XGBoost
    X_tr_xgb, y_tr_xgb = X_train_xgb.iloc[train_idx], y[train_idx]
    X_va_xgb, y_va_xgb = X_train_xgb.iloc[val_idx], y[val_idx]
    
    xgb_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        enable_categorical=True,
        random_state=42,
        tree_method='hist'
    )
    xgb_model.fit(
        X_tr_xgb, y_tr_xgb,
        eval_set=[(X_va_xgb, y_va_xgb)],
        verbose=False
    )
    xgb_preds += xgb_model.predict(X_test_xgb) / 5.0

final_preds = (lgb_preds + cb_preds + xgb_preds) / 3.0

submission = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    target_col: final_preds
})

submission.to_csv('predictions.csv', index=False)