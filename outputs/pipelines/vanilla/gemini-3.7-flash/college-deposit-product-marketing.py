import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/college-deposit-product-marketing/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/college-deposit-product-marketing/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'TOTAL_ACTIVE_ACCOUNTS_'
y_train = train_df[target_col].values

X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

combined = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)

# Replace common missing value indicators
missing_values = ['.', 'nan', 'None', 'NULL', 'null', '', ' ', 'N/A', 'NA']
combined = combined.replace(missing_values, np.nan)

# Parse potential date columns
date_cols = [col for col in combined.columns if 'DATE' in col.upper()]
for col in date_cols:
    combined[col] = pd.to_datetime(combined[col], errors='coerce')
    combined[f'{col}_year'] = combined[col].dt.year
    combined[f'{col}_month'] = combined[col].dt.month
    combined[f'{col}_day'] = combined[col].dt.day
    combined.drop(columns=[col], inplace=True)

# Convert object columns that are numeric to numeric
for col in combined.columns:
    if combined[col].dtype == 'object':
        cleaned = combined[col].astype(str).str.replace(',', '').str.replace('$', '', regex=False)
        converted = pd.to_numeric(cleaned, errors='coerce')
        if converted.notna().sum() > 0.5 * len(combined[col].dropna()):
            combined[col] = converted

# Extract year features from AWARD_YEAR if present
if 'AWARD_YEAR' in combined.columns:
    combined['AWARD_YEAR_START'] = pd.to_numeric(combined['AWARD_YEAR'].astype(str).str.extract(r'(\d{4})')[0], errors='coerce')

# Process remaining categorical columns
cat_cols = []
for col in combined.columns:
    if combined[col].dtype == 'object' or pd.api.types.is_categorical_dtype(combined[col]):
        combined[col] = combined[col].astype(str).astype('category')
        cat_cols.append(col)

X_train_proc = combined.iloc[:len(train_df)].copy().reset_index(drop=True)
X_test_proc = combined.iloc[len(train_df):].copy().reset_index(drop=True)

# K-Fold Ensemble
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_test_preds = np.zeros(len(X_test_proc))
cb_test_preds = np.zeros(len(X_test_proc))

# Categorical column names for LightGBM and CatBoost
cat_indices = [X_train_proc.columns.get_loc(c) for c in cat_cols]

for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train_proc, y_train)):
    X_tr, y_tr = X_train_proc.iloc[trn_idx], y_train[trn_idx]
    X_va, y_va = X_train_proc.iloc[val_idx], y_train[val_idx]
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_test_preds += lgb_model.predict(X_test_proc) / kf.n_splits
    
    # CatBoost
    X_tr_cb = X_tr.copy()
    X_va_cb = X_va.copy()
    X_te_cb = X_test_proc.copy()
    for col in cat_cols:
        X_tr_cb[col] = X_tr_cb[col].astype(str).fillna('missing')
        X_va_cb[col] = X_va_cb[col].astype(str).fillna('missing')
        X_te_cb[col] = X_te_cb[col].astype(str).fillna('missing')
        
    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        cat_features=cat_cols,
        random_seed=42 + fold,
        verbose=0
    )
    cb_model.fit(
        X_tr_cb, y_tr,
        eval_set=(X_va_cb, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cb_test_preds += cb_model.predict(X_te_cb) / kf.n_splits

final_predictions = 0.5 * lgb_test_preds + 0.5 * cb_test_preds

output_df = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    target_col: final_predictions
})

output_df.to_csv('predictions.csv', index=False)