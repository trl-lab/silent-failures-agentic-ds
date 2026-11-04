import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb
from catboost import CatBoostRegressor

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/community-banking_wide/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/community-banking_wide/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'LOANTOASSET'
y_train = train_df[target_col].values

# Combine for uniform preprocessing
n_train = len(train_df)
combined = pd.concat([train_df.drop(columns=[target_col]), test_df], axis=0, ignore_index=True)

# Drop raw Id if present as feature
if 'Id' in combined.columns:
    combined = combined.drop(columns=['Id'])

# Clean string columns that may contain numbers (percentages, commas)
def clean_numeric_str(series):
    if series.dtype == 'object':
        s = series.astype(str).str.replace('%', '', regex=False).str.replace(',', '', regex=False).str.strip()
        s = s.replace(['None', 'nan', 'NaN', 'null', ''], np.nan)
        try:
            return pd.to_numeric(s)
        except Exception:
            return series
    return series

for col in combined.columns:
    cleaned = clean_numeric_str(combined[col])
    if pd.api.types.is_numeric_dtype(cleaned):
        combined[col] = cleaned

# Identify categorical and numeric columns
cat_cols = []
num_cols = []

for col in combined.columns:
    if combined[col].dtype == 'object' or combined[col].dtype.name == 'category':
        cat_cols.append(col)
        combined[col] = combined[col].astype(str).fillna('missing')
    else:
        num_cols.append(col)

# Convert categorical columns to category dtype for LightGBM
for col in cat_cols:
    combined[col] = combined[col].astype('category')

X_train = combined.iloc[:n_train].copy()
X_test = combined.iloc[n_train:].copy()

# K-Fold ensemble
kf = KFold(n_splits=5, shuffle=True, random_state=42)
lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))

cat_features_idx = [X_train.columns.get_loc(c) for c in cat_cols]

for train_idx, val_idx in kf.split(X_train, y_train):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_val = X_train.iloc[val_idx], y_train[val_idx]
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_preds += lgb_model.predict(X_test) / kf.n_splits

    # CatBoost
    X_tr_cb = X_tr.copy()
    X_va_cb = X_va.copy()
    X_test_cb = X_test.copy()
    for c in cat_cols:
        X_tr_cb[c] = X_tr_cb[c].astype(str)
        X_va_cb[c] = X_va_cb[c].astype(str)
        X_test_cb[c] = X_test_cb[c].astype(str)
        
    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        cat_features=cat_cols,
        random_seed=42,
        verbose=0
    )
    cb_model.fit(
        X_tr_cb, y_tr,
        eval_set=(X_va_cb, y_val),
        early_stopping_rounds=50,
        verbose=False
    )
    cat_preds += cb_model.predict(X_test_cb) / kf.n_splits

final_preds = 0.5 * lgb_preds + 0.5 * cat_preds

output_df = pd.DataFrame({
    'id': range(len(final_preds)),
    'LOANTOASSET': final_preds
})
output_df.to_csv('predictions.csv', index=False)