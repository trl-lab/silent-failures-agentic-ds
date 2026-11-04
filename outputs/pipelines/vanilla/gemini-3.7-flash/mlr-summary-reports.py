import re
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/mlr-summary-reports/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/mlr-summary-reports/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Adjusted MLR'
y = train_df[target_col].values
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

def preprocess(df):
    data = df.copy()
    data.columns = [c.strip() for c in data.columns]
    
    # Clean currency and comma-formatted numeric strings
    for col in data.columns:
        if data[col].dtype == 'object':
            sample = data[col].dropna().astype(str)
            if len(sample) > 0 and sample.str.contains(r'[\$,]').any():
                cleaned = data[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip()
                data[col] = pd.to_numeric(cleaned, errors='coerce')
    
    # Date parsing
    date_cols = [c for c in data.columns if 'date' in c.lower()]
    for col in date_cols:
        dates = pd.to_datetime(data[col], errors='coerce')
        data[f'{col}_year'] = dates.dt.year
        data[f'{col}_month'] = dates.dt.month
        data[f'{col}_day'] = dates.dt.day
        data[f'{col}_dayofweek'] = dates.dt.dayofweek
        data.drop(columns=[col], inplace=True)
        
    # MLR specific ratio features
    num_cols = [c for c in data.columns if 'numerator' in c.lower()]
    den_cols = [c for c in data.columns if 'denominator' in c.lower()]
    if num_cols and den_cols:
        num = data[num_cols[0]]
        den = data[den_cols[0]]
        ratio = (num / (den + 1e-8)).replace([np.inf, -np.inf], np.nan)
        data['raw_mlr_ratio'] = ratio
        data['raw_mlr_pct'] = ratio * 100.0
        data['log_mlr_pct'] = np.log(np.maximum(ratio * 100.0, 1e-6))
        data['num_den_diff'] = den - num

    return data

X_train_proc = preprocess(X_train)
X_test_proc = preprocess(X_test)

# Align columns
all_cols = [c for c in X_train_proc.columns if c in X_test_proc.columns]
X_train_proc = X_train_proc[all_cols].copy()
X_test_proc = X_test_proc[all_cols].copy()

# Categorical handling via frequency and factorize
cat_cols = X_train_proc.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    train_str = X_train_proc[col].astype(str).fillna('missing')
    test_str = X_test_proc[col].astype(str).fillna('missing')
    
    freq = train_str.value_counts(normalize=True).to_dict()
    X_train_proc[f'{col}_freq'] = train_str.map(freq).fillna(0.0).astype(float)
    X_test_proc[f'{col}_freq'] = test_str.map(freq).fillna(0.0).astype(float)
    
    codes, uniques = pd.factorize(train_str)
    mapping = {val: i for i, val in enumerate(uniques)}
    X_train_proc[col] = codes
    X_test_proc[col] = test_str.map(mapping).fillna(-1).astype(int)

# Ensure numeric types everywhere
for col in X_train_proc.columns:
    X_train_proc[col] = pd.to_numeric(X_train_proc[col], errors='coerce')
    X_test_proc[col] = pd.to_numeric(X_test_proc[col], errors='coerce')

# Sanitize feature names for LightGBM / XGBoost JSON support
clean_cols = [re.sub(r'[^A-Za-z0-9_]+', '_', c).strip('_') for c in X_train_proc.columns]
# Ensure uniqueness of feature names
seen = {}
unique_cols = []
for c in clean_cols:
    if c not in seen:
        seen[c] = 0
        unique_cols.append(c)
    else:
        seen[c] += 1
        unique_cols.append(f"{c}_{seen[c]}")

X_train_proc.columns = unique_cols
X_test_proc.columns = unique_cols

# K-Fold ensemble
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test_proc))
cat_preds = np.zeros(len(X_test_proc))
xgb_preds = np.zeros(len(X_test_proc))

for train_idx, val_idx in kf.split(X_train_proc, y):
    X_tr, y_tr = X_train_proc.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train_proc.iloc[val_idx], y[val_idx]
    
    # LightGBM
    lgb = LGBMRegressor(n_estimators=500, learning_rate=0.03, random_state=42, verbose=-1)
    lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[])
    lgb_preds += lgb.predict(X_test_proc) / kf.n_splits
    
    # CatBoost
    cb = CatBoostRegressor(iterations=600, learning_rate=0.04, random_seed=42, verbose=0)
    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)
    cat_preds += cb.predict(X_test_proc) / kf.n_splits
    
    # XGBoost
    xgb = XGBRegressor(n_estimators=500, learning_rate=0.03, random_state=42, verbosity=0)
    xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    xgb_preds += xgb.predict(X_test_proc) / kf.n_splits

final_preds = 0.4 * lgb_preds + 0.4 * cat_preds + 0.2 * xgb_preds

submission = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'Adjusted MLR': final_preds
})

submission.to_csv('predictions.csv', index=False)