import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/sf-building-permits/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/sf-building-permits/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'time_to_approve'
y_train = train_df[target_col].values

# Combine for uniform preprocessing
n_train = len(train_df)
df = pd.concat([train_df.drop(columns=[target_col]), test_df], ignore_index=True)

# 2. Date parsing & feature engineering
date_cols = [
    'Permit Creation Date',
    'Current Status Date',
    'Completed Date',
    'First Construction Document Date',
    'Permit Expiration Date'
]

for col in date_cols:
    if col in df.columns:
        dt = pd.to_datetime(df[col], errors='coerce')
        df[f'{col}_year'] = dt.dt.year
        df[f'{col}_month'] = dt.dt.month
        df[f'{col}_day'] = dt.dt.day
        df[f'{col}_dayofweek'] = dt.dt.dayofweek
        df[f'{col}_timestamp'] = dt.astype('int64') // 10**9
        df.loc[dt.isna(), f'{col}_timestamp'] = np.nan

# Compute date differences
if 'Permit Creation Date' in df.columns and 'First Construction Document Date' in df.columns:
    dt_create = pd.to_datetime(df['Permit Creation Date'], errors='coerce')
    dt_doc = pd.to_datetime(df['First Construction Document Date'], errors='coerce')
    diff_days = (dt_doc - dt_create).dt.total_seconds() / 86400.0
    df['diff_doc_create_days'] = diff_days
    df['log_diff_doc_create'] = np.log1p(np.maximum(0, diff_days))

if 'Permit Creation Date' in df.columns and 'Permit Expiration Date' in df.columns:
    dt_create = pd.to_datetime(df['Permit Creation Date'], errors='coerce')
    dt_exp = pd.to_datetime(df['Permit Expiration Date'], errors='coerce')
    diff_days_exp = (dt_exp - dt_create).dt.total_seconds() / 86400.0
    df['diff_exp_create_days'] = diff_days_exp

# 3. Location features
if 'Location' in df.columns:
    lat = df['Location'].astype(str).str.extract(r'\(([^,]+),')[0].astype(float)
    lon = df['Location'].astype(str).str.extract(r',\s*([^)]+)\)')[0].astype(float)
    df['latitude'] = lat
    df['longitude'] = lon

# 4. Text features on Description
if 'Description' in df.columns:
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', token_pattern=r'(?u)\b\w+\b')
    svd = TruncatedSVD(n_components=20, random_state=42)
    desc_clean = df['Description'].fillna('').astype(str)
    tfidf_mat = tfidf.fit_transform(desc_clean)
    svd_mat = svd.fit_transform(tfidf_mat)
    for i in range(20):
        df[f'desc_svd_{i}'] = svd_mat[:, i]

# 5. Drop raw text/date/id columns not needed directly
drop_cols = date_cols + ['Description', 'Location', 'Record ID', 'Permit Number']
drop_cols = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=drop_cols)

# 6. Categorical encoding
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    df[col] = df[col].astype('category')

X_train = df.iloc[:n_train].copy()
X_test = df.iloc[n_train:].copy()

# 7. Model Training with K-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'n_estimators': 1500,
    'learning_rate': 0.03,
    'num_leaves': 63,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    test_preds += model.predict(X_test) / kf.n_splits

# Ensure non-negative predictions
test_preds = np.clip(test_preds, a_min=0, a_max=None)

# 8. Save predictions
submission = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'time_to_approve': test_preds
})

submission.to_csv('predictions.csv', index=False)