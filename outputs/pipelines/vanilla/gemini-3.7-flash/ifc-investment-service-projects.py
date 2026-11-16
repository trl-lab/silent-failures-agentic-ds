import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
from catboost import CatBoostRegressor

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/ifc-investment-service-projects/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/ifc-investment-service-projects/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Total IFC investment as approved by Board(Million - USD)'
y_train = train_df[target_col].values

# Combine datasets for consistent feature engineering
combined_df = pd.concat([train_df.drop(columns=[target_col]), test_df], ignore_index=True)

# Date feature engineering
date_cols = ['Date Disclosed', 'Projected Board Date', 'IFC Approval Date', 'IFC Signed Date', 'IFC Invested Date']
for col in date_cols:
    dt = pd.to_datetime(combined_df[col], errors='coerce')
    combined_df[f'{col}_year'] = dt.dt.year.astype('float32')
    combined_df[f'{col}_month'] = dt.dt.month.astype('float32')
    combined_df[f'{col}_day'] = dt.dt.day.astype('float32')
    combined_df[f'{col}_timestamp'] = dt.astype('int64') // 10**9
    # Handle NaT converted to large negative numbers in timestamp
    combined_df.loc[dt.isna(), f'{col}_timestamp'] = np.nan

# Date differences in days
combined_df['diff_approval_disclosed'] = (
    pd.to_datetime(combined_df['IFC Approval Date'], errors='coerce') - 
    pd.to_datetime(combined_df['Date Disclosed'], errors='coerce')
).dt.days

combined_df['diff_signed_approval'] = (
    pd.to_datetime(combined_df['IFC Signed Date'], errors='coerce') - 
    pd.to_datetime(combined_df['IFC Approval Date'], errors='coerce')
).dt.days

combined_df['diff_invested_signed'] = (
    pd.to_datetime(combined_df['IFC Invested Date'], errors='coerce') - 
    pd.to_datetime(combined_df['IFC Signed Date'], errors='coerce')
).dt.days

# Text features via TF-IDF + SVD
text_col = (
    combined_df['Project Name'].fillna('') + ' ' +
    combined_df['Company Name'].fillna('') + ' ' +
    combined_df['Industry'].fillna('') + ' ' +
    combined_df['Department'].fillna('')
)

tfidf = TfidfVectorizer(max_features=5000, stop_words='english', token_pattern=r'(?u)\b\w+\b')
tfidf_mat = tfidf.fit_transform(text_col)

svd = TruncatedSVD(n_components=20, random_state=42)
svd_feats = svd.fit_transform(tfidf_mat)
for i in range(20):
    combined_df[f'text_svd_{i}'] = svd_feats[:, i]

# Categorical columns
cat_cols = ['Document Type', 'Product Line', 'Country', 'IFC Country Code', 
            'Industry', 'Environmental Category', 'Department', 'Status', 'WB Country Code']

for col in cat_cols:
    combined_df[col] = combined_df[col].astype('category')

# Drop raw text / raw date columns
cols_to_drop = date_cols + ['Project Name', 'Project Url', 'Company Name']
combined_df = combined_df.drop(columns=[c for c in cols_to_drop if c in combined_df.columns])

# Split back to train and test
n_train = len(train_df)
X_train = combined_df.iloc[:n_train].copy()
X_test = combined_df.iloc[n_train:].copy()

# Cross-validated modeling
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))

# Categorical column names for LightGBM and CatBoost
cat_features_names = [col for col in cat_cols if col in X_train.columns]

# For CatBoost, fill NaNs in categorical columns
X_train_cb = X_train.copy()
X_test_cb = X_test.copy()
for col in cat_features_names:
    X_train_cb[col] = X_train_cb[col].astype(str).fillna('missing')
    X_test_cb[col] = X_test_cb[col].astype(str).fillna('missing')

for train_idx, val_idx in kf.split(X_train):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_preds += lgb_model.predict(X_test) / kf.n_splits

    # CatBoost
    X_tr_cb, y_tr_cb = X_train_cb.iloc[train_idx], y_train[train_idx]
    X_va_cb, y_va_cb = X_train_cb.iloc[val_idx], y_train[val_idx]
    
    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.04,
        depth=6,
        cat_features=cat_features_names,
        random_seed=42,
        verbose=0
    )
    cb_model.fit(
        X_tr_cb, y_tr_cb,
        eval_set=(X_va_cb, y_va_cb),
        early_stopping_rounds=50
    )
    cb_preds += cb_model.predict(X_test_cb) / kf.n_splits

# Ensemble predictions
final_predictions = 0.5 * lgb_preds + 0.5 * cb_preds

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(test_df)),
    target_col: final_predictions
})
output_df.to_csv('predictions.csv', index=False)