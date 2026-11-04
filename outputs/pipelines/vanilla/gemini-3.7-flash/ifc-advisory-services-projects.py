import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/ifc-advisory-services-projects/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/ifc-advisory-services-projects/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Estimated Total Budget ($)'
y = train_df[target_col].values

# Combine for uniform preprocessing
n_train = len(train_df)
df = pd.concat([train_df.drop(columns=[target_col]), test_df], axis=0).reset_index(drop=True)

# Parse date columns
date_cols = ['Disclosure Date', 'IFC Approval Date', 'Projected Start Date']
for col in date_cols:
    parsed_date = pd.to_datetime(df[col], errors='coerce')
    df[f'{col}_year'] = parsed_date.dt.year
    df[f'{col}_month'] = parsed_date.dt.month
    df[f'{col}_day'] = parsed_date.dt.day
    df[f'{col}_dayofweek'] = parsed_date.dt.dayofweek
    df[f'{col}_timestamp'] = parsed_date.astype('int64') // 10**9
    # Handle NaNs in timestamp resulting from NaT
    df.loc[parsed_date.isna(), f'{col}_timestamp'] = np.nan

# Date difference features
d_disc = pd.to_datetime(df['Disclosure Date'], errors='coerce')
d_appr = pd.to_datetime(df['IFC Approval Date'], errors='coerce')
d_start = pd.to_datetime(df['Projected Start Date'], errors='coerce')

df['diff_disc_appr'] = (d_disc - d_appr).dt.days
df['diff_appr_start'] = (d_appr - d_start).dt.days
df['diff_disc_start'] = (d_disc - d_start).dt.days

# Text features from Project Name
text_col = df['Project Name'].fillna('').astype(str)
df['name_len'] = text_col.str.len()
df['name_word_count'] = text_col.apply(lambda x: len(x.split()))

tfidf = TfidfVectorizer(max_features=5000, stop_words='english', token_pattern=r'(?u)\b\w+\b')
tfidf_matrix = tfidf.fit_transform(text_col)
svd = TruncatedSVD(n_components=10, random_state=42)
svd_feats = svd.fit_transform(tfidf_matrix)
for i in range(10):
    df[f'tfidf_svd_{i}'] = svd_feats[:, i]

# Categorical columns
cat_cols = ['Country', 'IFC Country Code', 'IFC Region', 'Business Line', 'Status', 'WB Country Code']
for col in cat_cols:
    df[col] = df[col].astype('category')

# Frequency encoding for categoricals
for col in cat_cols:
    freq = df[col].value_counts(normalize=True)
    df[f'{col}_freq'] = df[col].map(freq).astype(float)

# Drop raw text / url / date string columns
drop_cols = ['Project Name', 'Project URL'] + date_cols
df = df.drop(columns=drop_cols)

X = df.iloc[:n_train].copy()
X_test = df.iloc[n_train:].copy()

# Cross-validation training
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))

cat_feature_names = cat_cols

for train_idx, val_idx in kf.split(X, y):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]
    
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
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_preds += lgb_model.predict(X_test) / kf.n_splits

    # CatBoost
    X_tr_cb = X_tr.copy()
    X_va_cb = X_va.copy()
    X_test_cb = X_test.copy()
    for col in cat_cols:
        X_tr_cb[col] = X_tr_cb[col].astype(str).fillna('missing')
        X_va_cb[col] = X_va_cb[col].astype(str).fillna('missing')
        X_test_cb[col] = X_test_cb[col].astype(str).fillna('missing')
        
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
        eval_set=(X_va_cb, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cb_preds += cb_model.predict(X_test_cb) / kf.n_splits

# Ensemble predictions
final_preds = 0.5 * lgb_preds + 0.5 * cb_preds

# Save predictions
sub = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'Estimated Total Budget ($)': final_preds
})
sub.to_csv('predictions.csv', index=False)