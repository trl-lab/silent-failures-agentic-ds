import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
from catboost import CatBoostRegressor

# 1. Load Data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/fts-funding/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/fts-funding/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'amountUSD'
y_train = train_df[target_col].values
n_train = len(train_df)
n_test = len(test_df)

# Combine for uniform preprocessing
df = pd.concat([train_df.drop(columns=[target_col]), test_df], axis=0, ignore_index=True)

# 2. Date features
date_cols = ['date', 'firstReportedDate', 'decisionDate', 'createdAt', 'updatedAt']
for col in date_cols:
    if col in df.columns:
        dt = pd.to_datetime(df[col], errors='coerce')
        df[f'{col}_year'] = dt.dt.year.astype(float)
        df[f'{col}_month'] = dt.dt.month.astype(float)
        df[f'{col}_day'] = dt.dt.day.astype(float)
        df[f'{col}_dayofweek'] = dt.dt.dayofweek.astype(float)
        df[f'{col}_ts'] = dt.astype('int64') / 1e11
        df[f'{col}_ts'] = df[f'{col}_ts'].mask(dt.isna(), np.nan)
        df.drop(columns=[col], inplace=True)

# 3. Numeric differences and interaction features
if 'srcUsageYearStart' in df.columns and 'srcUsageYearEnd' in df.columns:
    df['src_usage_duration'] = df['srcUsageYearEnd'] - df['srcUsageYearStart']
if 'destUsageYearStart' in df.columns and 'destUsageYearEnd' in df.columns:
    df['dest_usage_duration'] = df['destUsageYearEnd'] - df['destUsageYearStart']

# 4. Text features from description
if 'description' in df.columns:
    df['desc_len'] = df['description'].fillna('').astype(str).str.len()
    df['desc_word_count'] = df['description'].fillna('').astype(str).apply(lambda x: len(x.split()))
    
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', token_pattern=r'(?u)\b\w+\b')
    desc_text = df['description'].fillna('').astype(str)
    tfidf_mat = tfidf.fit_transform(desc_text)
    
    svd = TruncatedSVD(n_components=20, random_state=42)
    svd_feats = svd.fit_transform(tfidf_mat)
    for i in range(20):
        df[f'desc_svd_{i}'] = svd_feats[:, i]
    df.drop(columns=['description'], inplace=True)

# 5. Handle remaining categorical and object columns
cat_cols = []
for col in df.columns:
    if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
        cat_cols.append(col)
        # Frequency encoding
        freq = df[col].value_counts(dropna=False)
        df[f'{col}_freq'] = df[col].map(freq)
        df[col] = df[col].astype(str).astype('category')

# Separate train and test
X_train = df.iloc[:n_train].copy()
X_test = df.iloc[n_train:].copy()

# 6. Cross-Validation and Model Training
kf = KFold(n_splits=5, shuffle=True, random_state=42)
lgb_preds = np.zeros(n_test)
cat_preds = np.zeros(n_test)

# Categorical column names for LightGBM
categorical_feature_names = [col for col in cat_cols if col in X_train.columns]

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'n_estimators': 1200,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_preds += lgb_model.predict(X_test) / kf.n_splits

    # CatBoost
    # Prepare data for CatBoost (convert category to str and fillna)
    X_tr_cb = X_tr.copy()
    X_va_cb = X_va.copy()
    X_te_cb = X_test.copy()
    for c in categorical_feature_names:
        X_tr_cb[c] = X_tr_cb[c].astype(str).fillna('missing')
        X_va_cb[c] = X_va_cb[c].astype(str).fillna('missing')
        X_te_cb[c] = X_te_cb[c].astype(str).fillna('missing')

    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.06,
        depth=6,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42 + fold,
        cat_features=categorical_feature_names,
        verbose=0
    )
    cat_model.fit(
        X_tr_cb, y_tr,
        eval_set=(X_va_cb, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cat_preds += cat_model.predict(X_te_cb) / kf.n_splits

# Ensemble predictions
final_preds = 0.5 * lgb_preds + 0.5 * cat_preds

# 7. Save submission
submission_df = pd.DataFrame({
    'id': np.arange(n_test),
    'amountUSD': final_preds
})
submission_df.to_csv('predictions.csv', index=False)