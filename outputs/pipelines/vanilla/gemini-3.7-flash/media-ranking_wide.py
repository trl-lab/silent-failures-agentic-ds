import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/media-ranking_wide/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/media-ranking_wide/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

train_df = train_df.dropna(subset=['Overall']).reset_index(drop=True)
y_train = train_df['Overall'].values

# Feature engineering
def preprocess(df):
    df_feat = df.copy()
    str_cols = ['Media', 'Domain', 'Country', 'Region', 'Language', 'Typology']
    for col in str_cols:
        if col in df_feat.columns:
            df_feat[col] = df_feat[col].fillna('Missing').astype(str)
        else:
            df_feat[col] = 'Missing'
            
    df_feat['tld'] = df_feat['Domain'].apply(lambda x: x.split('.')[-1].lower() if '.' in x else 'none')
    df_feat['media_len'] = df_feat['Media'].str.len()
    df_feat['media_words'] = df_feat['Media'].apply(lambda x: len(x.split()))
    df_feat['domain_len'] = df_feat['Domain'].str.len()
    df_feat['domain_dots'] = df_feat['Domain'].apply(lambda x: x.count('.'))
    df_feat['all_text'] = (
        df_feat['Media'] + ' ' + 
        df_feat['Domain'] + ' ' + 
        df_feat['Country'] + ' ' + 
        df_feat['Region'] + ' ' + 
        df_feat['Language'] + ' ' + 
        df_feat['Typology']
    )
    return df_feat

X_train_df = preprocess(train_df.drop(columns=['Overall']))
X_test_df = preprocess(test_df)

cat_cols = ['Country', 'Region', 'Language', 'Typology', 'tld']
num_cols = ['media_len', 'media_words', 'domain_len', 'domain_dots']

# Frequency encoding
for col in cat_cols + ['Media', 'Domain']:
    freq = pd.concat([X_train_df[col], X_test_df[col]]).value_counts()
    X_train_df[f'{col}_freq'] = X_train_df[col].map(freq).fillna(0)
    X_test_df[f'{col}_freq'] = X_test_df[col].map(freq).fillna(0)

freq_cols = [f'{col}_freq' for col in cat_cols + ['Media', 'Domain']]

# 2. Out-of-fold Target Encoding & Model Training
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# TF-IDF on all_text
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b')
X_train_tfidf = tfidf.fit_transform(X_train_df['all_text'])
X_test_tfidf = tfidf.transform(X_test_df['all_text'])

# OneHotEncoder for low-to-medium cardinality categoricals
ohe = OneHotEncoder(handle_unknown='ignore')
X_train_ohe = ohe.fit_transform(X_train_df[cat_cols])
X_test_ohe = ohe.transform(X_test_df[cat_cols])

# Convert categorical columns to category dtype for LightGBM
for col in cat_cols:
    X_train_df[col] = X_train_df[col].astype('category')
    X_test_df[col] = X_test_df[col].astype('category')

features_lgb = cat_cols + num_cols + freq_cols

# Array allocations
oof_ridge = np.zeros(len(train_df))
test_preds_ridge = np.zeros(len(test_df))

oof_lgb = np.zeros(len(train_df))
test_preds_lgb = np.zeros(len(test_df))

oof_cb = np.zeros(len(train_df))
test_preds_cb = np.zeros(len(test_df))

from scipy.sparse import hstack

X_train_sparse = hstack([X_train_tfidf, X_train_ohe, X_train_df[num_cols + freq_cols].values]).tocsr()
X_test_sparse = hstack([X_test_tfidf, X_test_ohe, X_test_df[num_cols + freq_cols].values]).tocsr()

# Target encoding features for CatBoost
cb_cat_cols = ['Country', 'Region', 'Language', 'Typology', 'tld']
X_train_cb = X_train_df[cb_cat_cols + num_cols + freq_cols].copy()
X_test_cb = X_test_df[cb_cat_cols + num_cols + freq_cols].copy()
for col in cb_cat_cols:
    X_train_cb[col] = X_train_cb[col].astype(str)
    X_test_cb[col] = X_test_cb[col].astype(str)

for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    y_tr, y_va = y_train[train_idx], y_train[val_idx]
    
    # 1. Ridge Regressor on Sparse Features
    ridge = Ridge(alpha=1.5, random_state=42)
    ridge.fit(X_train_sparse[train_idx], y_tr)
    oof_ridge[val_idx] = ridge.predict(X_train_sparse[val_idx])
    test_preds_ridge += ridge.predict(X_test_sparse) / n_splits
    
    # 2. LightGBM Regressor
    X_tr_lgb = X_train_df.loc[train_idx, features_lgb]
    X_va_lgb = X_train_df.loc[val_idx, features_lgb]
    
    lgb = LGBMRegressor(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbose=-1
    )
    lgb.fit(
        X_tr_lgb, y_tr,
        eval_set=[(X_va_lgb, y_va)],
        callbacks=[]
    )
    oof_lgb[val_idx] = lgb.predict(X_va_lgb)
    test_preds_lgb += lgb.predict(X_test_df[features_lgb]) / n_splits
    
    # 3. CatBoost Regressor
    X_tr_cb = X_train_cb.loc[train_idx]
    X_va_cb = X_train_cb.loc[val_idx]
    
    cb = CatBoostRegressor(
        iterations=700,
        learning_rate=0.04,
        depth=6,
        cat_features=cb_cat_cols,
        random_seed=42 + fold,
        verbose=0
    )
    cb.fit(X_tr_cb, y_tr, eval_set=(X_va_cb, y_va), early_stopping_rounds=50, verbose=False)
    oof_cb[val_idx] = cb.predict(X_va_cb)
    test_preds_cb += cb.predict(X_test_cb) / n_splits

# Ensemble weights using simple linear regression / ridge on OOF
oof_matrix = np.column_stack([oof_ridge, oof_lgb, oof_cb])
test_matrix = np.column_stack([test_preds_ridge, test_preds_lgb, test_preds_cb])

meta_model = Ridge(alpha=10.0, positive=True)
meta_model.fit(oof_matrix, y_train)
final_predictions = meta_model.predict(test_matrix)

# 3. Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'Overall': final_predictions
})
output_df.to_csv('predictions.csv', index=False)