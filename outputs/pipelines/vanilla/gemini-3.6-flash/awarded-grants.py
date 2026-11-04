import os
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor, early_stopping
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/awarded-grants/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/awarded-grants/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = 'Financial Assistance'
y = train[target_col].values.astype(np.float64)

X_train_raw = train.drop(columns=[target_col])
X_test_raw = test.copy()

common_cols = [c for c in X_train_raw.columns if c in X_test_raw.columns]
X_train_raw = X_train_raw[common_cols]
X_test_raw = X_test_raw[common_cols]

n_train = len(train)
df_all = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

datetime_cols = []
for col in df_all.columns:
    if pd.api.types.is_datetime64_any_dtype(df_all[col]):
        datetime_cols.append(col)
    elif df_all[col].dtype == 'object':
        sample = df_all[col].dropna().head(50)
        if len(sample) > 0:
            try:
                pd.to_datetime(sample)
                df_all[col] = pd.to_datetime(df_all[col], errors='coerce')
                datetime_cols.append(col)
            except Exception:
                pass

for col in datetime_cols:
    df_all[f'{col}_year'] = df_all[col].dt.year
    df_all[f'{col}_month'] = df_all[col].dt.month
    df_all[f'{col}_day'] = df_all[col].dt.day
    df_all[f'{col}_dayofweek'] = df_all[col].dt.dayofweek
    df_all[f'{col}_dayofyear'] = df_all[col].dt.dayofyear
    df_all.drop(columns=[col], inplace=True)

obj_cols = df_all.select_dtypes(include=['object', 'category']).columns.tolist()

for col in obj_cols:
    freq = df_all[col].value_counts(dropna=False).to_dict()
    df_all[f'{col}_freq'] = df_all[col].map(freq).astype(np.float32)
    df_all[f'{col}_cat'] = pd.factorize(df_all[col].astype(str))[0]

if len(obj_cols) > 0:
    text_data = df_all[obj_cols].astype(str).agg(' '.join, axis=1)
    
    tfidf = TfidfVectorizer(max_features=35000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_tfidf = tfidf.fit_transform(text_data)
    
    n_comp = min(20, X_tfidf.shape[1] - 1) if X_tfidf.shape[1] > 1 else 0
    if n_comp > 0:
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        X_svd = svd.fit_transform(X_tfidf)
        for i in range(X_svd.shape[1]):
            df_all[f'svd_{i}'] = X_svd[:, i]
            
    if X_tfidf.shape[1] > 0:
        kf_text = KFold(n_splits=5, shuffle=True, random_state=42)
        X_tfidf_tr = X_tfidf[:n_train]
        X_tfidf_te = X_tfidf[n_train:]
        
        oof_ridge = np.zeros(n_train)
        test_ridge = np.zeros(len(test))
        
        for tr_idx, va_idx in kf_text.split(X_tfidf_tr):
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_tfidf_tr[tr_idx], y[tr_idx])
            oof_ridge[va_idx] = ridge.predict(X_tfidf_tr[va_idx])
            test_ridge += ridge.predict(X_tfidf_te) / kf_text.n_splits
            
        df_all['ridge_text_pred'] = np.concatenate([oof_ridge, test_ridge])

df_all.drop(columns=obj_cols, inplace=True,