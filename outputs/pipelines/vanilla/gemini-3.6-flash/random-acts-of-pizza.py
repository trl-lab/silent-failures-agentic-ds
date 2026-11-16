import os
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

train_path = '/path/to/user/workspace/ds_failures/datasets/mlebench/random-acts-of-pizza/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/mlebench/random-acts-of-pizza/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

y = train['requester_received_pizza'].astype(int).values

def stringify(val):
    if val is None:
        return ""
    if isinstance(val, (list, np.ndarray)):
        return " ".join([str(x) for x in val if x is not None and str(x) != "None"])
    if pd.isna(val):
        return ""
    return str(val)

common_cols = [c for c in train.columns if c in test.columns and c != 'requester_received_pizza']

def get_text_col(df, col_name):
    if col_name in df.columns:
        return df[col_name].apply(stringify)
    return pd.Series([""] * len(df))

train_title = get_text_col(train, 'request_title')
test_title = get_text_col(test, 'request_title')

train_text = get_text_col(train, 'request_text_edit_aware')
test_text = get_text_col(test, 'request_text_edit_aware')

train_subreddits = get_text_col(train, 'requester_subreddits_at_request')
test_subreddits = get_text_col(test, 'requester_subreddits_at_request')

train_full_text = train_title + " " + train_text
test_full_text = test_title + " " + test_text

def extract_text_stats(text_series, title_series):
    stats = pd.DataFrame()
    stats['text_len'] = text_series.apply(len)
    stats['title_len'] = title_series.apply(len)
    stats['text_word_count'] = text_series.apply(lambda x: len(x.split()))
    stats['title_word_count'] = title_series.apply(lambda x: len(x.split()))
    stats['has_exclamation'] = text_series.apply(lambda x: int('!' in x))
    stats['has_question'] = text_series.apply(lambda x: int('?' in x))
    
    keywords = ['please', 'thank', 'thanks', 'hungry', 'job', 'student', 'paycheck', 'broke', 'rent', 'family', 'kids', 'sad', 'help', 'reciprocate', 'payback']
    for kw in keywords:
        stats[f'kw_{kw}'] = text_series.apply(lambda x: int(kw in x.lower()))
    return stats

train_stats = extract_text_stats(train_text, train_title)
test_stats = extract_text_stats(test_text, test_title)

num_cols = []
for c in common_cols:
    if c in ['request_id', 'requester_username', 'request_title', 'request_text', 'request_text_edit_aware', 'requester_subreddits_at_request']:
        continue
    if np.issubdtype(train[c].dtype, np.number) or np.issubdtype(train[c].dtype, np.bool_):
        num_cols.append(c)

train_num = train[num_cols].astype(float).copy()
test_num = test[num_cols].astype(float).copy()

for df in [train_num, test_num]:
    if 'requester_upvotes_minus_downvotes_at_request' in df.columns and 'requester_upvotes_plus_downvotes_at_request' in df.columns:
        df['upvote_ratio'] = df['requester_upvotes_minus_downvotes_at_request'] / (df['requester_upvotes_plus_downvotes_at_request'] + 1.0)
    if 'unix_timestamp_of_request_utc' in df.columns:
        dt = pd.to_datetime(df['unix_timestamp_of_request_utc'], unit='s', errors='coerce')
        df['hour'] = dt.dt.hour
        df['dayofweek'] = dt.dt.dayofweek
        df['day'] = dt.dt.day

for col in train_num.columns:
    med = train_num[col].median()
    if np.isnan(med):
        med = 0.0
    train_num[col] = train_num[col].fillna(med)
    test_num[col] = test_num[col].fillna(med)

tfidf_text = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True, stop_words='english')
X_text_tr = tfidf_text.fit_transform(train_full_text)
X_text_te = tfidf_text.transform(test_full_text)

svd_text = TruncatedSVD(n_components=50, random_state=42)
svd_text_tr = svd_text.fit_transform(X_text_tr)
svd_text_te = svd_text.transform(X_text_te)

tfidf_sub = TfidfVectorizer(max_features=3000, sublinear_tf=True)
X_sub_tr = tfidf_sub.fit_transform(train_subreddits)
X_sub_te = tfidf_sub.transform(test_subreddits)

svd_sub = TruncatedSVD(n_components=20, random_state=42)
svd_sub_tr = svd_sub.fit_transform(X_sub_tr)
svd_sub_te = svd_sub.transform(X_sub_te)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_text_lr = np.zeros(len(train))
test_text_lr = np.zeros(len(test))

for tr_idx, va_idx in skf.split(X_text_tr, y):
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(X_text_tr[tr_idx], y[tr_idx])
    oof_text_lr[va_idx] = lr.predict_proba(X_text_tr[va_idx])[:, 1]
    test_text_lr += lr.predict_proba(X_text_te)[:, 1] / 5.0

oof_sub_lr = np.zeros(len(train))
test_sub_lr = np.zeros(len(test))

for tr_idx, va_idx in skf.split(X_sub_tr, y):
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(X_sub_tr[tr_idx], y[tr_idx])
    oof_sub_lr[va_idx] = lr.predict_proba(X_sub_tr[va_idx])[:, 1]
    test_sub_lr += lr.predict_proba(X_sub_te)[:, 1] / 5.0

X_tr =