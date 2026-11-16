import pandas as pd
import numpy as np
import re
import os
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

SEED = 42
np.random.seed(SEED)

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/drug-drugsfda/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/drug-drugsfda/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'marketing_status'

y_raw = train_df[target_col].astype(str)
le = LabelEncoder()
y = le.fit_transform(y_raw)

X_train = train_df.drop(columns=[target_col], errors='ignore')
X_test = test_df.drop(columns=[target_col], errors='ignore')

id_cols = [c for c in X_train.columns if c.lower() in ['id', 'unnamed: 0', 'index']]
if id_cols:
    X_train = X_train.drop(columns=id_cols)
    X_test = X_test.drop(columns=id_cols)

train_len = len(X_train)
df_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)

new_cols = []
seen = {}
for col in df_all.columns:
    clean_col = re.sub(r'[^\w]', '_', str(col))
    if clean_col in seen:
        seen[clean_col] += 1
        clean_col = f"{clean_col}_{seen[clean_col]}"
    else:
        seen[clean_col] = 0
    new_cols.append(clean_col)
df_all.columns = new_cols

for c in df_all.columns:
    s = df_all[c]
    non_nulls = s.dropna()
    if not non_nulls.empty:
        first_val = non_nulls.iloc[0]
        if isinstance(first_val, (list, dict, tuple, np.ndarray)):
            df_all[c] = s.apply(lambda x: ' '.join(map(str, x)) if isinstance(x, (list, tuple, np.ndarray)) else (str(x) if x is not None else ''))

cat_cols = []
num_cols = []
for c in df_all.columns:
    if np.issubdtype(df_all[c].dtype, np.number):
        num_cols.append(c)
    else:
        cat_cols.append(c)

if len(cat_cols) > 0:
    all_str_text = df_all[cat_cols].fillna('').astype(str).agg(' '.join, axis=1)
    try:
        tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=2, token_pattern=r'(?u)\b\w+\b')
        text_mat = tfidf.fit_transform(all_str_text)
        n_comp = min(32, text_mat.shape[1] - 1)
        if n_comp > 1:
            svd = TruncatedSVD(n_components=n_comp, random_state=SEED)
            svd_feats = svd.fit_transform(text_mat)
            for i in range(n_comp):
                df_all[f'svd_{i}'] = svd_feats[:, i]
    except Exception:
        pass

for c in cat_cols:
    freq = df_all[c].astype(str).value_counts(normalize=True).to_dict()
    df_all[f'{c}_freq'] = df_all[c].astype(str).map(freq).fillna(0)
    df_all[f'{c}_code'] = LabelEncoder().fit_transform(df_all[c].fillna('missing').astype(str))
    df_all[c] = df_all[c].fillna('missing').astype(str).astype('category')