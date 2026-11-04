import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
import xgboost as xgb

warnings.filterwarnings('ignore')

train_path = (
    '/path/to/user/workspace/ds_failures/datasets/strable/hospitals/data/train.parquet'
)
test_path = (
    '/path/to/user/workspace/ds_failures/datasets/strable/hospitals/data/test.parquet'
)

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = 'BEDS'
train = train[train[target_col].notna()].copy()
train = train[train[target_col] >= 0].copy()

y = train[target_col].values
X_train = train.drop(columns=[target_col]).copy()
X_test = test.drop(columns=[target_col], errors='ignore').copy()

n_train = len(X_train)
df_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)

for col in df_all.columns:
  if pd.api.types.is_numeric_dtype(df_all[col]):
    df_all.loc[df_all[col] <= -900, col] = np.nan

for col in list(df_all.columns):
  if 'DATE' in col.upper() or 'TIME' in col.upper():
    try:
      dt = pd.to_datetime(df_all[col], errors='coerce')
      if dt.notna().sum() > 0:
        df_all[col + '_year'] = dt.dt.year
        df_all[col + '_month'] = dt.dt.month
        df_all[col + '_day'] = dt.dt.day
    except Exception:
      pass

name_cols = [
    c
    for c in df_all.columns
    if 'NAME' in c.upper()
    and not c.endswith('_freq')
    and not pd.api.types.is_numeric_dtype(df_all[c])
]

tfidf_dfs = []
for nc in name_cols:
  try:
    vec = TfidfVectorizer(
        max_features=50, stop_words='english', ngram_range=(1, 2)
    )
    text_data = df_all[nc].fillna('').astype(str)
    tfidf_mat = vec.fit_transform(text_data).toarray()
    if tfidf_mat.shape[1] > 0:
      cols = [f'tfidf_{nc}_{i}' for i in range(tfidf_mat.shape[1])]
      tfidf_dfs.append(pd.DataFrame(tfidf_mat, columns=cols))
  except Exception:
    pass

cat_cols = []
for col in list(df_all.columns):
  if (
      df_all[col].dtype == 'object'
      or str(df_all[col].dtype) == 'string'
      or df_all[col].dtype.name == 'category'
  ):
    val_counts = df_all[col].value_counts(dropna=False)
    df_all[col + '_freq'] = df_all[col].map(val_counts)
    cat_cols.append(col)

if tfidf_dfs:
  df_all = pd.concat([df_all] + tfidf_dfs, axis=1)

df_lgb = df_all.copy()
for c in cat_cols:
  df_lgb[c] = df_lgb[c].astype(str).astype('category')

df_cb = df_all.copy()
for c in cat_cols:
  df_cb[c] = df_cb[c].fillna('MISSING').astype(str)

X_train_lgb = df_lgb.iloc[:n_train].copy()
X_test_lgb = df_lgb.iloc[n_train:].copy()