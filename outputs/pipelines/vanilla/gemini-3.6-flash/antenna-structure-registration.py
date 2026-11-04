```python
import os
import re
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import xgboost as xgb

warnings.filterwarnings("ignore")

# Load data
train_path = "/path/to/user/workspace/ds_failures/datasets/strable/antenna-structure-registration/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/antenna-structure-registration/data/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = "Strucht"
y = train[target_col].values.astype(np.float32)

# Prepare combined dataframe for preprocessing
df_train = train.drop(columns=[target_col])
df_test = test.copy()
if target_col in df_test.columns:
  df_test = df_test.drop(columns=[target_col])

n_train = len(df_train)
df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)

# Categorize initial columns
date_cols = []
cat_cols = []
num_cols = []

for col in df_all.columns:
  if pd.api.types.is_datetime64_any_dtype(df_all[col]):
    date_cols.append(col)
  elif df_all[col].dtype == "object" or isinstance(
      df_all[col].dtype, pd.CategoricalDtype
  ):
    sample_s = df_all[col].dropna().astype(str)
    if len(sample_s) > 0 and sample_s.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.5:
      try:
        df_all[col] = pd.to_datetime(df_all[col], errors="coerce")
        date_cols.append(col)
      except Exception:
        cat_cols.append(col)
    else:
      cat_cols.append(col)
  else:
    num_cols.append(col)

# Date feature engineering
for col in date_cols:
  df_all[f"{col}_year"] = df_all[col].dt.year
  df_all[f"{col}_month"] = df_all[col].dt.month
  df_all[f"{col}_day"] = df_all[col].dt.day
  df_all[f"{col}_dayofweek"] = df_all[col].dt.dayofweek
  df_all[f"{col}_dayofyear"] = df_all[col].dt.dayofyear
  df_all[f"{col}_ts"] = (
      df_all[col].astype("int64", errors="ignore") // 10**9
  )
  df_all.drop(columns=[col], inplace=True)

# Coordinate feature engineering if lat/lon present
lat_cols = [c for c in df_all.columns if re.search(r"lat", c, re.IGNORECASE)]
lon_cols = [
    c for c in df_all.columns if re.search(r"lon|lng|long", c, re.IGNORECASE)
]

if lat_cols and lon_cols:
  try:
    lat = df_all[lat_cols[0]].astype(float)
    lon = df_all[lon_cols[0]].astype(float)
    df_all["coord_x"] = np.cos(np.radians(lat)) * np.cos(np.radians(lon))
    df_all["coord_y"] = np.cos(np.radians(lat)) * np.sin(np.radians(lon))
    df_all["coord_z"] = np.sin(np.radians(lat))
    df_all["coord_rot45"] = lat + lon
    df_all["coord_rot_45"] = lat - lon
  except Exception:
    pass

# Height / Elevation interactions
height_cols = [
    c
    for c in num_cols
    if any(k in c.lower() for k in ["hgt", "height", "elev", "alt"])
]
for i in range(len(height_cols)):
  for j in range(i + 1, len(height_cols)):
    c1, c2 = height_cols[i], height_cols[j]
    if (
        c1 in df_all.columns
        and c2 in df_all.columns
        and pd.api.types.is_numeric_dtype(df_all[c1])
        and pd.api.types.is_numeric_dtype(df_all[c2])
    ):
      df_all[f"{c1}_minus_{c2}"] = df_all[c1] - df_all[c2]
      df_all[f"{c1}_plus_{c2}"] = df_all[c1] + df_all[c2]
      df_all[f"{c1}_ratio_{c2}"] = df_all[c1] / (df_all[c2].abs() + 1e-5)

# Group aggregations
num_for_agg = [
    c for c in num_cols if c in df_all.columns and df_all[c].nunique() > 20
][:5]
cat_for_agg = [
    c
    for c in cat_cols
    if c in df_all.columns and 5 <= df_all[c].nunique() <= 1000
][:3]

for cat in cat_for_agg:
  for num in num_for_agg:
    grouped = df_all.groupby(cat)[num]
    df_all[f"{num}_mean_by_{cat}"] = grouped.transform("mean")
    df_all[f"{num}_std_by_{cat}"] = grouped.transform("std")

# Frequency & Label encoding for categoricals
for col in cat_cols:
  if col in df_all.columns:
    freq = df_all[col].value_counts(dropna=False).to_dict()
    df_all[f"{col}_freq"] = df_all[col].map(freq)
    df_all[col] = pd.factorize(df_all[col].astype(str))[0]

# Split back to train/test
df_train_feat = df_all.iloc[:n_train].copy()
df_test_feat = df_all.iloc[n_train:].copy()

# Out-of-fold target encoding for categoricals
kf = KFold(n_splits=5, shuffle=True, random_state=42)
te_cols = [
    col
    for col in cat_cols
    if col in df_train_feat.columns
    and 2 <= df_train_feat[col].nunique() <= 10000
]

for col in te_cols:
  te_train = np.zeros(n_train, dtype=np.float32)
  for tr_idx, va_idx in kf.split(df_train_feat):
    tr_series = df_train_feat[col].iloc[tr_idx]
    va_series = df_train_feat[col].iloc[va_idx]
    means = tr_series.groupby(tr_series).apply(lambda s: y[s.index].mean())
    global_mean = y[tr_idx].mean()
    te_train[va_idx] = va_series.map(means).fillna(global_mean).values

  means_all = df_train_feat[col].groupby(df_train_feat[col]).apply(
      lambda s: y[s.index].mean()
  )
  global_mean_all = y.mean()
  te_test = (
      df_test_feat[col].map(means_all).fillna(global_mean_all).values.astype(
          np.float32
      )
  )

  df_train_feat[f"{col}_te"] = te_train
  df_test_feat[f"{col}_te"] = te_test

# Column name sanitization
clean_cols = [
    re.sub(r"[^\w\s]", "_", str(col)).replace(" ", "_")
    for col in df_train_feat.columns
]
cols_seen = {}
final_cols = []
for col in clean_cols:
  if col not in cols_seen:
    cols_seen[col] = 0
    final_cols.append(col)
  else:
    cols_seen[col] += 1
    final_cols.append(f"{col}_{cols_seen[col]}")

df_train_feat.columns = final_cols
df_test_feat.columns = final_cols

# Convert all columns to numeric float32
for col in final_cols:
  df_train_feat[col] = pd.to_numeric(
      df_train_feat[col], errors="coerce"
  ).astype(np.float32)
  df_test_feat[col] = pd.to_numeric(df_test_feat[col], errors="coerce").astype(
      np.float32
  )

df_train_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
df_test_feat.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop useless constant or highly missing columns
drop_cols = []
for c in df_train_feat.columns:
  if (