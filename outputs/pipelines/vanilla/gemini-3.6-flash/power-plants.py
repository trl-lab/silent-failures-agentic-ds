import os
import random
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import xgboost as xgb

warnings.filterwarnings("ignore")


def seed_everything(seed=42):
  random.seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)
  np.random.seed(seed)


seed_everything(42)

train_path = "/path/to/user/workspace/ds_failures/datasets/strable/power-plants/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/power-plants/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "NAMPLT_CAP"

train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)
y_train = train_df[target_col].values.astype(np.float64)

X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

for col in ["id", "ID", "Id"]:
  if col in X_train.columns and col in X_test.columns:
    X_train = X_train.drop(columns=[col])
    X_test = X_test.drop(columns=[col])

n_train = len(X_train)
combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)

datetime_cols = []
cat_cols = []
num_cols = []

for col in combined.columns:
  if pd.api.types.is_datetime64_any_dtype(combined[col]):
    datetime_cols.append(col)
  elif combined[col].dtype == "object" or isinstance(
      combined[col].dtype, pd.CategoricalDtype
  ):
    col_str = str(col).lower()
    if "date" in col_str or "time" in col_str:
      try:
        combined[col] = pd.to_datetime(combined[col])
        datetime_cols.append(col)
        continue
      except Exception:
        pass
    cat_cols.append(col)
  else:
    num_cols.append(col)

for col in datetime_cols:
  combined[f"{col}_year"] = combined[col].dt.year
  combined[f"{col}_month"] = combined[col].dt.month
  combined[f"{col}_day"] = combined[col].dt.day
  combined[f"{col}_dayofweek"] = combined[col].dt.dayofweek
  combined = combined.drop(columns=[col])

for col in cat_cols:
  combined[col] = combined[col].fillna("missing").astype(str)
  freq = combined[col].value_counts()
  combined[f"{col}_freq"] = combined[col].map(freq).astype(np.float32)
  combined[col] = combined[col].astype("category")

for col in num_cols:
  combined[col] = pd.to_numeric(combined[col], errors="coerce").astype(
      np.float32
  )

X_train_proc = combined.iloc[:n_train].copy().reset_index(drop=True)
X_test_proc = combined.iloc[n_train:].copy().reset_index(drop=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
global_mean = float(np.mean(y_train))

for col in cat_cols:
  train_te = pd.Series(index=X_train_proc.index, dtype=float)
  for tr_idx, val_idx in kf.split(X_train_proc):
    tr_sub_cat = X_train_proc[col].iloc[tr_idx]
    tr_sub_y = y_train[tr_idx]
    grouped = (
        pd.DataFrame({"cat": tr_sub_cat, "y": tr_sub_y})
        .groupby("cat", observed=False)["y"]
        .agg(["count", "mean"])
    )
    smooth = 10.0
    smoothed = (grouped["count"] * grouped["mean"] + smooth * global_mean) / (
        grouped["count"] + smooth