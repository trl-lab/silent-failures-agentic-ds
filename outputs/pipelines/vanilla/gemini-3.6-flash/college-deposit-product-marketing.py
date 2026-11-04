```python
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold
import xgboost as xgb

warnings.filterwarnings("ignore")

# 1. Load data
train_path = "/path/to/user/workspace/ds_failures/datasets/strable/college-deposit-product-marketing/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/college-deposit-product-marketing/data/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = "TOTAL_ACTIVE_ACCOUNTS_"

# Drop missing target rows if any
train = train.dropna(subset=[target_col]).reset_index(drop=True)
y = train[target_col].values.astype(np.float64)

# Separate features
X_train = train.drop(columns=[target_col])
X_test = test.copy()

n_train = len(X_train)
n_test = len(X_test)

# Combine for uniform processing
df_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)

# Drop ID-like columns or constant columns
cols_to_drop = []
for c in df_all.columns:
  if c.lower() in [
      "id",
      "index",
      "unnamed: 0",
      "row_id",
  ] or df_all[c].nunique(dropna=False) <= 1:
    cols_to_drop.append(c)

df_all = df_all.drop(columns=cols_to_drop)

# Datetime parsing
for c in list(df_all.columns):
  if (
      df_all[c].dtype == "object"
      or "date" in c.lower()
      or "year" in c.lower()
      or "month" in c.lower()
  ):
    try:
      parsed = pd.to_datetime(df_all[c], errors="coerce")
      if parsed.notna().sum() > 0.5 * len(df_all):
        df_all[f"{c}_parsed_year"] = parsed.dt.year
        df_all[f"{c}_parsed_month"] = parsed.dt.month
        df_all[f"{c}_parsed_day"] = parsed.dt.day
        df_all[f"{c}_parsed_dayofweek"] = parsed.dt.dayofweek
    except Exception:
      pass

# Identify column types
cat_cols = []
num_cols = []

for c in df_all.columns:
  if (
      df_all[c].dtype == "object"
      or df_all[c].dtype.name == "category"
      or df_all[c].dtype == "bool"
  ):
    cat_cols.append(c)
  elif np.issubdtype(df_all[c].dtype, np.number):
    num_cols.append(c)

# Clean object/categorical features
for c in cat_cols:
  df_all[c] = df_all[c].fillna("missing").astype(str)

# Handle numerical infinite / missing values
for c in num_cols:
  df_all[c] = df_all[c].replace([np.inf, -np.inf], np.nan)

# Frequency encoding
for c in cat_cols:
  freq = df_all[c].value_counts().to_dict()
  df_all[f"{c}_freq"] = df_all[c].map(freq)

# Separate train and test
X_tr = df_all.iloc[:n_train].copy()
X_te = df_all.iloc[n_train:].copy()

# Out-of-fold Target Encoding
kf = KFold(n_splits=5, shuffle=True, random_state=42)
global_mean = float(np.mean(y))

for c in cat_cols:
  target_enc_tr = np.zeros(n_train)

  for tr_idx, val_idx in kf.split(X_tr):
    df_fold = pd.DataFrame(
        {"cat": X_tr.iloc[tr_idx][c].values, "target": y[tr_idx]}
    )
    means = df_fold.groupby("cat", observed=False)["target"].mean().to_dict()
    val_cats = X_tr.iloc[val_idx][c].values
    target_enc_tr[val_idx] = [means.get(val, global_mean) for val in val_cats]

  df_full = pd.DataFrame({"cat": X_tr[c].values, "target": y})
  means_full = df_full.groupby("cat", observed=False)["target"].mean().to_dict()
  te_cats = X_te[c].values
  target_enc_te = np.array([means_full.get(val, global_mean) for val in te_cats])

  X_tr[f"{c}_te"] = target_enc_tr
  X_te[f"{c}_te"] = target_enc_te

# Create LightGBM datasets with category dtype
X_tr_lgb = X_tr.copy()
X_te_lgb = X_te.copy()
for c in cat_cols:
  X_tr_lgb[c] = X_tr_lgb[c].astype("category")
  X_te_lgb[c] = X_te_lgb[c].astype("category")

# Create XGBoost numeric datasets with ordinal coding
X_tr_num = X_tr.copy()
X_te_num = X_te.copy()
for c in cat_cols:
  codes, uniques = pd.factorize(
      pd.concat([X_tr_num[c], X_te_num[c]], axis=0), sort=True
  )
  X_tr_num[c] = codes[:n_train]
  X_te_num[c] = codes[n_train:]

# Create CatBoost datasets with string dtype
X_tr_cb = X_tr.copy()
X_te_cb = X_te.copy()
for c in cat_cols:
  X_tr_cb[c] = X_tr_cb[c].astype(str)
  X_te_cb[c] = X_te_cb[c].astype(str)

# Arrays for OOF predictions and test predictions
oof_lgb = np.zeros(n_train)
oof_cb = np.zeros(n_train)
oof_xgb = np.zeros(n_train)

lgb_preds = np.zeros(n_test)
cb_preds = np.zeros(n_test)
xgb_preds = np.zeros(n_test)

# Train ensemble models across folds
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr)):
  y_tr, y_val = y[tr_idx], y[val_idx]

  # 1. LightGBM
  model_lgb = lgb.LGBMRegressor(
      n_estimators=1000,
      learning_rate=0.03,
      num_leaves=31,