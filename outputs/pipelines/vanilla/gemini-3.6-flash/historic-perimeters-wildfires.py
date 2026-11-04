import os
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import xgboost as xgb

warnings.filterwarnings('ignore')

train_path = (
    "/path/to/user/workspace/ds_failures/datasets/strable/historic-perimeters-wildfires/data/train.parquet"
)
test_path = (
    "/path/to/user/workspace/ds_failures/datasets/strable/historic-perimeters-wildfires/data/test.parquet"
)

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

if 'gisacres' in train.columns:
  train = train.dropna(subset=['gisacres']).reset_index(drop=True)

y = train['gisacres'].values
X = train.drop(columns=['gisacres'], errors='ignore')
X_test = test.drop(columns=['gisacres'], errors='ignore')

n_train = len(X)
combined = pd.concat([X, X_test], axis=0, ignore_index=True)

cols_to_drop = []
for col in combined.columns:
  sample_val = (
      combined[col].dropna().iloc[0]
      if not combined[col].dropna().empty
      else None
  )

  if sample_val is not None and hasattr(sample_val, 'bounds'):
    combined[f'{col}_minx'] = combined[col].apply(
        lambda g: g.bounds[0] if hasattr(g, 'bounds') else np.nan
    )
    combined[f'{col}_miny'] = combined[col].apply(
        lambda g: g.bounds[1] if hasattr(g, 'bounds') else np.nan
    )
    combined[f'{col}_maxx'] = combined[col].apply(
        lambda g: g.bounds[2] if hasattr(g, 'bounds') else np.nan
    )
    combined[f'{col}_maxy'] = combined[col].apply(
        lambda g: g.bounds[3] if hasattr(g, 'bounds') else np.nan
    )
    combined[f'{col}_area'] = combined[col].apply(
        lambda g: g.area if hasattr(g, 'area') else np.nan
    )
    combined[f'{col}_length'] = combined[col].apply(
        lambda g: g.length if hasattr(g, 'length') else np.nan
    )
    cols_to_drop.append(col)
    continue

  if pd.api.types.is_datetime64_any_dtype(combined[col]):
    combined[f'{col}_year'] = combined[col].dt.year
    combined[f'{col}_month'] = combined[col].dt.month
    combined[f'{col}_day'] = combined[col].dt.day
    combined[f'{col}_dayofweek'] = combined[col].dt.dayofweek
    combined[f'{col}_timestamp'] = combined[col].astype('int64') // 10**9
    cols_to_drop.append(col)
    continue

  if pd.api.types.is_object_dtype(
      combined[col]
  ) or pd.api.types.is_categorical_dtype(combined[col]):
    try:
      parsed_dt = pd.to_datetime(combined[col], errors='coerce')
      if parsed_dt.notna().sum() > 0.5 * len(combined[col].dropna()):
        combined[f'{col}_year'] = parsed_dt.dt.year
        combined[f'{col}_month'] = parsed_dt.dt.month
        combined[f'{col}_day'] = parsed_dt.dt.day
        combined[f'{col}_dayofweek'] = parsed_dt.dt.dayofweek
        cols_to_drop.append(col)
        continue
    except Exception:
      pass

    s = combined[col].astype(str)
    cat_type = pd.CategoricalDtype(categories=s.unique())
    combined[col] = s.astype(cat_type).cat.codes
    combined[col] = combined[col].replace(-1, np.nan)

if cols_to_drop:
  combined.drop(columns=cols_to_drop, inplace=True)

for col in combined.columns:
  if not pd.api.types.is_numeric_dtype(combined[col]):
    combined[col] = pd.to_numeric(combined[col], errors='coerce')

X_tr = combined.iloc[:n_train].copy()
X_te = combined.iloc[n_train:].copy()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds_lgb = np.zeros(len(X_te))
test_preds_xgb = np.zeros(len(X_te))
test_preds_cb = np.zeros(len(X_te))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr, y)):
  X_train_fold, y_train_fold = X_tr.iloc[tr_idx], y[tr_idx]
  X_val_fold, y_val_fold = X_tr.iloc[val_idx], y[val_idx]

  model_lgb = lgb.LGBMRegressor(
      n_estimators=1200,
      learning_rate=0.03,
      num_leaves=63,
      max_depth=8,
      subsample=0.8,
      colsample_bytree=0.8,
      random_state=42 + fold,
      verbose=-1,
      n_jobs=-1,
  )
  model_lgb.fit(
      X_train_fold,
      y_train_fold,
      eval_set=[(X_val_fold, y_val_fold)],
      callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
  )
  test_preds_lgb += model_lgb.predict(X_te) / 5.0

  model_xgb = xgb.XGBRegressor(
      n_estimators=1200,
      learning_rate=0.03,
      max_depth=6,
      subsample=0.8,
      colsample_bytree=0.8,
      random_state=42 + fold,
      n_jobs=-1,
      early_stopping_rounds=50,
  )
  model_xgb.fit(
      X_train