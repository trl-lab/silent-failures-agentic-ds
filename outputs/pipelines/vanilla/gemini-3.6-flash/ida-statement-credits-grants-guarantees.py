import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

train_path = "/path/to/user/workspace/ds_failures/datasets/strable/ida-statement-credits-grants-guarantees/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/ida-statement-credits-grants-guarantees/data/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = "Original Principal Amount (US$)"
y = train[target_col].values.astype(np.float64)

X_train = train.drop(columns=[target_col], errors="ignore")
X_test = test.drop(columns=[target_col], errors="ignore")

y_min = y.min()

n_train = len(X_train)
combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)

for col in list(combined.columns):
  if (
      pd.api.types.is_datetime64_any_dtype(combined[col])
      or "date" in col.lower()
      or "period" in col.lower()
  ):
    try:
      dt_col = pd.to_datetime(combined[col], errors="coerce")
      if dt_col.notna().sum() > 0:
        combined[f"{col}_year"] = dt_col.dt.year
        combined[f"{col}_month"] = dt_col.dt.month
        combined[f"{col}_day"] = dt_col.dt.day
        combined[f"{col}_dayofweek"] = dt_col.dt.dayofweek
        combined.drop(columns=[col], inplace=True)
        continue
    except Exception:
      pass

  if combined[col].dtype == "object" or combined[col].dtype.name == "category":
    combined[col] = combined[col].astype(str)
    freq = combined[col].value_counts()
    combined[f"{col}_freq"] = combined[col].map(freq)
    combined[col] = combined[col].astype("category")

X_proc = combined.iloc[:n_train].copy()
X_test_proc = combined.iloc[n_train:].copy()

X_num = X_proc.copy()
X_test_num = X_test_proc.copy()

for c in X_num.columns:
  if X_num[c].dtype.name == "category":
    X_num[c] = X_num[c].cat.codes
    X_test_num[c] = X_test_num[c].cat.codes

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test_proc))
cat_preds = np.zeros(len(X_test_proc))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_proc)):
  X_tr, y_tr = X_proc.iloc[train_idx], y[train_idx]

  lgb = LGBMRegressor(
      n_estimators=1000,
      learning_rate=0.03,
      num_leaves=31,
      random_state=42 + fold,
      n_jobs=-1,
      verbose=-1,
  )
  lgb.fit(X_tr, y_tr)
  lgb_preds += lgb.predict(X_test_proc) / kf.n_splits

  X_tr_num = X_num.iloc[train_idx]
  cat = CatBoostRegressor(
      iterations=1000,
      learning_rate=0.05,
      depth=6,
      random_state=42 + fold,
      verbose=0,
      thread_count=-1,
  )
  cat.fit(X_tr_num, y_tr)
  cat_preds += cat.predict(X_test_num) / kf.n_splits

final_preds = 0.5 * lgb_preds + 0.5 * cat_preds

if y_min >= 0:
  final_preds = np.clip(final_preds, a_min=0, a_max=None)

sub = pd.DataFrame(
    {"id": np.arange(len(final_preds)), target_col: final_preds}
)
sub.to_csv("predictions.csv", index=False)