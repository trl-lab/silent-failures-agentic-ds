import gc
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

train_path = (
    "/path/to/user/workspace/ds_failures/datasets/kaggle/ashrae-energy-prediction/data/train.parquet"
)
test_path = (
    "/path/to/user/workspace/ds_failures/datasets/kaggle/ashrae-energy-prediction/data/test.parquet"
)

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

drop_cols = ["id", "row_id"]
for col in drop_cols:
  if col in train.columns:
    train.drop(columns=[col], inplace=True)
  if col in test.columns:
    test.drop(columns=[col], inplace=True)

y = np.log1p(train["meter_reading"].clip(lower=0).values)
train.drop(columns=["meter_reading"], inplace=True)


def extract_time_features(df):
  if "timestamp" in df.columns:
    dt = pd.to_datetime(df["timestamp"])
    df["hour"] = dt.dt.hour.astype(np.int8)
    df["dayofweek"] = dt.dt.dayofweek.astype(np.int8)
    df["month"] = dt.dt.month.astype(np.int8)
    df["day"] = dt.dt.day.astype(np.int8)
    df.drop(columns=["timestamp"], inplace=True)


extract_time_features(train)
extract_time_features(test)

cat_candidate_cols = ["building_id", "meter", "site_id", "primary_use"]
cat_cols = [c for c in cat_candidate_cols if c in train.columns]

for col in train.columns:
  if train[col].dtype == "object" or train[col].dtype.name == "category":
    le = LabelEncoder()
    all_vals = pd.concat(
        [train[col].astype(str), test[col].astype(str)]
    ).unique()
    le.fit(all_vals)
    train[col] = le.transform(train[col].astype(str)).astype(np.int16)
    test[col] = le.transform(test[col].astype(str)).astype(np.int16)
    if col not in cat_cols:
      cat_cols.append(col)

for c in cat_cols:
  if c in train.columns:
    train[c] = train[c].astype("category")
    test[c] = test[c].astype("category")

for col in train.columns:
  if col not in cat_cols:
    if train[col].dtype == np.float64:
      train[col] = train[col].astype(np.float32)
      test[col] = test[col].astype(np.float32)
    elif train[col].dtype == np.int64:
      train[col] = train[col].astype(np.int32)
      test[col] = test[col].astype(np.int32)

gc.collect()

kf = KFold(n_splits=3, shuffle=True, random_state=42)
test_preds = np.zeros(len(test), dtype=np.float32)

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.1,
    "num_leaves": 127,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "n_estimators": 600,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (trn_idx, val_idx) in enumerate(kf.split(train, y)):
  X_tr, y_tr = train.iloc[trn_idx], y[trn_idx]
  X_va, y_va = train.iloc[val_idx], y[val_idx]

  model = LGBMRegressor(**lgb_params)
  model.fit(
      X_tr,
      y_tr,
      eval_set=[(X_va, y_va)],
      callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
  )

  chunk_size = 5000000
  for i in range(0, len(test), chunk_size):
    test_preds[i : i + chunk_size] += (
        model.predict(test.iloc[i : i + chunk_size]) / kf.n_splits
    )

  del X_tr, y_tr, X_va, y_va, model
  gc.collect()

final_preds = np.expm1(test_preds)
final_preds = np.clip(final_preds, 0, None)

sub = pd.DataFrame(
    {"id": np.arange(len(final_preds), dtype=np.int32), "meter_reading": final_preds}
)

sub.to_csv("predictions.csv", index=False)