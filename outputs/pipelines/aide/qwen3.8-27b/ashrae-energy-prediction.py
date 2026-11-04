import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb

os.makedirs("./working", exist_ok=True)


def rmsle_from_log(y_log, pred_log):
    y_log = np.asarray(y_log, dtype=np.float64)
    pred_log = np.asarray(pred_log, dtype=np.float64)
    pred_log = np.clip(pred_log, 0.0, 20.0)
    return float(np.sqrt(np.mean((y_log - pred_log) ** 2)))


def make_features(df):
    ts = df["timestamp"]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        ts = pd.to_datetime(ts, errors="coerce")
    if ts.isna().any():
        ts = ts.fillna(pd.Timestamp("2016-01-01 00:00:00"))

    out = pd.DataFrame(index=df.index)
    out["building_id"] = df["building_id"].astype(np.int32)
    out["meter"] = df["meter"].astype(np.int8)
    out["hour"] = ts.dt.hour.astype(np.int8)
    out["dayofweek"] = ts.dt.dayofweek.astype(np.int8)
    out["month"] = ts.dt.month.astype(np.int8)
    out["day"] = ts.dt.day.astype(np.int8)
    out["dayofyear"] = ts.dt.dayofyear.astype(np.int16)
    out["year"] = ts.dt.year.astype(np.int16)
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(np.int8)
    return out


train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

train["timestamp"] = pd.to_datetime(train["timestamp"], errors="coerce")
test["timestamp"] = pd.to_datetime(test["timestamp"], errors="coerce")

if train["timestamp"].isna().any():
    train["timestamp"] = train["timestamp"].fillna(pd.Timestamp("2016-01-01 00:00:00"))
if test["timestamp"].isna().any():
    test["timestamp"] = test["timestamp"].fillna(pd.Timestamp("2016-01-01 00:00:00"))

for df in (train, test):
    if df["building_id"].isna().any():
        df["building_id"] = df["building_id"].fillna(0)
    if df["meter"].isna().any():
        df["meter"] = df["meter"].fillna(0)

train = train[np.isfinite(train["meter_reading"])].reset_index(drop=True)

# Representative timestamp subsample for fast time-based CV.
cv_frac = 0.2
if len(train) > 5_000_000:
    unique_train_ts = train["timestamp"].unique()
    sample_ts = pd.Series(unique_train_ts).sample(frac=cv_frac, random_state=42).values
    cv = train[train["timestamp"].isin(sample_ts)].copy()
else:
    cv = train.copy()

cv = cv.sort_values("timestamp").reset_index(drop=True)

# Split selected timestamps into 6 consecutive blocks; use blocks 1..5 as validation folds.
n_chunks = 6
ts_values = cv["timestamp"].values
unique_cv_ts, inverse = np.unique(ts_values, return_inverse=True)
n_ts = len(unique_cv_ts)

if n_ts >= n_chunks:
    ts_chunk = max(1, n_ts // n_chunks)
    ts_bounds = [0]
    for i in range(1, n_chunks):
        ts_bounds.append(i * ts_chunk)
    ts_bounds.append(n_ts)

    chunk_ids = np.empty(n_ts, dtype=np.int8)
    for c in range(n_chunks):
        chunk_ids[ts_bounds[c] : ts_bounds[c + 1]] = c
    row_chunk = chunk_ids[inverse]
else:
    n_cv = len(cv)
    chunk = max(1, n_cv // n_chunks)
    row_chunk = np.repeat(np.arange(n_chunks, dtype=np.int8), chunk)
    row_chunk = row_chunk[:n_cv]

X_cv = make_features(cv)
y_cv = np.log1p(cv["meter_reading"].values).astype(np.float32)
del cv
gc.collect()

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 1000,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 42,
    "num_threads": -1,
}

cat_features = ["building_id", "meter", "hour", "dayofweek", "month"]
num_boost_round = 600
metrics = []
best_iters = []

for k in range(1, n_chunks):
    val_mask = row_chunk == k
    train_mask = row_chunk < k

    if not val_mask.any() or not train_mask.any():
        continue

    X_tr = X_cv[train_mask]
    X_val = X_cv[val_mask]
    y_tr = y_cv[train_mask]
    y_val = y_cv[val_mask]

    dtrain = lgb.Dataset(
        X_tr, label=y_tr, categorical_feature=cat_features, free_raw_data=True
    )
    dval = lgb.Dataset(
        X_val,
        label=y_val,
        reference=dtrain,
        categorical_feature=cat_features,
        free_raw_data=True,
    )

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    best_iter = (
        booster.best_iteration
        if booster.best_iteration and booster.best_iteration > 0
        else num_boost_round
    )
    pred_log = booster.predict(X_val, num_iteration=best_iter)

    metrics.append(rmsle_from_log(y_val, pred_log))
    best_iters.append(best_iter)

    del booster, dtrain, dval, X_tr, X_val, y_tr, y_val
    gc.collect()

if metrics:
    print(f"Validation RMSLE (5 time-based folds): {np.mean(metrics):.6f}")
else:
    print("Validation RMSLE: no valid folds")

del X_cv, y_cv, row_chunk
gc.collect()

# Final model on the full training set.
X_train = make_features(train)
y_train = np.log1p(train["meter_reading"].values).astype(np.float32)
del train
gc.collect()

if best_iters:
    final_rounds = int(np.clip(np.mean(best_iters) * 1.1 + 50, 200, 600))
else:
    final_rounds = 600

dtrain_all = lgb.Dataset(
    X_train, label=y_train, categorical_feature=cat_features, free_raw_data=True
)
booster = lgb.train(params, dtrain_all, num_boost_round=final_rounds)
del dtrain_all, X_train, y_train
gc.collect()

X_test = make_features(test)
del test
gc.collect()

pred_log = np.clip(np.asarray(booster.predict(X_test), dtype=np.float64), 0.0, 20.0)
pred = np.expm1(pred_log)
pred = np.clip(pred, 0.0, None)
pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

submission = pd.DataFrame(
    {
        "id": np.arange(len(pred), dtype=np.int64),
        "meter_reading": pred.astype(np.float64),
    }
)
submission.to_csv("./working/submission.csv", index=False)
