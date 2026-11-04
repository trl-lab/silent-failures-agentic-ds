import os
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from xgboost.callback import EarlyStopping

os.makedirs("./working", exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")


def add_features(df):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["year"] = df["datetime"].dt.year
    df["dayofyear"] = df["datetime"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


train = add_features(train)
test = add_features(test)

feature_cols = [
    "season",
    "hour",
    "day",
    "month",
    "dayofweek",
    "weekend",
    "workingday",
    "holiday",
    "weather",
    "temp",
    "atemp",
    "humidity",
    "windspeed",
    "year",
    "dayofyear",
    "hour_sin",
    "hour_cos",
]

train = train.sort_values("datetime").reset_index(drop=True)
X_train = train[feature_cols].astype(np.float32)
y_log = np.log1p(train["count"].astype(np.float32))
y_count = train["count"].astype(np.float32)
X_test = test[feature_cols].astype(np.float32)

params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "seed": 42,
    "verbosity": 0,
}

tscv = TimeSeriesSplit(n_splits=5)
scores = []
best_rounds = []

for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
    dtrain = xgb.DMatrix(X_train.iloc[tr_idx], label=y_log[tr_idx])
    dval = xgb.DMatrix(X_train.iloc[val_idx], label=y_log[val_idx])
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dval, "val")],
        callbacks=[EarlyStopping(rounds=50, save_best=True)],
    )
    best_iter = model.best_iteration
    if best_iter is None:
        best_iter = 2000
    best_rounds.append(best_iter + 1)

    pred_log = model.predict(dval, iteration_range=(0, best_iter + 1))
    pred = np.expm1(pred_log)
    pred = np.clip(pred, 0.0, None)
    actual_log = np.log1p(y_count[val_idx])
    rmsle = np.sqrt(np.mean((actual_log - np.log1p(pred)) ** 2))
    scores.append(rmsle)
    print(f"Fold {fold} RMSLE: {rmsle:.6f}")

mean_rmsle = float(np.mean(scores))
print(f"Validation RMSLE: {mean_rmsle:.6f}")

final_rounds = int(np.mean(best_rounds)) + 10
dtrain_full = xgb.DMatrix(X_train, label=y_log)
final_model = xgb.train(params, dtrain_full, num_boost_round=final_rounds)
dtest = xgb.DMatrix(X_test)

final_pred_log = final_model.predict(dtest)
final_pred = np.expm1(final_pred_log)
final_pred = np.clip(final_pred, 0.0, None)

submission = pd.DataFrame({"id": np.arange(len(test)), "count": final_pred})
submission.to_csv("./working/submission.csv", index=False)
