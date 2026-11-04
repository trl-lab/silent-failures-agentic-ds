import os
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb

os.makedirs("./working", exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

train = train.copy()
test = test.copy()
train["orig_idx"] = np.arange(len(train), dtype=int)
test["orig_idx"] = np.arange(len(test), dtype=int)

if "ISE" not in train.columns:
    train["ISE"] = np.nan
if "ISE" not in test.columns:
    test["ISE"] = np.nan

exclude = {
    "date",
    "ISE",
    "orig_idx",
    "id",
    "index",
    "target",
    "y",
    "dayofweek",
    "month",
    "dayofyear",
}
candidate_cols = [
    c
    for c in train.columns
    if c not in exclude and not str(c).upper().startswith("ISE")
]

exog_cols = []
for c in candidate_cols:
    converted = pd.to_numeric(train[c], errors="coerce")
    if converted.notna().sum() > 0:
        exog_cols.append(c)


def parse_dates(s):
    s = pd.Series(s)
    if pd.api.types.is_numeric_dtype(s):
        sample = s.dropna()
        if len(sample) > 0 and sample.between(10000000, 99999999).all():
            parsed = pd.to_datetime(
                sample.astype(int).astype(str), format="%Y%m%d", errors="coerce"
            )
            return parsed.reindex(s.index)
    return pd.to_datetime(s, errors="coerce")


def make_feature_cols(cols):
    feature_cols = (
        cols
        + [f"{c}__lag1" for c in cols]
        + [f"{c}__roll5" for c in cols]
        + ["dayofweek", "month", "dayofyear"]
    )
    return list(dict.fromkeys(feature_cols))


def prepare_date_numeric(df, cols):
    df = df.copy()
    if len(df) == 0:
        return df

    if "date" not in df.columns:
        df["date"] = pd.Timestamp("2000-01-01")

    df["date"] = parse_dates(df["date"])
    if df["date"].isna().any():
        df["date"] = df["date"].ffill().bfill()
        if df["date"].isna().any():
            df["date"] = pd.Timestamp("2000-01-01")

    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df


def build_features(df, cols, history=None):
    feature_cols = make_feature_cols(cols)
    df = df.copy()

    if len(df) == 0:
        for c in feature_cols:
            df[c] = 0.0
        return df

    df = prepare_date_numeric(df, cols)

    if history is None or len(history) == 0:
        work = df.sort_values("date", kind="mergesort").reset_index(drop=True)
        base = work[cols].copy()

        for c in cols:
            base[f"{c}__lag1"] = base[c].shift(1)
            base[f"{c}__roll5"] = base[c].shift(1).rolling(5, min_periods=1).mean()

        out = work.copy()
        for c in base.columns:
            out[c] = base[c].to_numpy()

    else:
        hist = prepare_date_numeric(history, cols)
        hist = hist.sort_values("date", kind="mergesort").reset_index(drop=True)

        target = df.sort_values("date", kind="mergesort").reset_index(drop=True)
        target["target_order"] = np.arange(len(target), dtype=int)

        hist_part = hist[cols].copy()
        hist_part["date"] = hist["date"].to_numpy()
        hist_part["is_target"] = 0

        target_part = target[cols].copy()
        target_part["date"] = target["date"].to_numpy()
        target_part["is_target"] = 1
        target_part["target_order"] = target["target_order"].to_numpy()

        base = pd.concat([hist_part, target_part], ignore_index=True)
        base = base.sort_values("date", kind="mergesort").reset_index(drop=True)

        for c in cols:
            base[f"{c}__lag1"] = base[c].shift(1)
            base[f"{c}__roll5"] = base[c].shift(1).rolling(5, min_periods=1).mean()

        feat = base[base["is_target"] == 1].copy()
        feat = feat.sort_values("target_order").reset_index(drop=True)

        out = target.copy()
        lag_cols = [f"{c}__lag1" for c in cols] + [f"{c}__roll5" for c in cols]

        if len(feat) != len(out):
            if len(feat) < len(out):
                pad = pd.DataFrame(
                    0.0, index=range(len(out) - len(feat)), columns=lag_cols
                )
                feat = pd.concat([pad, feat], ignore_index=True)
            else:
                feat = feat.iloc[-len(out) :].reset_index(drop=True)

        for c in lag_cols:
            out[c] = feat[c].to_numpy()

    out["dayofweek"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["dayofyear"] = out["date"].dt.dayofyear

    for c in feature_cols:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out


train_feat = build_features(train, exog_cols, history=None)
test_feat = build_features(test, exog_cols, history=train)

feature_cols = make_feature_cols(exog_cols)

train_feat["ISE"] = pd.to_numeric(train_feat["ISE"], errors="coerce")
train_feat["ISE"] = train_feat["ISE"].replace([np.inf, -np.inf], np.nan)
train_feat = train_feat.dropna(subset=["ISE"]).reset_index(drop=True)

test_feat = test_feat.sort_values("orig_idx").reset_index(drop=True)
for c in feature_cols:
    if c not in test_feat.columns:
        test_feat[c] = 0.0

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 7,
    "max_depth": 3,
    "min_data_in_leaf": 20,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "feature_fraction": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "max_bin": 63,
    "seed": 42,
    "num_threads": -1,
    "verbose": -1,
}

NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50

if len(train_feat) == 0:
    val_rmse = float("nan")
    pred_test = np.zeros(len(test_feat), dtype=float)
else:
    X_train = train_feat[feature_cols].to_numpy(dtype=float)
    y_train = train_feat["ISE"].to_numpy(dtype=float)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)

    if len(X_train) >= 10:
        n_splits = 5
        if len(X_train) >= 300:
            test_size = min(50, max(1, (len(X_train) - 1) // (n_splits + 1)))
        else:
            test_size = max(1, len(X_train) // (n_splits + 1))

        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

        scores = []
        best_iters = []

        for tr_idx, va_idx in tscv.split(X_train):
            if len(tr_idx) == 0 or len(va_idx) == 0:
                continue

            dtrain = lgb.Dataset(X_train[tr_idx], label=y_train[tr_idx])
            dval = lgb.Dataset(X_train[va_idx], label=y_train[va_idx], reference=dtrain)

            booster = lgb.train(
                params,
                dtrain,
                num_boost_round=NUM_BOOST_ROUND,
                valid_sets=[dval],
                valid_names=["val"],
                callbacks=[
                    lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )

            best_iter = getattr(booster, "best_iteration", 0)
            if best_iter is None or best_iter <= 0:
                best_iter = NUM_BOOST_ROUND

            pred = booster.predict(X_train[va_idx], num_iteration=int(best_iter))
            scores.append(root_mean_squared_error(y_train[va_idx], pred))
            best_iters.append(int(best_iter))

        if scores:
            val_rmse = float(np.mean(scores))
            final_iters = int(np.median(best_iters))
        else:
            val_rmse = float(np.std(y_train)) if len(y_train) > 0 else float("nan")
            final_iters = 100
    else:
        val_rmse = float(np.std(y_train)) if len(y_train) > 0 else float("nan")
        final_iters = 100

    final_iters = int(np.clip(final_iters, 1, NUM_BOOST_ROUND))

    dtrain_all = lgb.Dataset(X_train, label=y_train)
    final_booster = lgb.train(
        params,
        dtrain_all,
        num_boost_round=final_iters,
        callbacks=[lgb.log_evaluation(-1)],
    )

    if len(test_feat) == 0:
        pred_test = np.zeros(0, dtype=float)
    else:
        X_test = test_feat[feature_cols].to_numpy(dtype=float)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        pred_test = final_booster.predict(X_test, num_iteration=final_iters)

pred_test = np.nan_to_num(pred_test, nan=0.0, posinf=0.0, neginf=0.0).astype(float)

submission = pd.DataFrame(
    {"id": np.arange(len(test_feat), dtype=int), "ISE": pred_test}
)
submission.to_csv("./working/submission.csv", index=False)

print(f"Validation RMSE (5-fold TimeSeriesSplit): {val_rmse:.6f}")
