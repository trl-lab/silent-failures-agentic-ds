import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")


def add_time_features(df):
    df = df.copy()
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
    else:
        dt = pd.Series(pd.NaT, index=df.index)

    df["hour"] = dt.dt.hour.fillna(0).astype(float)
    df["minute"] = dt.dt.minute.fillna(0).astype(float)
    df["dayofweek"] = dt.dt.dayofweek.fillna(0).astype(float)
    df["month"] = dt.dt.month.fillna(0).astype(float)
    df["day"] = dt.dt.day.fillna(0).astype(float)
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(float)
    df["date_dt"] = dt
    return df


def add_temporal_features(df):
    df = df.copy()
    for col in ("CO2", "Light"):
        if col not in df.columns:
            continue

        s = df[col].astype(float)
        lag = s.shift(1)

        df[f"{col}_lag1"] = lag

        diff = s - lag
        if len(diff) > 0 and pd.isna(diff.iloc[0]):
            diff.iloc[0] = 0.0
        df[f"{col}_diff1"] = diff

        for w in (5, 15):
            roll = lag.rolling(window=w, min_periods=1).mean()
            df[f"{col}_roll{w}"] = roll.fillna(s)

    return df


def safe_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_prob)


def make_submission(pred_test):
    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame(
        {
            "id": np.arange(len(pred_test), dtype=int),
            "Occupancy": np.asarray(pred_test, dtype=float),
        }
    )
    submission.to_csv("./working/submission.csv", index=False)


def train_fold(X_tr, y_tr, X_va, y_va, features, params, num_boost_round):
    fold_params = params.copy()
    fold_params["metric"] = "auc" if y_va.nunique() > 1 else "binary_logloss"

    train_ds = lgb.Dataset(
        X_tr,
        label=y_tr,
        feature_name=features,
        free_raw_data=False,
    )
    valid_ds = lgb.Dataset(
        X_va,
        label=y_va,
        reference=train_ds,
        free_raw_data=False,
    )

    model = lgb.train(
        fold_params,
        train_ds,
        num_boost_round=num_boost_round,
        valid_sets=[valid_ds],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    best_iter = model.best_iteration if model.best_iteration > 0 else num_boost_round
    pred_va = model.predict(X_va, num_iteration=best_iter)
    score = safe_auc(y_va.values, pred_va)

    return score, best_iter


train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

train = add_time_features(train)
test = add_time_features(test)

train["Occupancy"] = pd.to_numeric(train["Occupancy"], errors="coerce")
train = train.dropna(subset=["Occupancy"]).reset_index(drop=True)
train = train.sort_values("date_dt", kind="mergesort", na_position="last").reset_index(
    drop=True
)

test = test.copy()
test["original_idx"] = np.arange(len(test), dtype=int)
test = test.sort_values("date_dt", kind="mergesort", na_position="last").reset_index(
    drop=True
)

numeric_features = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]

for col in numeric_features:
    if col not in train.columns:
        train[col] = np.nan
    if col not in test.columns:
        test[col] = np.nan

    train[col] = pd.to_numeric(train[col], errors="coerce")
    test[col] = pd.to_numeric(test[col], errors="coerce")

    train[col] = train[col].replace([np.inf, -np.inf], np.nan)
    test[col] = test[col].replace([np.inf, -np.inf], np.nan)

train = add_temporal_features(train)
test = add_temporal_features(test)

test = test.sort_values("original_idx").reset_index(drop=True)

time_features = [
    "hour",
    "minute",
    "dayofweek",
    "month",
    "day",
    "is_weekend",
]

temporal_features = [
    f"{col}_{suffix}"
    for col in ("CO2", "Light")
    for suffix in ("lag1", "diff1", "roll5", "roll15")
]
temporal_features = [
    c for c in temporal_features if c in train.columns and c in test.columns
]

features = numeric_features + time_features + temporal_features

for col in features:
    if col not in train.columns:
        train[col] = np.nan
    if col not in test.columns:
        test[col] = np.nan

X = train[features].astype(float)
y = train["Occupancy"].astype(int)
X_test = test[features].astype(float)

os.makedirs("./working", exist_ok=True)

if len(train) == 0 or y.nunique() < 2:
    pred_test = np.full(len(test), 0.5, dtype=float)
    make_submission(pred_test)
    print("CV AUC: 0.500000")
else:
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "seed": 42,
        "feature_fraction_seed": 42,
        "bagging_seed": 42,
        "data_random_seed": 42,
        "verbose": -1,
        "num_threads": 4,
    }

    scores = []
    best_iters = []
    num_boost_round = 2000
    n_splits = 5

    if len(train) >= n_splits + 1:
        tscv = TimeSeriesSplit(n_splits=n_splits)

        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X)):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            score, best_iter = train_fold(
                X_tr,
                y_tr,
                X_va,
                y_va,
                features,
                params,
                num_boost_round,
            )

            scores.append(score)
            best_iters.append(best_iter)

            print(f"Fold {fold + 1} AUC: {score:.6f}")
    else:
        split = max(1, min(len(train) - 1, int(0.8 * len(train))))
        X_tr, X_va = X.iloc[:split], X.iloc[split:]
        y_tr, y_va = y.iloc[:split], y.iloc[split:]

        score, best_iter = train_fold(
            X_tr,
            y_tr,
            X_va,
            y_va,
            features,
            params,
            num_boost_round,
        )

        scores.append(score)
        best_iters.append(best_iter)

        print(f"Holdout AUC: {score:.6f}")

    final_rounds = max(1, int(np.mean(best_iters)))
    final_ds = lgb.Dataset(X, label=y, feature_name=features, free_raw_data=False)
    final_model = lgb.train(params, final_ds, num_boost_round=final_rounds)

    if len(test) == 0:
        pred_test = np.array([], dtype=float)
    else:
        pred_test = final_model.predict(X_test)

    pred_test = np.clip(np.asarray(pred_test, dtype=float), 0.0, 1.0)
    make_submission(pred_test)
    print(f"CV AUC: {np.mean(scores):.6f}")
