import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


def to_datetime_safe(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series.astype(str), format="%Y%m%d", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "dteday" in df.columns:
        dt = to_datetime_safe(df["dteday"])
        df["day"] = dt.dt.day
        df["dteday_month"] = dt.dt.month
        df["dteday_year"] = dt.dt.year

    df = df.replace([np.inf, -np.inf], np.nan)

    drop_cols = [
        c
        for c in ["cnt", "casual", "registered", "instant", "dteday"]
        if c in df.columns
    ]
    df = df.drop(columns=drop_cols)

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("category").cat.codes

    df = df.fillna(0)

    cyclical = [
        ("hr", 24),
        ("mnth", 12),
        ("weekday", 7),
        ("season", 4),
        ("day", 31),
        ("dteday_month", 12),
    ]

    for col, period in cyclical:
        if col in df.columns:
            df[f"{col}_sin"] = np.sin(2 * np.pi * df[col].astype(float) / period)
            df[f"{col}_cos"] = np.cos(2 * np.pi * df[col].astype(float) / period)

    if "temp" in df.columns and "hum" in df.columns:
        df["temp_hum"] = df["temp"] * df["hum"]

    if "temp" in df.columns and "windspeed" in df.columns:
        df["temp_wind"] = df["temp"] * df["windspeed"]

    if "atemp" in df.columns and "windspeed" in df.columns:
        df["atemp_wind"] = df["atemp"] * df["windspeed"]

    if "hr" in df.columns:
        df["is_rush_hour"] = (
            ((df["hr"] >= 7) & (df["hr"] <= 9)) | ((df["hr"] >= 16) & (df["hr"] <= 18))
        ).astype(int)

    return df


def main() -> None:
    os.makedirs("./working", exist_ok=True)

    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    train = train.dropna(subset=["cnt"]).reset_index(drop=True)

    if "dteday" in train.columns:
        train["_dt"] = to_datetime_safe(train["dteday"])
        sort_cols = ["_dt"]
        if "instant" in train.columns:
            sort_cols.append("instant")
        train = train.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        train = train.drop(columns=["_dt"])
    elif "instant" in train.columns:
        train = train.sort_values("instant", kind="mergesort").reset_index(drop=True)

    X_train = preprocess(train)
    y_train = train["cnt"].astype(float).values
    X_test = preprocess(test)

    feature_cols = list(X_train.columns)
    X_train = X_train[feature_cols].astype(float)
    X_test = X_test.reindex(columns=feature_cols, fill_value=0).astype(float)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "seed": 42,
        "verbosity": -1,
    }

    tscv = TimeSeriesSplit(n_splits=5)
    fold_rmses = []
    best_iters = []

    for tr_idx, va_idx in tscv.split(X_train):
        dtrain = lgb.Dataset(X_train.iloc[tr_idx], label=y_train[tr_idx])
        dvalid = lgb.Dataset(
            X_train.iloc[va_idx],
            label=y_train[va_idx],
            reference=dtrain,
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=3000,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
        )

        best_iter = model.best_iteration if model.best_iteration else None
        pred = model.predict(X_train.iloc[va_idx], num_iteration=best_iter)
        rmse = mean_squared_error(y_train[va_idx], pred, squared=False)

        fold_rmses.append(rmse)
        best_iters.append(model.best_iteration if model.best_iteration else 3000)

    mean_rmse = float(np.mean(fold_rmses))
    print(f"5-fold time-series CV RMSE: {mean_rmse:.4f}")

    final_rounds = int(np.median(best_iters)) if best_iters else 500
    final_rounds = max(final_rounds, 100)

    dtrain_full = lgb.Dataset(X_train, label=y_train)
    final_model = lgb.train(
        params,
        dtrain_full,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    pred_test = final_model.predict(X_test)
    pred_test = np.clip(pred_test, 0, None)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test), dtype=int),
            "cnt": pred_test,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)

    print(f"Saved submission to ./working/submission.csv with {len(submission)} rows.")


if __name__ == "__main__":
    main()
