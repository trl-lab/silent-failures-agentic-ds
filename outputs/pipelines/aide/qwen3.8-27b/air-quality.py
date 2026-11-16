import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

INPUT_DIR = Path("./input")
WORK_DIR = Path("./working")
WORK_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "CO(GT)"
SEED = 42


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    # Time features
    df["hour"] = np.nan
    df["minute"] = np.nan
    df["second"] = np.nan
    if "Time" in df.columns:
        t = df["Time"].astype(str).str.strip()
        parts = t.str.split(":", n=2)
        df["hour"] = parts.str[0].astype(float)
        df["minute"] = parts.str[1].astype(float)
        df["second"] = parts.str[2].astype(float)

    # Date features
    date = pd.Series(pd.NaT, index=df.index)
    if "Date" in df.columns:
        date = pd.to_datetime(df["Date"], errors="coerce")

    df["year"] = date.dt.year
    df["month"] = date.dt.month
    df["day"] = date.dt.day
    df["dayofweek"] = date.dt.dayofweek

    df["is_weekend"] = np.where(date.dt.dayofweek >= 5, 1.0, 0.0)
    df.loc[date.isna(), "is_weekend"] = np.nan

    # Cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    # Datetime column for chronological ordering
    dt = pd.Series(pd.NaT, index=df.index)
    if "Date" in df.columns and "Time" in df.columns:
        dt = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce"
        )
        if dt.isna().sum() > date.isna().sum():
            dt = date
    elif "Date" in df.columns:
        dt = date
    elif "Time" in df.columns:
        dt = pd.to_datetime(df["Time"].astype(str), errors="coerce")

    df["datetime"] = dt

    # Drop original raw date/time columns
    df = df.drop(columns=[c for c in ["Date", "Time"] if c in df.columns])
    return df


def main():
    train = pd.read_parquet(INPUT_DIR / "train.parquet")
    test = pd.read_parquet(INPUT_DIR / "test.parquet")

    train = preprocess(train)
    test = preprocess(test)

    # Ensure target is numeric and remove rows with missing target for training
    train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
    valid_target_mask = train[TARGET].notna()
    train = train.loc[valid_target_mask].copy()

    # Chronological ordering for time-series validation
    if train["datetime"].notna().any():
        train = train.sort_values("datetime", kind="mergesort").reset_index(drop=True)

    y = train[TARGET].astype(float)

    # Use only numeric feature columns present in train
    feature_cols = [
        col
        for col in train.columns
        if col not in [TARGET, "datetime"] and pd.api.types.is_numeric_dtype(train[col])
    ]

    # Drop features that are entirely missing in the training data
    feature_cols = [col for col in feature_cols if train[col].notna().any()]

    X_train = train[feature_cols].astype(float)
    X_test = test.reindex(columns=feature_cols).astype(float)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "seed": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }

    # Time-aware 5-fold cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    fold_scores = []
    best_iterations = []

    for train_idx, val_idx in tscv.split(X_train):
        X_tr = X_train.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_va = X_train.iloc[val_idx]
        y_va = y.iloc[val_idx]

        if len(X_tr) < 50 or len(X_va) < 10:
            continue

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        preds = model.predict(X_va, num_iteration=model.best_iteration)
        preds = np.clip(preds, 0.0, None)
        fold_scores.append(np.sqrt(mean_squared_error(y_va, preds)))
        best_iterations.append(model.best_iteration)

    if fold_scores:
        cv_rmse = float(np.mean(fold_scores))
        print(f"CV RMSE: {cv_rmse:.6f}")
    else:
        cv_rmse = np.nan
        print("CV RMSE: nan")

    # Final model trained on all available training data
    final_rounds = int(np.median(best_iterations)) if best_iterations else 1000
    final_rounds = max(1, final_rounds)

    dtrain_full = lgb.Dataset(X_train, label=y)
    final_model = lgb.train(
        params,
        dtrain_full,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_preds = final_model.predict(X_test)
    test_preds = np.clip(test_preds, 0.0, None)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test)),
            TARGET: test_preds,
        }
    )
    submission.to_csv(WORK_DIR / "submission.csv", index=False)


if __name__ == "__main__":
    main()
