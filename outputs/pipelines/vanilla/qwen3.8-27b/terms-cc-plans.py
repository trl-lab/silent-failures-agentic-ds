import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/terms-cc-plans/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/terms-cc-plans/data/test.parquet"
TARGET = "APR"


def preprocess(df):
    df = df.copy()

    if TARGET in df.columns:
        df = df.drop(columns=[TARGET])

    if "Agreement Date" in df.columns:
        dt = pd.to_datetime(df["Agreement Date"], errors="coerce")
        df["date_year"] = dt.dt.year.astype("float32")
        df["date_month"] = dt.dt.month.astype("float32")
        df["date_year_month"] = (dt.dt.year * 12 + dt.dt.month).astype("float32")
        df["date_days"] = (dt - pd.Timestamp("1900-01-01")).dt.days.astype("float32")
        df["date_missing"] = dt.isna().astype("float32")
        df = df.drop(columns=["Agreement Date"])

    for col in ["Grace Period", "Annual Fee", "Late Fee"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in df.columns:
        if str(df[col].dtype) in ("object", "category", "string"):
            df[col] = df[col].fillna("missing").astype(str).str.strip()
            df[col] = df[col].replace({"nan": "missing", "None": "missing", "": "missing"})

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    constant_cols = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)

    return df


def save_predictions(pred, n_rows, fallback=None):
    pred = np.asarray(pred, dtype=float).reshape(-1)

    if pred.size != n_rows:
        if pred.size == 0:
            pred = np.zeros(n_rows, dtype=float)
        else:
            pred = np.resize(pred, n_rows)

    if fallback is None:
        finite = pred[np.isfinite(pred)]
        fallback = float(finite.mean()) if finite.size else 0.0
    else:
        fallback = float(fallback)

    if not np.isfinite(fallback):
        fallback = 0.0

    pred = np.nan_to_num(pred, nan=fallback, posinf=fallback, neginf=fallback)
    pred = np.clip(pred, 0.0, None)

    out = pd.DataFrame({
        "id": np.arange(n_rows, dtype=int),
        "APR": pred
    })
    out.to_csv("predictions.csv", index=False)


def get_best_iteration(model, default_iterations):
    best_iter = None

    try:
        best_iter = model.get_best_iteration()
    except Exception:
        best_iter = getattr(model, "best_iteration_", None)

    if best_iter is None:
        return int(default_iterations)

    best_iter = int(best_iter)

    if best_iter < 0:
        return int(default_iterations)

    if best_iter < 100:
        best_iter = 100

    return min(best_iter, int(default_iterations))


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if len(test_df) == 0:
        save_predictions(np.array([], dtype=float), 0)
        return

    if TARGET not in train_df.columns:
        save_predictions(np.zeros(len(test_df), dtype=float), len(test_df), 0.0)
        return

    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    train_df = train_df.dropna(subset=[TARGET])

    if train_df.empty:
        save_predictions(np.zeros(len(test_df), dtype=float), len(test_df), 0.0)
        return

    y_train = train_df[TARGET].astype(float)
    X_train = preprocess(train_df)
    X_test = preprocess(test_df)

    common_cols = [col for col in X_train.columns if col in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    y_mean = float(np.nanmean(y_train)) if len(y_train) else 0.0
    if not np.isfinite(y_mean):
        y_mean = 0.0

    if X_train.shape[1] == 0 or X_test.shape[1] == 0 or len(X_train) < 2:
        save_predictions(np.full(len(test_df), y_mean, dtype=float), len(test_df), y_mean)
        return

    if len(X_train) > 20:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
    else:
        X_tr, X_val, y_tr, y_val = X_train, X_train.iloc[0:0], y_train, y_train.iloc[0:0]

    cat_cols = [
        col for col in X_tr.columns
        if str(X_tr[col].dtype) in ("object", "category", "string")
    ]

    params = {
        "iterations": 2000,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": 0,
        "cat_features": cat_cols,
    }

    model = CatBoostRegressor(**params)

    if len(X_val) > 0:
        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=100,
            verbose=0,
        )
        best_iter = get_best_iteration(model, params["iterations"])
    else:
        model.fit(X_tr, y_tr, verbose=0)
        best_iter = int(params["iterations"])

    final_params = params.copy()
    final_params["iterations"] = int(best_iter)

    final_model = CatBoostRegressor(**final_params)
    final_model.fit(X_train, y_train, verbose=0)

    pred = final_model.predict(X_test)
    save_predictions(pred, len(test_df), y_mean)


if __name__ == "__main__":
    main()