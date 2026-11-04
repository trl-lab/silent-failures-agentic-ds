import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/financial-intermediary-funds-cash-transfers/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/financial-intermediary-funds-cash-transfers/data/test.parquet"
TARGET = "Amount in USD"


def _clean_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        out = series.astype(float)
    else:
        try:
            out = pd.to_numeric(series, errors="coerce")
        except Exception:
            out = pd.Series(np.nan, index=series.index, dtype=float)
    return out.replace([np.inf, -np.inf], np.nan)


def _write_predictions(values, n_rows: int) -> None:
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values), values, 0.0)
    pd.DataFrame({
        "id": np.arange(n_rows, dtype=int),
        TARGET: values
    }).to_csv("predictions.csv", index=False)


def main() -> None:
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    train_df[TARGET] = _clean_numeric(train_df[TARGET])
    train_df = train_df.dropna(subset=[TARGET])

    feature_cols = [c for c in train_df.columns if c != TARGET]

    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = np.nan

    cat_cols = []
    num_cols = []

    for col in feature_cols:
        dtype = train_df[col].dtype
        if dtype == object or dtype.name == "category" or str(dtype).startswith("string"):
            converted = pd.to_numeric(train_df[col], errors="coerce")
            non_null = int(train_df[col].notna().sum())
            if non_null > 0 and converted.notna().sum() / non_null > 0.8:
                num_cols.append(col)
            else:
                cat_cols.append(col)
        else:
            num_cols.append(col)

    for col in cat_cols:
        train_df[col] = train_df[col].astype(object).fillna("missing").astype(str)
        test_df[col] = test_df[col].astype(object).fillna("missing").astype(str)

    for col in num_cols:
        train_df[col] = _clean_numeric(train_df[col])
        test_df[col] = _clean_numeric(test_df[col])

        median = train_df[col].median()
        if pd.isna(median):
            median = 0.0

        train_df[col] = train_df[col].fillna(median)
        test_df[col] = test_df[col].fillna(median)

    if train_df.empty or not feature_cols:
        _write_predictions(np.zeros(len(test_df), dtype=float), len(test_df))
        return

    X = train_df[feature_cols].copy()
    y = train_df[TARGET].astype(float)
    X_test = test_df[feature_cols].copy()

    if len(test_df) == 0:
        _write_predictions(np.array([], dtype=float), 0)
        return

    if len(X) < 10:
        _write_predictions(np.full(len(test_df), float(y.mean()), dtype=float), len(test_df))
        return

    params = {
        "iterations": 2000,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": 0,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
    }

    if len(X) >= 20:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        model = CatBoostRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            cat_features=cat_cols,
        )

        best_iter = None
        try:
            best_iter = model.get_best_iteration()
        except Exception:
            best_iter = None

        if best_iter is None:
            best_iter = getattr(model, "best_iteration_", None)

        try:
            best_iter_int = int(best_iter)
        except Exception:
            best_iter_int = -1

        if best_iter_int < 0:
            best_iter_int = params["iterations"]
        else:
            best_iter_int = best_iter_int + 1

        best_iter_int = max(1, min(best_iter_int, params["iterations"]))
    else:
        best_iter_int = 500

    final_params = dict(params)
    final_params["iterations"] = best_iter_int

    final_model = CatBoostRegressor(**final_params)
    final_model.fit(X, y, cat_features=cat_cols)

    predictions = final_model.predict(X_test)
    _write_predictions(predictions, len(test_df))


if __name__ == "__main__":
    main()