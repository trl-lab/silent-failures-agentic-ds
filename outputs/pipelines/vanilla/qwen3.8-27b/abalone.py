import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/abalone/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/abalone/data/test.parquet"
TARGET = "Class_number_of_rings"
RANDOM_STATE = 42


def clean_numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for col in columns:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
            frame[col] = frame[col].replace([np.inf, -np.inf], np.nan)
    return frame


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df[TARGET] = pd.to_numeric(train_df[TARGET], errors="coerce")
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    train_df = train_df.dropna(subset=[TARGET])

    y_train = train_df[TARGET].astype(float)
    X_train = train_df.drop(columns=[TARGET])
    X_test = test_df.drop(columns=[TARGET], errors="ignore")

    categorical_cols = [
        col
        for col in X_train.columns
        if X_train[col].dtype == object or str(X_train[col].dtype) == "category"
    ]
    numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

    X_train = clean_numeric_frame(X_train, numeric_cols)
    X_test = clean_numeric_frame(X_test, numeric_cols)

    for col in numeric_cols:
        median = X_train[col].median()
        if pd.isna(median):
            median = 0.0
        X_train[col] = X_train[col].fillna(median)
        if col in X_test.columns:
            X_test[col] = X_test[col].fillna(median)
        else:
            X_test[col] = median

    for col in categorical_cols:
        train_series = X_train[col].astype("object")
        test_series = X_test[col].astype("object") if col in X_test.columns else pd.Series(index=X_test.index, dtype="object")

        if train_series.isna().any():
            fill_value = "missing"
        else:
            mode = train_series.mode(dropna=True)
            fill_value = mode.iloc[0] if len(mode) else "missing"

        train_series = train_series.fillna(fill_value)
        test_series = test_series.fillna(fill_value)

        categories = sorted(train_series.unique().tolist())
        for category in categories:
            encoded_name = f"{col}__{category}"
            X_train[encoded_name] = (train_series == category).astype(int)
            X_test[encoded_name] = (test_series == category).astype(int)

        X_train = X_train.drop(columns=[col])
        X_test = X_test.drop(columns=[col], errors="ignore")

    for col in X_train.columns:
        if col not in X_test.columns:
            X_test[col] = 0

    X_test = X_test[X_train.columns]

    return X_train, y_train, X_test


def train_lightgbm(X_fit, y_fit, X_val, y_val, X_full, y_full, X_test):
    params = {
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "min_child_samples": 20,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }

    model = LGBMRegressor(**params)
    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=False,
    )

    val_pred = model.predict(X_val)
    val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))

    best_iter = getattr(model, "best_iteration_", None)
    if best_iter is None or best_iter < 1:
        best_iter = params["n_estimators"]

    final_model = LGBMRegressor(**{**params, "n_estimators": int(best_iter)})
    final_model.fit(X_full, y_full)

    return final_model.predict(X_test), val_rmse


def train_xgboost(X_fit, y_fit, X_val, y_val, X_full, y_full, X_test):
    params = {
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
    }

    model = XGBRegressor(**params)
    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=False,
    )

    val_pred = model.predict(X_val)
    val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))

    best_iter = getattr(model, "best_iteration", None)
    if best_iter is None or best_iter < 0:
        best_iter = params["n_estimators"]
    else:
        best_iter = best_iter + 1

    final_model = XGBRegressor(**{**params, "n_estimators": int(best_iter)})
    final_model.fit(X_full, y_full)

    return final_model.predict(X_test), val_rmse


def train_catboost(X_fit, y_fit, X_val, y_val, X_full, y_full, X_test):
    params = {
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": RANDOM_STATE,
        "thread_count": -1,
        "verbose": 0,
    }

    model = CatBoostRegressor(**params)
    model.fit(
        X_fit,
        y_fit,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        verbose=0,
    )

    val_pred = model.predict(X_val)
    val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))

    best_iter = model.get_best_iteration()
    if best_iter is None or best_iter < 0:
        best_iter = params["iterations"]
    else:
        best_iter = best_iter + 1

    final_model = CatBoostRegressor(**{**params, "iterations": int(best_iter)})
    final_model.fit(X_full, y_full, verbose=0)

    return final_model.predict(X_test), val_rmse


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    X_train, y_train, X_test = prepare_features(train_df, test_df)

    if len(X_train) > 10:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )
    else:
        X_fit, X_val, y_fit, y_val = X_train, X_train, y_train, y_train

    model_results = []

    for trainer in (train_lightgbm, train_xgboost, train_catboost):
        try:
            pred, val_rmse = trainer(
                X_fit,
                y_fit,
                X_val,
                y_val,
                X_train,
                y_train,
                X_test,
            )
            model_results.append((pred, val_rmse))
        except Exception:
            continue

    if model_results:
        weights = np.array([1.0 / max(val_rmse, 1e-8) for _, val_rmse in model_results])
        weights = weights / weights.sum()
        final_pred = np.zeros(len(X_test), dtype=float)
        for (pred, _), weight in zip(model_results, weights):
            final_pred += weight * pred
    else:
        fallback = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        fallback.fit(X_train, y_train)
        final_pred = fallback.predict(X_test)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            TARGET: final_pred.astype(float),
        }
    )
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()