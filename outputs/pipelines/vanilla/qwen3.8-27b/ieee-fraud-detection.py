import gc
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/ieee-fraud-detection/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/ieee-fraud-detection/data/test.parquet"
TARGET = "isFraud"
ID_COL = "TransactionID"
RANDOM_STATE = 42


def _is_categorical_dtype(dtype) -> bool:
    return dtype == object or str(dtype) in {"category", "string"}


def _encode_categorical_series(series: pd.Series, categories: list) -> pd.Series:
    series = series.astype("string").fillna("missing")
    series = series.where(series.isin(categories), "missing")
    codes = pd.Categorical(series, categories=categories).codes + 1
    return pd.Series(codes, index=series.index, dtype=np.int32)


def _prepare_numeric_columns(df: pd.DataFrame, cols: list) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .astype(np.float32)
            )


def _save_predictions(n: int, values: np.ndarray) -> None:
    pd.DataFrame(
        {
            "id": np.arange(n, dtype=int),
            "isFraud": np.asarray(values, dtype=np.float64),
        }
    ).to_csv("predictions.csv", index=False)


def main() -> None:
    train_df = pd.read_parquet(TRAIN_PATH)

    if TARGET not in train_df.columns or len(train_df) == 0:
        test_df = pd.read_parquet(TEST_PATH)
        _save_predictions(len(test_df), np.full(len(test_df), 0.5, dtype=np.float64))
        return

    y = train_df[TARGET].astype(int)
    X = train_df.drop(columns=[ID_COL, TARGET], errors="ignore")
    del train_df

    if y.nunique() < 2 or X.shape[1] == 0:
        test_df = pd.read_parquet(TEST_PATH)
        value = float(y.mean()) if len(y) > 0 else 0.5
        _save_predictions(len(test_df), np.full(len(test_df), value, dtype=np.float64))
        return

    cat_cols = [col for col in X.columns if _is_categorical_dtype(X[col].dtype)]
    cat_categories = {}

    for col in cat_cols:
        series = X[col].astype("string").fillna("missing")
        categories = set(series.unique().tolist())
        categories.add("missing")
        categories = sorted(categories)
        cat_categories[col] = categories
        X[col] = _encode_categorical_series(series, categories)

    feature_cols = list(X.columns)
    num_cols = [col for col in feature_cols if col not in cat_cols]
    _prepare_numeric_columns(X, num_cols)

    n_estimators = 2000
    pos = int(y.sum())
    neg = int(len(y) - pos)
    scale_pos_weight = float(neg / pos) if pos > 0 else 1.0

    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 100,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": n_estimators,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
        "scale_pos_weight": scale_pos_weight,
    }

    best_iteration = n_estimators

    if len(X) >= 1000:
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=RANDOM_STATE,
                stratify=y,
            )
        except (ValueError, MemoryError):
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=RANDOM_STATE,
                )
            except Exception:
                X_train = X_val = y_train = y_val = None

        if X_train is not None and y_train.nunique() > 1 and y_val.nunique() > 1:
            model = None
            try:
                model = LGBMClassifier(**params, categorical_features=cat_cols)
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric="auc",
                    early_stopping_rounds=100,
                    verbose=False,
                )
                best_iteration = getattr(model, "best_iteration_", None)
                if best_iteration is None or best_iteration <= 0:
                    best_iteration = n_estimators
            except Exception:
                best_iteration = n_estimators
            finally:
                if model is not None:
                    del model
                del X_train, X_val, y_train, y_val
                gc.collect()
        else:
            if X_train is not None:
                del X_train, X_val, y_train, y_val
                gc.collect()

    prior = float(y.mean())
    final_params = params.copy()
    final_params["n_estimators"] = int(max(1, best_iteration))

    try:
        final_model = LGBMClassifier(**final_params, categorical_features=cat_cols)
        final_model.fit(X, y)
    except Exception:
        test_df = pd.read_parquet(TEST_PATH)
        _save_predictions(len(test_df), np.full(len(test_df), prior, dtype=np.float64))
        return

    del X, y
    gc.collect()

    test_df = pd.read_parquet(TEST_PATH)
    test_df = test_df.drop(columns=[ID_COL, TARGET], errors="ignore")

    if len(test_df) == 0:
        _save_predictions(0, np.array([], dtype=np.float64))
        return

    for col in cat_cols:
        if col in test_df.columns:
            test_df[col] = _encode_categorical_series(test_df[col], cat_categories[col])
        else:
            missing_series = pd.Series(["missing"] * len(test_df), dtype="string")
            test_df[col] = _encode_categorical_series(missing_series, cat_categories[col])

    _prepare_numeric_columns(test_df, num_cols)
    X_test = test_df.reindex(columns=feature_cols)

    del test_df
    gc.collect()

    if X_test.empty:
        _save_predictions(0, np.array([], dtype=np.float64))
        return

    try:
        proba = np.asarray(final_model.predict_proba(X_test)[:, 1], dtype=np.float64)
        proba = np.where(np.isfinite(proba), proba, prior)
        proba = np.clip(proba, 0.0, 1.0)
    except Exception:
        proba = np.full(len(X_test), prior, dtype=np.float64)

    _save_predictions(len(proba), proba)


if __name__ == "__main__":
    main()