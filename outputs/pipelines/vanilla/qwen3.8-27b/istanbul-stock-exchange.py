import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/istanbul-stock-exchange/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/istanbul-stock-exchange/data/test.parquet"
TARGET = "ISE"
RANDOM_STATE = 42


def _as_float_series(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _prepare_date_features(frame, date_col="date"):
    if date_col not in frame.columns:
        return frame, None

    if len(frame) == 0:
        return frame.drop(columns=[date_col]), None

    parsed = pd.to_datetime(frame[date_col], errors="coerce")
    if parsed.notna().mean() < 0.5:
        return frame.drop(columns=[date_col]), None

    out = frame.drop(columns=[date_col]).copy()
    out["date_year"] = parsed.dt.year.astype("float64")
    out["date_month"] = parsed.dt.month.astype("float64")
    out["date_day"] = parsed.dt.day.astype("float64")
    out["date_dayofweek"] = parsed.dt.dayofweek.astype("float64")
    out["date_dayofyear"] = parsed.dt.dayofyear.astype("float64")
    out["date_quarter"] = parsed.dt.quarter.astype("float64")
    return out, parsed


def _drop_uninformative_columns(X, y=None, target_corr_threshold=0.999):
    X = X.copy()
    if X.shape[1] == 0:
        return X

    keep = []
    for col in X.columns:
        series = X[col]
        if series.isna().all():
            continue

        if series.nunique(dropna=True) <= 1:
            continue

        if y is not None and pd.api.types.is_numeric_dtype(series):
            mask = series.notna() & y.notna()
            if mask.sum() >= 10:
                x = series[mask].to_numpy(dtype=float)
                t = y[mask].to_numpy(dtype=float)
                finite = np.isfinite(x) & np.isfinite(t)
                if finite.sum() >= 10:
                    x = x[finite]
                    t = t[finite]
                    if np.std(x) > 0 and np.std(t) > 0:
                        corr = np.corrcoef(x, t)[0, 1]
                        if np.isfinite(corr) and abs(corr) >= target_corr_threshold:
                            continue

        keep.append(col)

    return X[keep]


def _make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _build_preprocessor(numeric_cols, categorical_cols):
    transformers = []

    if numeric_cols:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_transformer, list(numeric_cols)))

    if categorical_cols:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", _make_onehot_encoder()),
            ]
        )
        transformers.append(("cat", categorical_transformer, list(categorical_cols)))

    if not transformers:
        return None

    return ColumnTransformer(transformers=transformers, remainder="drop", n_jobs=1)


def _make_models():
    return {
        "ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=7,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=1,
            tree_method="hist",
        ),
        "catboost": CatBoostRegressor(
            iterations=300,
            learning_rate=0.03,
            depth=3,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            verbose=0,
            thread_count=1,
        ),
    }


def _rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.inf
    return float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))


def _save_predictions(predictions, default=0.0):
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    predictions = np.where(np.isfinite(predictions), predictions, float(default))
    pd.DataFrame(
        {
            "id": np.arange(len(predictions), dtype=int),
            TARGET: predictions,
        }
    ).to_csv("predictions.csv", index=False)


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET not in train_df.columns:
        raise ValueError(f"Target column {TARGET!r} not found in train data.")

    y_full = _as_float_series(train_df[TARGET])
    X_full = train_df.drop(columns=[TARGET], errors="ignore").copy()
    X_test = test_df.drop(columns=[TARGET], errors="ignore").copy()

    for col in X_full.columns:
        if col not in X_test.columns:
            X_test[col] = np.nan
    X_test = X_test[list(X_full.columns)].copy()

    X_full, train_dates = _prepare_date_features(X_full)
    X_test, _ = _prepare_date_features(X_test)

    for col in X_full.columns:
        if col not in X_test.columns:
            X_test[col] = np.nan
    X_test = X_test[list(X_full.columns)].copy()

    if len(X_test) == 0:
        _save_predictions(np.array([], dtype=float))
        return

    categorical_cols = []
    for col in X_full.columns:
        if (
            pd.api.types.is_object_dtype(X_full[col])
            or pd.api.types.is_categorical_dtype(X_full[col])
            or pd.api.types.is_bool_dtype(X_full[col])
        ):
            if X_full[col].nunique(dropna=True) > max(20, 0.5 * len(X_full)):
                continue
            categorical_cols.append(col)

    for col in categorical_cols:
        X_full[col] = X_full[col].astype("object")
        X_test[col] = X_test[col].astype("object")

    numeric_cols = [col for col in X_full.columns if col not in categorical_cols]
    for col in numeric_cols:
        X_full[col] = _as_float_series(X_full[col])
        X_test[col] = _as_float_series(X_test[col])

    X_full = _drop_uninformative_columns(X_full, y_full)
    X_test = X_test[X_full.columns].copy()

    numeric_cols = [col for col in X_full.columns if col not in categorical_cols]
    categorical_cols = [col for col in categorical_cols if col in X_full.columns]

    valid_mask = y_full.notna() & np.isfinite(y_full)
    X_valid = X_full[valid_mask].copy()
    y_valid = y_full[valid_mask].copy()

    if len(X_valid) <= 5 or X_valid.shape[1] == 0:
        _save_predictions(np.full(len(X_test), float(y_valid.mean()) if len(y_valid) else 0.0))
        return

    if train_dates is not None:
        train_dates_valid = train_dates[valid_mask].reset_index(drop=True)
        if train_dates_valid.notna().sum() >= max(10, int(0.2 * len(train_dates_valid))):
            date_values = train_dates_valid.values.astype("int64")
            order = np.argsort(date_values, kind="stable")
            X_sorted = X_valid.iloc[order].reset_index(drop=True)
            y_sorted = y_valid.iloc[order].reset_index(drop=True)

            val_size = max(1, int(0.2 * len(X_sorted)))
            X_train = X_sorted.iloc[:-val_size].copy()
            y_train = y_sorted.iloc[:-val_size].copy()
            X_val = X_sorted.iloc[-val_size:].copy()
            y_val = y_sorted.iloc[-val_size:].copy()
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_valid,
                y_valid,
                test_size=0.2,
                random_state=RANDOM_STATE,
            )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_valid,
            y_valid,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )

    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)
    if preprocessor is None:
        _save_predictions(np.full(len(X_test), float(y_valid.mean())))
        return

    preprocessor.fit(X_train)
    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    best_model = None
    best_score = np.inf

    for model in _make_models().values():
        try:
            model.fit(X_train_proc, y_train)
            val_pred = model.predict(X_val_proc)
            score = _rmse(y_val, val_pred)
            if np.isfinite(score) and score < best_score:
                best_score = score
                best_model = model
        except Exception:
            continue

    if best_model is None:
        _save_predictions(np.full(len(X_test), float(y_valid.mean())))
        return

    best_model = clone(best_model)
    X_all_proc = np.vstack([X_train_proc, X_val_proc])
    y_all = pd.concat([y_train, y_val], ignore_index=True)

    best_model.fit(X_all_proc, y_all)
    final_pred = best_model.predict(X_test_proc)

    _save_predictions(final_pred, default=float(y_valid.mean()))


if __name__ == "__main__":
    main()