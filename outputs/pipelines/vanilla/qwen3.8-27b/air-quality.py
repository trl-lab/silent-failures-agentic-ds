import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/air-quality/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/air-quality/data/test.parquet"
TARGET = "CO(GT)"
RANDOM_STATE = 42


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].replace(-200, np.nan)
    return df


def parse_date_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({"nan": np.nan, "NaT": np.nan, "": np.nan})

    if s.isna().all():
        return pd.Series(pd.NaT, index=series.index)

    for fmt in ["%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d"]:
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        if parsed.notna().any():
            if parsed.isna().any():
                generic = pd.to_datetime(s, errors="coerce")
                parsed = parsed.fillna(generic)
            return parsed

    return pd.to_datetime(s, errors="coerce")


def parse_time_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({"nan": np.nan, "NaT": np.nan, "": np.nan})

    if s.isna().all():
        return pd.Series(pd.NaT, index=series.index)

    for fmt in ["%H:%M:%S", "%H:%M"]:
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        if parsed.notna().any():
            if parsed.isna().any():
                generic = pd.to_datetime(s, errors="coerce")
                parsed = parsed.fillna(generic)
            return parsed

    return pd.to_datetime(s, errors="coerce")


def make_datetime(df: pd.DataFrame) -> pd.Series:
    if "Date" in df.columns and "Time" in df.columns:
        date = parse_date_series(df["Date"])
        time = parse_time_series(df["Time"])
        time_delta = time - pd.Timestamp("1970-01-01")
        return date + time_delta

    if "Date" in df.columns:
        return parse_date_series(df["Date"])

    return pd.Series(pd.NaT, index=df.index)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Date" in df.columns:
        date = parse_date_series(df["Date"])
        df["year"] = date.dt.year
        df["month"] = date.dt.month
        df["day"] = date.dt.day
        df["dayofweek"] = date.dt.dayofweek
        df["dayofyear"] = date.dt.dayofyear

        df["is_weekend"] = (date.dt.dayofweek >= 5).astype(float)
        df["is_weekend"] = df["is_weekend"].fillna(0)

        df["month_sin"] = np.sin(2 * np.pi * date.dt.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * date.dt.month / 12)

    if "Time" in df.columns:
        time = parse_time_series(df["Time"])
        df["hour"] = time.dt.hour
        df["minute"] = time.dt.minute
        df["second"] = time.dt.second

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        night = ((df["hour"] < 6) | (df["hour"] >= 18)).astype(float)
        df["is_night"] = night.fillna(0)

    df = df.drop(columns=["Date", "Time"], errors="ignore")
    return df


def encode_categorical(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols):
    cat_cols = []
    for col in feature_cols:
        if col not in train_df.columns or col not in test_df.columns:
            continue
        if train_df[col].dtype == object or str(train_df[col].dtype) == "category":
            cat_cols.append(col)

    if not cat_cols:
        return train_df, test_df

    train_cat = train_df[cat_cols].astype(object).fillna("missing")
    test_cat = test_df[cat_cols].astype(object).fillna("missing")

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_df[cat_cols] = encoder.fit_transform(train_cat)
    test_df[cat_cols] = encoder.transform(test_cat)

    return train_df, test_df


def make_validation_split(X: pd.DataFrame, y: pd.Series, dt: pd.Series):
    n = len(X)
    if n <= 20:
        return X, y, X, y

    dt = dt.reindex(X.index)

    if dt.notna().sum() > max(20, n // 10):
        sorted_idx = dt.sort_values(kind="mergesort", na_position="last").index
        sorted_idx = sorted_idx.intersection(X.index, sort=False)

        if len(sorted_idx) == n:
            val_size = max(1, int(n * 0.2))
            if val_size < n:
                val_idx = sorted_idx[-val_size:]
                train_idx = sorted_idx[:-val_size]
                return X.loc[train_idx], X.loc[val_idx], y.loc[train_idx], y.loc[val_idx]

    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


def make_models():
    return {
        "lgbm": LGBMRegressor(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            min_child_samples=20,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": XGBRegressor(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            objective="reg:squarederror",
        ),
        "cat": CatBoostRegressor(
            iterations=800,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            random_seed=RANDOM_STATE,
            verbose=0,
            allow_writing_files=False,
        ),
    }


def sanitize_predictions(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(pred, 0.0, None)


def save_predictions(pred: np.ndarray, n: int) -> None:
    pred = sanitize_predictions(pred)
    pd.DataFrame(
        {
            "id": np.arange(n, dtype=int),
            TARGET: pred,
        }
    ).to_csv("predictions.csv", index=False)


def fallback_predictions(n: int, y: pd.Series) -> None:
    if len(y) > 0:
        fallback = float(np.nanmean(y))
        if not np.isfinite(fallback):
            fallback = 0.0
    else:
        fallback = 0.0

    save_predictions(np.full(n, fallback, dtype=float), n)


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET not in train_df.columns:
        fallback_predictions(len(test_df), pd.Series(dtype=float))
        return

    train_df = clean_numeric(train_df)
    test_df = clean_numeric(test_df)

    train_dt = make_datetime(train_df)

    train_df = add_features(train_df)
    test_df = add_features(test_df)

    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])

    feature_cols = [
        col
        for col in train_df.columns
        if col in test_df.columns and col != TARGET
    ]

    train_df, test_df = encode_categorical(train_df, test_df, feature_cols)

    y = train_df[TARGET].replace(-200, np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)
    valid = y.notna()

    X = train_df.loc[valid, feature_cols].copy()
    y = y.loc[valid].astype(float)

    feature_cols = [col for col in feature_cols if col in X.columns]
    feature_cols = [col for col in feature_cols if X[col].notna().any()]
    X = X[feature_cols]

    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    test_X = test_df[feature_cols].copy()
    for col in test_X.columns:
        if not pd.api.types.is_numeric_dtype(test_X[col]):
            test_X[col] = pd.to_numeric(test_X[col], errors="coerce")

    test_X = test_X.replace([np.inf, -np.inf], np.nan)

    if len(feature_cols) == 0 or len(X) == 0:
        fallback_predictions(len(test_df), y)
        return

    X_train, X_val, y_train, y_val = make_validation_split(X, y, train_dt)

    if len(X_train) < 2 or len(y_train) < 2:
        fallback_predictions(len(test_df), y)
        return

    models = make_models()
    val_models = {}
    val_rmses = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            val_pred = sanitize_predictions(model.predict(X_val))
            rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
            if np.isfinite(rmse):
                val_models[name] = model
                val_rmses[name] = rmse
        except Exception:
            continue

    if not val_rmses:
        try:
            fallback_model = LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )
            fallback_model.fit(X, y)
            test_pred = sanitize_predictions(fallback_model.predict(test_X))
            save_predictions(test_pred, len(test_X))
        except Exception:
            fallback_predictions(len(test_df), y)
        return

    inv_rmse = {name: 1.0 / max(rmse, 1e-8) for name, rmse in val_rmses.items()}
    total = sum(inv_rmse.values())
    weights = {name: value / total for name, value in inv_rmse.items()}

    final_models = make_models()
    test_pred = np.zeros(len(test_X), dtype=float)

    for name in val_rmses:
        try:
            final_model = final_models[name]
            final_model.fit(X, y)
            pred = sanitize_predictions(final_model.predict(test_X))
        except Exception:
            pred = sanitize_predictions(val_models[name].predict(test_X))

        test_pred += weights[name] * pred

    save_predictions(test_pred, len(test_X))


if __name__ == "__main__":
    main()