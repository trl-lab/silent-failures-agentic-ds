import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/historic-perimeters-wildfires/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/historic-perimeters-wildfires/data/test.parquet"
TARGET = "gisacres"
RANDOM_STATE = 42


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def is_id_like(col: str) -> bool:
    c = str(col).lower()
    tokens = [
        "objectid",
        "identifier",
        "uuid",
        "irwinid",
        "inciwebid",
        "uniquefireidentifier",
        "localincidentidentifier",
        "complexparentirwinid",
        "incidentname",
        "complexname",
        "comments",
    ]
    return any(t in c for t in tokens) or c.endswith("id")


def safe_to_datetime(series: pd.Series) -> pd.Series:
    s = series.astype("string")

    if s.isna().all():
        return pd.to_datetime(
            pd.Series(index=series.index, dtype="string"),
            errors="coerce",
            utc=True,
        )

    try:
        res = pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
        if res.notna().any():
            return res
    except Exception:
        pass

    try:
        return pd.to_datetime(s, errors="coerce", utc=True)
    except Exception:
        return pd.to_datetime(
            pd.Series(index=series.index, dtype="string"),
            errors="coerce",
            utc=True,
        )


def parse_date_features(df: pd.DataFrame, date_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    parsed = {}

    for col in date_cols:
        if col not in df.columns:
            continue

        dt = safe_to_datetime(df[col])
        parsed[col] = dt

        df[f"{col}_year"] = dt.dt.year.astype("float32")
        df[f"{col}_month"] = dt.dt.month.astype("float32")
        df[f"{col}_day"] = dt.dt.day.astype("float32")
        df[f"{col}_hour"] = dt.dt.hour.astype("float32")
        df[f"{col}_dayofweek"] = dt.dt.dayofweek.astype("float32")
        df[f"{col}_dayofyear"] = dt.dt.dayofyear.astype("float32")

    if "datecurrent" in parsed and "perimeterdatetime" in parsed:
        dc = parsed["datecurrent"]
        pdt = parsed["perimeterdatetime"]
        mask = dc.notna() & pdt.notna()

        if mask.any():
            fill = pd.Timestamp("2000-01-01", tz="UTC")
            dc_safe = dc.where(mask, fill)
            pdt_safe = pdt.where(mask, fill)

            try:
                diff = (dc_safe - pdt_safe).dt.total_seconds() / 86400.0
            except Exception:
                try:
                    dc_i = dc_safe.astype("int64").to_numpy()
                    pd_i = pdt_safe.astype("int64").to_numpy()
                    diff = (dc_i - pd_i) / 1e9 / 86400.0
                except Exception:
                    diff = np.full(len(df), np.nan, dtype="float64")

            df["date_diff_days"] = pd.Series(diff, index=df.index).where(mask, np.nan).astype("float32")
        else:
            df["date_diff_days"] = np.nan

    for col in date_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


def prepare_features(train: pd.DataFrame, test: pd.DataFrame):
    train = train.copy()
    test = test.copy()

    if TARGET in test.columns:
        test = test.drop(columns=[TARGET])

    train = train.dropna(subset=[TARGET]).copy()
    y_full = train[TARGET].astype("float64")
    X_full = train.drop(columns=[TARGET])

    date_cols = [
        col
        for col in X_full.columns
        if ("date" in str(col).lower() or "datetime" in str(col).lower())
        and not is_id_like(col)
    ]

    X_full = parse_date_features(X_full, date_cols)
    test = parse_date_features(test, date_cols)

    drop_cols = [col for col in X_full.columns if is_id_like(col)]
    X_full = X_full.drop(columns=drop_cols, errors="ignore")
    test = test.drop(columns=drop_cols, errors="ignore")

    for col in X_full.columns:
        if X_full[col].isna().all():
            X_full = X_full.drop(columns=[col])
            if col in test.columns:
                test = test.drop(columns=[col])

    numeric_cols = [
        col
        for col in X_full.columns
        if pd.api.types.is_numeric_dtype(X_full[col])
    ]

    categorical_cols = [
        col
        for col in X_full.columns
        if not pd.api.types.is_numeric_dtype(X_full[col])
    ]

    keep_categorical = []
    for col in categorical_cols:
        nunique = X_full[col].nunique(dropna=True)
        missing_frac = X_full[col].isna().mean()
        if nunique <= 1 or missing_frac > 0.98:
            continue
        if nunique > 1000:
            continue
        keep_categorical.append(col)

    drop_categorical = [col for col in categorical_cols if col not in keep_categorical]
    X_full = X_full.drop(columns=drop_categorical, errors="ignore")
    test = test.drop(columns=drop_categorical, errors="ignore")

    for col in keep_categorical:
        X_full[col] = X_full[col].astype("category")
        if col in test.columns:
            test[col] = test[col].astype("category")
            test[col] = test[col].cat.set_categories(X_full[col].cat.categories)

    feature_cols = list(X_full.columns)
    X_full = X_full.reindex(columns=feature_cols)
    test = test.reindex(columns=feature_cols)

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(X_full[col]):
            X_full[col] = X_full[col].replace([np.inf, -np.inf], np.nan).astype("float32")
            if col in test.columns:
                test[col] = test[col].replace([np.inf, -np.inf], np.nan).astype("float32")

    return X_full, y_full, test, keep_categorical


def transform_target(y: np.ndarray, kind: str) -> np.ndarray:
    if kind == "identity":
        return y.astype("float64")
    if kind == "log1p":
        return np.log1p(np.maximum(y, 0.0))
    if kind == "sqrt":
        return np.sqrt(np.maximum(y, 0.0))
    raise ValueError(f"Unknown transform: {kind}")


def inverse_target(y: np.ndarray, kind: str) -> np.ndarray:
    if kind == "identity":
        return y
    if kind == "log1p":
        return np.expm1(y)
    if kind == "sqrt":
        return y * y
    raise ValueError(f"Unknown transform: {kind}")


def make_model():
    return LGBMRegressor(
        objective="regression",
        n_estimators=1200,
        learning_rate=0.04,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    X_full, y_full, X_test, categorical_cols = prepare_features(train_df, test_df)

    X_train, X_val, y_train, y_val = train_test_split(
        X_full,
        y_full,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    transforms = ["identity", "log1p", "sqrt"]
    best_transform = "log1p"
    best_val_rmse = np.inf

    for kind in transforms:
        y_train_t = transform_target(y_train.to_numpy(), kind)
        y_val_t = transform_target(y_val.to_numpy(), kind)

        model = make_model()
        model.fit(X_train, y_train_t)

        val_pred_t = model.predict(X_val)
        val_pred = inverse_target(val_pred_t, kind)
        val_pred = np.maximum(val_pred, 0.0)

        score = rmse(y_val.to_numpy(), val_pred)
        if score < best_val_rmse:
            best_val_rmse = score
            best_transform = kind

    y_full_t = transform_target(y_full.to_numpy(), best_transform)
    final_model = make_model()
    final_model.fit(X_full, y_full_t)

    test_pred_t = final_model.predict(X_test)
    test_pred = inverse_target(test_pred_t, best_transform)
    test_pred = np.maximum(test_pred, 0.0)

    output = pd.DataFrame(
        {
            "id": np.arange(len(test_pred), dtype=int),
            "gisacres": test_pred.astype("float64"),
        }
    )
    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()