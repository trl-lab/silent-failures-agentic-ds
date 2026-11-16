import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/foreign-gift-and-contract/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/foreign-gift-and-contract/data/test.parquet"
TARGET = "Amount"
ID_COLS = ["OPEID"]
MAX_CARDINALITY = 5000
RANDOM_STATE = 42


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    drop_cols = [c for c in ID_COLS if c in df.columns]
    if TARGET in df.columns:
        drop_cols.append(TARGET)
    return df.drop(columns=drop_cols)


def _as_datetime_seconds(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        s = series
        if getattr(s.dtype, "tz", None) is not None:
            s = s.dt.tz_convert(None)
        try:
            return s.astype("datetime64[s]")
        except Exception:
            return pd.Series(pd.NaT, index=s.index, dtype="datetime64[s]")

    try:
        parsed = pd.to_datetime(series, errors="coerce", format="ISO8601")
    except Exception:
        try:
            parsed = pd.to_datetime(series, errors="coerce")
        except Exception:
            return pd.Series(pd.NaT, index=series.index, dtype="datetime64[s]")

    try:
        return parsed.astype("datetime64[s]")
    except Exception:
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[s]")


def _add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

    for col in date_cols:
        d = _as_datetime_seconds(df[col])
        valid = d.notna()

        days = pd.Series(np.nan, index=d.index, dtype="float64")
        if valid.any():
            days[valid] = d[valid].astype("int64") / 86400.0

        df[f"{col}_days"] = days
        df[f"{col}_year"] = d.dt.year.astype("float64")
        df[f"{col}_month"] = d.dt.month.astype("float64")
        df[f"{col}_day"] = d.dt.day.astype("float64")
        df[f"{col}_weekday"] = d.dt.weekday.astype("float64")
        df = df.drop(columns=[col])

    if "Contract Start Date_days" in df.columns and "Contract End Date_days" in df.columns:
        df["Contract_Duration_Days"] = df["Contract End Date_days"] - df["Contract Start Date_days"]

    if "Receipt Date_days" in df.columns and "Contract Start Date_days" in df.columns:
        df["Receipt_To_Start_Days"] = df["Contract Start Date_days"] - df["Receipt Date_days"]

    return df


def _select_features(X: pd.DataFrame):
    numeric_cols = []
    categorical_cols = []

    for col in X.columns:
        s = X[col]

        if pd.api.types.is_numeric_dtype(s):
            if s.notna().sum() > 0 and s.nunique(dropna=True) > 1:
                numeric_cols.append(col)
        elif pd.api.types.is_string_dtype(s) or pd.api.types.is_categorical_dtype(s):
            if s.notna().sum() > 0 and 0 < s.nunique(dropna=True) <= MAX_CARDINALITY:
                categorical_cols.append(col)

    return numeric_cols, categorical_cols


def _prepare_frame(
    df: pd.DataFrame,
    feature_cols,
    numeric_cols,
    categorical_cols,
) -> pd.DataFrame:
    df = df.reindex(columns=feature_cols)

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("object").where(df[col].notna(), None)

    return df


def _make_model() -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=RANDOM_STATE,
        verbose=0,
        allow_writing_files=False,
    )


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    y = train_df[TARGET].astype("float64")
    X = train_df.drop(columns=[TARGET])

    valid = y.notna() & np.isfinite(y.to_numpy())
    X = X.loc[valid]
    y = y.loc[valid]

    X = _clean_frame(X)
    test = _clean_frame(test_df)

    X = _add_date_features(X)
    test = _add_date_features(test)

    numeric_cols, categorical_cols = _select_features(X)
    feature_cols = numeric_cols + categorical_cols

    if len(feature_cols) == 0:
        mean_amount = float(y.mean()) if len(y) > 0 else 0.0
        predictions = pd.DataFrame(
            {
                "id": np.arange(len(test), dtype="int64"),
                "Amount": mean_amount,
            }
        )
        predictions.to_csv("predictions.csv", index=False)
        return

    X = _prepare_frame(X, feature_cols, numeric_cols, categorical_cols)
    test = _prepare_frame(test, feature_cols, numeric_cols, categorical_cols)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    cat_features = [c for c in categorical_cols if c in X_train.columns]

    model = _make_model()
    model.fit(X_train, y_train, cat_features=cat_features)

    final_model = _make_model()
    final_model.fit(X, y, cat_features=cat_features)

    test_pred = final_model.predict(test)

    predictions = pd.DataFrame(
        {
            "id": np.arange(len(test_pred), dtype="int64"),
            "Amount": np.asarray(test_pred, dtype="float64"),
        }
    )
    predictions.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()