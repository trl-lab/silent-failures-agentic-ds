import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/medically-underserved-areas-populations/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/medically-underserved-areas-populations/data/test.parquet"
TARGET = "IMU Score"
MAX_CARDINALITY = 50
RANDOM_STATE = 42


def drop_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    drop_cols = []
    for col in df.columns:
        col_lower = str(col).lower()
        if col_lower in {"mua/p id", "mua/p area code", "mua/p service area name"}:
            drop_cols.append(col)
        elif "date string" in col_lower or "text format" in col_lower:
            drop_cols.append(col)
    return df.drop(columns=drop_cols, errors="ignore")


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    pop1 = "Designation Population in a Medically Underserved Area/Population (MUA/P)"
    pop2 = "Medically Underserved Area/Population (MUA/P) Total Resident Civilian Population"
    try:
        if pop1 in df.columns and pop2 in df.columns:
            p1 = pd.to_numeric(df[pop1], errors="coerce")
            p2 = pd.to_numeric(df[pop2], errors="coerce")
            df["log_designation_population"] = np.log1p(p1.clip(lower=0))
            df["log_total_population"] = np.log1p(p2.clip(lower=0))
            df["population_ratio"] = p1 / p2.replace(0, np.nan)
            df["population_diff"] = p1 - p2
    except Exception:
        pass

    for col in list(df.columns):
        try:
            if isinstance(col, str) and "fips" in col.lower() and not pd.api.types.is_numeric_dtype(df[col]):
                new_col = f"{col}_num"
                if new_col not in df.columns:
                    df[new_col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            pass

    for col in list(df.columns):
        try:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                series = df[col]
                if getattr(series.dt, "tz", None) is not None:
                    series = series.dt.tz_convert("UTC")
                vals = series.astype("datetime64[ns]").astype("int64")
                vals = vals.replace(pd.NaT.value, np.nan)
                df[col] = vals / 1e9
        except Exception:
            df[col] = np.nan

    designation = "Designation Date"
    update = "MUA/P Update Date"
    withdrawal = "Medically Underserved Area/Population (MUA/P) Withdrawal Date"
    try:
        if designation in df.columns and update in df.columns:
            df["years_update_designation"] = df[update] - df[designation]
        if update in df.columns and withdrawal in df.columns:
            df["years_withdrawal_update"] = df[withdrawal] - df[update]
        if designation in df.columns and withdrawal in df.columns:
            df["years_withdrawal_designation"] = df[withdrawal] - df[designation]
    except Exception:
        pass

    return df


def clean_categorical_series(s: pd.Series) -> pd.Series:
    s = s.astype("object")
    s = s.mask(s.isna(), "missing")
    s = s.astype(str)
    s = s.replace({"nan": "missing", "None": "missing", "": "missing"})
    return s


def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_df = train_df.copy()
    test_df = test_df.copy()

    if TARGET in train_df.columns:
        train_df = train_df.drop(columns=[TARGET])
    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])

    train_df = drop_unwanted_columns(train_df)
    test_df = drop_unwanted_columns(test_df)

    train_df = add_derived_features(train_df)
    test_df = add_derived_features(test_df)

    common_cols = [c for c in train_df.columns if c in test_df.columns]
    train_df = train_df[common_cols]
    test_df = test_df[common_cols]

    numeric_cols = []
    categorical_cols = []

    for col in common_cols:
        try:
            if pd.api.types.is_numeric_dtype(train_df[col]) or pd.api.types.is_bool_dtype(train_df[col]):
                numeric_cols.append(col)
            else:
                nunique = train_df[col].nunique(dropna=True)
                if 0 < nunique <= MAX_CARDINALITY:
                    categorical_cols.append(col)
        except Exception:
            continue

    valid_numeric = []
    for col in numeric_cols:
        try:
            train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
            test_df[col] = pd.to_numeric(test_df[col], errors="coerce")
            train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan)
            test_df[col] = test_df[col].replace([np.inf, -np.inf], np.nan)
            valid_numeric.append(col)
        except Exception:
            pass
    numeric_cols = valid_numeric

    valid_categorical = []
    for col in categorical_cols:
        try:
            train_df[col] = clean_categorical_series(train_df[col])
            test_df[col] = clean_categorical_series(test_df[col])
            valid_categorical.append(col)
        except Exception:
            pass
    categorical_cols = valid_categorical

    feature_cols = [c for c in numeric_cols + categorical_cols if c in train_df.columns and c in test_df.columns]
    train_df = train_df[feature_cols]
    test_df = test_df[feature_cols]

    return train_df, test_df, numeric_cols, categorical_cols


def make_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_pipeline(n_estimators: int, numeric_cols, categorical_cols):
    transformers = []
    if numeric_cols:
        transformers.append(("num", "passthrough", numeric_cols))
    if categorical_cols:
        transformers.append(("cat", make_encoder(), categorical_cols))

    if not transformers:
        return None

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    model = LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("prep", preprocessor), ("model", model)])


def safe_predict(pipeline, X):
    try:
        pred = pipeline.predict(X)
        pred = np.asarray(pred, dtype=float).reshape(-1)
        return pred
    except Exception:
        return None


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET in train_df.columns:
        y = pd.to_numeric(train_df[TARGET], errors="coerce")
    else:
        y = pd.Series(np.zeros(len(train_df), dtype=float), index=train_df.index)

    valid = y.notna() & np.isfinite(y)
    y = y[valid]
    train_df = train_df.loc[valid]

    X, X_test, numeric_cols, categorical_cols = prepare_data(train_df, test_df)
    fallback = float(y.mean()) if len(y) else 0.0

    if X.shape[1] == 0 or len(X) == 0:
        preds = np.full(len(X_test), fallback, dtype=float)
    else:
        if len(X) < 5000:
            candidates = [100, 200, 400]
        elif len(X) < 20000:
            candidates = [200, 400, 800]
        else:
            candidates = [300, 600, 1200]

        best_n = candidates[0]
        best_rmse = np.inf

        if len(X) >= 50:
            X_fit, X_val, y_fit, y_val = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE
            )

            for n in candidates:
                pipe = make_pipeline(n, numeric_cols, categorical_cols)
                if pipe is None:
                    continue

                try:
                    pipe.fit(X_fit, y_fit)
                except Exception:
                    continue

                pred = safe_predict(pipe, X_val)
                if pred is None or len(pred) != len(y_val):
                    continue

                pred = np.where(np.isfinite(pred), pred, float(y_val.mean()))
                rmse = float(np.sqrt(np.mean((y_val - pred) ** 2)))

                if np.isfinite(rmse) and rmse < best_rmse:
                    best_rmse = rmse
                    best_n = n

        final_pipe = make_pipeline(best_n, numeric_cols, categorical_cols)

        if final_pipe is None:
            preds = np.full(len(X_test), fallback, dtype=float)
        else:
            try:
                final_pipe.fit(X, y)
            except Exception:
                preds = np.full(len(X_test), fallback, dtype=float)
            else:
                preds = safe_predict(final_pipe, X_test)
                if preds is None or len(preds) != len(X_test):
                    preds = np.full(len(X_test), fallback, dtype=float)
                else:
                    preds = np.where(np.isfinite(preds), preds, fallback)
                    finite_y = y[np.isfinite(y)]
                    if len(finite_y):
                        preds = np.clip(preds, float(finite_y.min()), float(finite_y.max()))

    output = pd.DataFrame(
        {
            "id": np.arange(len(X_test), dtype=int),
            "IMU Score": preds.astype(float),
        }
    )
    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()