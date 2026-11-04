import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/health-professional-shortage-areas/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/health-professional-shortage-areas/data/test.parquet"
TARGET = "HPSA Score"
MISSING_TOKEN = "__missing__"


def drop_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = []
    for col in df.columns:
        c = str(col).lower()
        if col == TARGET:
            drop_cols.append(col)
            continue

        if any(token in c for token in ["id", "identification"]):
            drop_cols.append(col)
        elif any(token in c for token in ["name", "address", "city", "postal", "zip"]):
            drop_cols.append(col)
        elif "bhcmis" in c:
            drop_cols.append(col)
        elif "hpsa component source" in c:
            drop_cols.append(col)
        elif "hpsa geography identification" in c:
            drop_cols.append(col)

    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")


def parse_ratio_series(s: pd.Series) -> pd.Series:
    def _parse(x):
        if pd.isna(x):
            return np.nan
        x = str(x).strip()
        if not x or x.lower() in {"none", "nan", "null", "not applicable"}:
            return np.nan
        if ":" in x:
            x = x.split(":", 1)[0]
        try:
            return float(x)
        except Exception:
            return np.nan

    return s.map(_parse)


def parse_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    date_cols = [c for c in df.columns if "date" in str(c).lower()]
    for col in date_cols:
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().sum() == 0:
            df = df.drop(columns=[col], errors="ignore")
            continue

        df[f"{col}_year"] = parsed.dt.year.astype("float64")
        df[f"{col}_month"] = parsed.dt.month.astype("float64")
        df[f"{col}_day"] = parsed.dt.day.astype("float64")
        df = df.drop(columns=[col], errors="ignore")

    return df


def parse_ratio_columns(df: pd.DataFrame) -> pd.DataFrame:
    ratio_cols = [
        c for c in df.columns
        if ("ratio" in str(c).lower() or "goal" in str(c).lower()) and df[c].dtype == object
    ]
    for col in ratio_cols:
        parsed = parse_ratio_series(df[col])
        if parsed.notna().sum() > 0:
            df[col] = parsed
    return df


def remove_low_information_columns(df: pd.DataFrame, max_cardinality: int = 5000) -> pd.DataFrame:
    drop_cols = []
    for col in df.columns:
        if df[col].nunique(dropna=True) <= 1:
            drop_cols.append(col)
        elif df[col].dtype == object and df[col].nunique(dropna=True) > max_cardinality:
            drop_cols.append(col)
    return df.drop(columns=drop_cols, errors="ignore")


def is_categorical_series(s: pd.Series) -> bool:
    return (
        s.dtype == object
        or s.dtype.name == "category"
        or pd.api.types.is_string_dtype(s)
    )


def prepare_features(train: pd.DataFrame, test: pd.DataFrame):
    train = train.copy()
    test = test.copy()

    if TARGET in train.columns:
        y = train[TARGET].astype("float64").replace([np.inf, -np.inf], np.nan)
        train = train.drop(columns=[TARGET])
    else:
        y = pd.Series(dtype="float64")

    if TARGET in test.columns:
        test = test.drop(columns=[TARGET])

    train = drop_unwanted_columns(train)
    test = drop_unwanted_columns(test)

    train = parse_date_columns(train)
    test = parse_date_columns(test)

    train = parse_ratio_columns(train)
    test = parse_ratio_columns(test)

    train = remove_low_information_columns(train)
    test = remove_low_information_columns(test)

    common_cols = [c for c in train.columns if c in test.columns]
    train = train[common_cols]
    test = test[common_cols]

    categorical_cols = [c for c in common_cols if is_categorical_series(train[c])]

    for col in categorical_cols:
        train_str = (
            train[col]
            .astype("object")
            .where(train[col].notna(), MISSING_TOKEN)
            .astype(str)
            .str.strip()
        )
        train_str = train_str.replace("", MISSING_TOKEN)

        categories = sorted(set(train_str.unique()) | {MISSING_TOKEN})

        train_str = train_str.where(train_str.isin(categories), MISSING_TOKEN)
        train[col] = pd.Categorical(train_str, categories=categories)

        test_str = (
            test[col]
            .astype("object")
            .where(test[col].notna(), MISSING_TOKEN)
            .astype(str)
            .str.strip()
        )
        test_str = test_str.replace("", MISSING_TOKEN)
        test_str = test_str.where(test_str.isin(categories), MISSING_TOKEN)
        test[col] = pd.Categorical(test_str, categories=categories)

    numeric_cols = [c for c in common_cols if c not in categorical_cols]
    for col in numeric_cols:
        train[col] = pd.to_numeric(train[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        test[col] = pd.to_numeric(test[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    drop_cols = [c for c in common_cols if train[c].isna().all()]
    train = train.drop(columns=drop_cols, errors="ignore")
    test = test.drop(columns=drop_cols, errors="ignore")

    categorical_cols = [c for c in categorical_cols if c in train.columns]

    return train, test, y, categorical_cols


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    X, X_test, y, categorical_cols = prepare_features(train_df, test_df)

    valid_mask = y.notna()
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]

    fallback_value = float(y.mean()) if len(y) > 0 else 0.0

    if len(X) == 0 or X.shape[1] == 0 or len(y) == 0:
        predictions = np.full(len(X_test), fallback_value, dtype="float64")
    else:
        categorical_indices = [X.columns.get_loc(c) for c in categorical_cols if c in X.columns]

        base_params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": -1,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
            "categorical_feature": categorical_indices,
        }

        try:
            if len(X) > 50:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                model = lgb.LGBMRegressor(n_estimators=5000, **base_params)
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
                )

                best_iter = max(1, int(getattr(model, "best_iteration_", 5000)))
                full_model = lgb.LGBMRegressor(n_estimators=best_iter, **base_params)
                full_model.fit(X, y)
                predictions = full_model.predict(X_test)
            else:
                model = lgb.LGBMRegressor(n_estimators=500, **base_params)
                model.fit(X, y)
                predictions = model.predict(X_test)
        except Exception:
            predictions = np.full(len(X_test), fallback_value, dtype="float64")

        predictions = np.asarray(predictions, dtype="float64")
        predictions = np.nan_to_num(predictions, nan=fallback_value, posinf=fallback_value, neginf=fallback_value)

        if len(y) > 0:
            predictions = np.clip(predictions, float(y.min()), float(y.max()))

    output = pd.DataFrame({
        "id": np.arange(len(predictions), dtype=int),
        "HPSA Score": predictions.astype("float64")
    })
    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()