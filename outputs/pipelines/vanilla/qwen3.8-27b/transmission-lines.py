import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from lightgbm import LGBMClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/transmission-lines/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/transmission-lines/data/test.parquet"
TARGET = "VOLT_CLASS"
ID_COLS = ["OBJECTID", "ID"]
DATE_COLS = ["SOURCEDATE", "VAL_DATE"]
RANDOM_STATE = 42


def clean_text(series: pd.Series) -> pd.Series:
    s = series.astype("object")
    s = s.where(s.notna(), None)
    s = s.map(
        lambda x: "UNKNOWN"
        if x is None or str(x).strip() in ("", "nan", "NaN", "None", "null", "NULL")
        else str(x).strip()
    )
    return s


def get_datetime_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_datetime(df[col], errors="coerce")
    return pd.Series(pd.NaT, index=df.index)


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    dt_source = get_datetime_series(df, "SOURCEDATE")
    dt_val = get_datetime_series(df, "VAL_DATE")

    for name, dt in [("SOURCEDATE", dt_source), ("VAL_DATE", dt_val)]:
        df[f"{name}_year"] = dt.dt.year.astype("float64")
        df[f"{name}_month"] = dt.dt.month.astype("float64")
        df[f"{name}_day"] = dt.dt.day.astype("float64")
        df[f"{name}_dayofweek"] = dt.dt.dayofweek.astype("float64")

    df["VAL_MINUS_SOURCE_DAYS"] = (dt_val - dt_source).dt.days.astype("float64")

    for col in DATE_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


def prepare_categorical(train_df: pd.DataFrame, test_df: pd.DataFrame, cat_cols: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    for col in cat_cols:
        if col not in train_df.columns:
            train_df[col] = "UNKNOWN"
        if col not in test_df.columns:
            test_df[col] = "UNKNOWN"

        train_df[col] = clean_text(train_df[col])
        test_df[col] = clean_text(test_df[col])

        train_categories = set(train_df[col].unique())
        train_categories.add("UNKNOWN")

        test_df[col] = test_df[col].where(test_df[col].isin(train_categories), "UNKNOWN")

        categories = sorted(train_categories)
        train_df[col] = train_df[col].astype("category").cat.set_categories(categories)
        test_df[col] = test_df[col].astype("category").cat.set_categories(categories)

    return train_df, test_df


def prepare_numeric(train_df: pd.DataFrame, test_df: pd.DataFrame, num_cols: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    for col in num_cols:
        if col not in train_df.columns:
            train_df[col] = np.nan
        if col not in test_df.columns:
            test_df[col] = np.nan

        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

        median = train_df[col].median()
        if pd.isna(median):
            median = 0.0

        train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan).fillna(median)
        test_df[col] = test_df[col].replace([np.inf, -np.inf], np.nan).fillna(median)

    return train_df, test_df


def main() -> None:
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])

    for col in train_df.columns:
        if train_df[col].dtype == object:
            train_df[col] = clean_text(train_df[col])

    for col in test_df.columns:
        if test_df[col].dtype == object:
            test_df[col] = clean_text(test_df[col])

    train_df = add_date_features(train_df)
    test_df = add_date_features(test_df)

    drop_cols = [c for c in ID_COLS if c in train_df.columns]
    train_df = train_df.drop(columns=drop_cols, errors="ignore")
    test_df = test_df.drop(columns=drop_cols, errors="ignore")

    exclude_cols = set(ID_COLS) | {TARGET} | set(DATE_COLS)
    cat_cols = [
        col
        for col in train_df.columns
        if col not in exclude_cols and (train_df[col].dtype == object or str(train_df[col].dtype) == "category")
    ]
    num_cols = [
        col
        for col in train_df.columns
        if col not in exclude_cols and col not in cat_cols and pd.api.types.is_numeric_dtype(train_df[col])
    ]

    train_df, test_df = prepare_categorical(train_df, test_df, cat_cols)
    train_df, test_df = prepare_numeric(train_df, test_df, num_cols)

    if "Shape__Length" in train_df.columns:
        train_df["Shape__Length_log"] = np.log1p(train_df["Shape__Length"].clip(lower=0))
        test_df["Shape__Length_log"] = np.log1p(test_df["Shape__Length"].clip(lower=0))
        num_cols.append("Shape__Length_log")

    cat_cols = [col for col in cat_cols if train_df[col].nunique(dropna=True) > 1]
    feature_cols = cat_cols + num_cols

    X = train_df[feature_cols]
    y = train_df[TARGET]

    if len(test_df) == 0:
        pd.DataFrame({"id": [], "VOLT_CLASS": []}).to_csv("predictions.csv", index=False)
        return

    if y.nunique() <= 1:
        prediction = np.full(len(test_df), y.iloc[0])
    else:
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
            )
        except ValueError:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE
            )

        eval_metric = "logloss" if y.nunique() == 2 else "multi_logloss"

        model = LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=eval_metric,
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)
        print(f"Validation accuracy: {val_accuracy:.6f}")

        best_iteration = getattr(model, "best_iteration_", None)
        if best_iteration is None or best_iteration <= 0:
            best_iteration = model.n_estimators

        final_model = LGBMClassifier(
            n_estimators=int(best_iteration),
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )

        final_model.fit(X, y, categorical_feature=cat_cols)
        prediction = final_model.predict(test_df[feature_cols])

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            "VOLT_CLASS": prediction,
        }
    )
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()