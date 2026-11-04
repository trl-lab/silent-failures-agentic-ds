import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/power-plants/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/power-plants/data/test.parquet"
TARGET = "NAMPLT_CAP"
RANDOM_STATE = 42

ID_LIKE_COLUMNS = [
    "OBJECTID",
    "ID",
    "NAME",
    "PLANT",
    "PLANT_CODE",
    "OPERAT_ID",
]

MISSING_STRINGS = {
    "nan": "MISSING",
    "None": "MISSING",
    "NOT AVAILABLE": "MISSING",
    "": "MISSING",
    "<NA>": "MISSING",
}


def clean_missing_strings(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    return s.replace(MISSING_STRINGS)


def extract_year(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(np.nan, index=pd.Index([]), dtype=float)

    s = series.astype(str)
    s = s.replace({k: np.nan for k in MISSING_STRINGS})
    years = s.str.extract(r"(\d{4})")[0]
    return pd.to_numeric(years, errors="coerce")


def load_and_prepare_raw(train_path: str, test_path: str):
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)

    if TARGET not in train.columns:
        train[TARGET] = np.nan

    train = train.dropna(subset=[TARGET])

    drop_cols = [c for c in ID_LIKE_COLUMNS if c in train.columns]
    train = train.drop(columns=drop_cols, errors="ignore")
    test = test.drop(columns=drop_cols, errors="ignore")

    if "RUN_DATE" in train.columns:
        train["RUN_YEAR"] = extract_year(train["RUN_DATE"])
        test["RUN_YEAR"] = extract_year(test["RUN_DATE"])

    if "RET_DATE" in train.columns:
        train["RET_YEAR"] = extract_year(train["RET_DATE"])
        test["RET_YEAR"] = extract_year(test["RET_DATE"])

    train = train.drop(columns=["RUN_DATE", "RET_DATE"], errors="ignore")
    test = test.drop(columns=["RUN_DATE", "RET_DATE"], errors="ignore")

    feature_cols = [c for c in train.columns if c != TARGET]

    for col in feature_cols:
        if col not in test.columns:
            test[col] = np.nan

    if len(train) > 0:
        high_card_threshold = max(1000, int(0.5 * len(train)))
    else:
        high_card_threshold = 1000

    keep_cols = []
    for col in feature_cols:
        nunique = train[col].nunique(dropna=True)
        if nunique <= high_card_threshold:
            keep_cols.append(col)

    X_full = train[keep_cols].copy()
    y_full = pd.to_numeric(train[TARGET], errors="coerce")

    valid = y_full.notna()
    X_full = X_full.loc[valid].copy()
    y_full = y_full.loc[valid].copy()

    X_test = test[keep_cols].copy()

    return X_full, y_full, X_test


class FeaturePreprocessor:
    def __init__(self):
        self.numeric_cols = []
        self.categorical_cols = []
        self.medians = {}
        self.categories = {}

    def fit(self, X: pd.DataFrame) -> "FeaturePreprocessor":
        self.numeric_cols = []
        self.categorical_cols = []
        self.medians = {}
        self.categories = {}

        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.numeric_cols.append(col)
                median = X[col].median()
                if pd.isna(median):
                    median = 0.0
                self.medians[col] = float(median)
            else:
                self.categorical_cols.append(col)
                vals = clean_missing_strings(X[col]).dropna().unique()
                self.categories[col] = set(vals.tolist())

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for col in self.numeric_cols:
            if col not in X.columns:
                X[col] = np.nan
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(self.medians[col])

        for col in self.categorical_cols:
            if col not in X.columns:
                X[col] = "MISSING"
            X[col] = clean_missing_strings(X[col])
            cats = self.categories[col]
            X[col] = X[col].where(X[col].isin(cats), "MISSING")

        return X[self.numeric_cols + self.categorical_cols]

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


def main():
    X_full_raw, y_full, X_test_raw = load_and_prepare_raw(TRAIN_PATH, TEST_PATH)

    if len(y_full) > 0:
        fallback = float(y_full.mean())
        if not np.isfinite(fallback):
            fallback = 0.0
    else:
        fallback = 0.0

    if X_full_raw.shape[1] == 0 or len(X_full_raw) < 20:
        predictions = np.full(len(X_test_raw), fallback, dtype=float)
    else:
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X_full_raw,
            y_full,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )

        val_preprocessor = FeaturePreprocessor().fit(X_train_raw)
        X_train = val_preprocessor.transform(X_train_raw)
        X_val = val_preprocessor.transform(X_val_raw)

        cat_features = [
            X_train.columns.get_loc(c) for c in val_preprocessor.categorical_cols
        ]

        model = CatBoostRegressor(
            iterations=3000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            verbose=0,
            allow_writing_files=False,
            cat_features=cat_features,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=100,
            verbose=0,
        )

        best_iteration = getattr(model, "best_iteration_", None)
        if best_iteration is None:
            best_iteration = 3000
        else:
            best_iteration = int(best_iteration) + 1

        best_iteration = max(1, min(best_iteration, 3000))

        final_preprocessor = FeaturePreprocessor().fit(X_full_raw)
        X_full = final_preprocessor.transform(X_full_raw)
        X_test = final_preprocessor.transform(X_test_raw)

        final_cat_features = [
            X_full.columns.get_loc(c) for c in final_preprocessor.categorical_cols
        ]

        final_model = CatBoostRegressor(
            iterations=best_iteration,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            verbose=0,
            allow_writing_files=False,
            cat_features=final_cat_features,
        )

        final_model.fit(X_full, y_full, verbose=0)
        predictions = final_model.predict(X_test)

    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    predictions = np.nan_to_num(
        predictions,
        nan=fallback,
        posinf=fallback,
        neginf=fallback,
    )

    if len(predictions) != len(X_test_raw):
        if len(predictions) < len(X_test_raw):
            predictions = np.concatenate(
                [predictions, np.full(len(X_test_raw) - len(predictions), fallback)]
            )
        else:
            predictions = predictions[: len(X_test_raw)]

    output = pd.DataFrame(
        {
            "id": np.arange(len(predictions), dtype=int),
            "NAMPLT_CAP": predictions,
        }
    )
    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()