import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/online-news-popularity/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/online-news-popularity/data/test.parquet"
TARGET = "shares"
EXCLUDE_COLUMNS = {"shares", "url", "timedelta"}
RANDOM_STATE = 42


def strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(col).strip() for col in df.columns]
    return df


def load_data(train_path: str, test_path: str):
    train_df = strip_columns(pd.read_parquet(train_path))
    test_df = strip_columns(pd.read_parquet(test_path))

    if TARGET not in train_df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in train data.")

    train_df = train_df.dropna(subset=[TARGET])
    train_df[TARGET] = pd.to_numeric(train_df[TARGET], errors="coerce")
    train_df = train_df.dropna(subset=[TARGET])

    feature_cols = [col for col in train_df.columns if col not in EXCLUDE_COLUMNS]
    feature_cols = [col for col in feature_cols if train_df[col].notna().any()]

    test_df = test_df.reindex(columns=feature_cols)

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan)
            test_df[col] = test_df[col].replace([np.inf, -np.inf], np.nan)

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET]
    X_test = test_df[feature_cols]

    return X_train, y_train, X_test, feature_cols


def split_feature_types(X_train: pd.DataFrame, feature_cols: list):
    categorical_cols = [
        col
        for col in feature_cols
        if pd.api.types.is_string_dtype(X_train[col])
        or pd.api.types.is_categorical_dtype(X_train[col])
    ]
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    transformers = []

    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_cols,
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )

    if not transformers:
        return ColumnTransformer([], remainder="drop")

    return ColumnTransformer(transformers, remainder="drop")


def make_model(use_log_target: bool) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def transform_target(y: pd.Series, use_log_target: bool) -> np.ndarray:
    y = y.to_numpy(dtype=float)
    if use_log_target:
        return np.log1p(np.maximum(y, 0.0))
    return y


def inverse_transform_target(y_pred: np.ndarray, use_log_target: bool) -> np.ndarray:
    if use_log_target:
        y_pred = np.expm1(y_pred)
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(y_pred, 0.0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    X_train, y_train, X_test, feature_cols = load_data(TRAIN_PATH, TEST_PATH)
    numeric_cols, categorical_cols = split_feature_types(X_train, feature_cols)

    if len(X_train) == 0:
        predictions = np.zeros(len(X_test), dtype=float)
    elif len(X_train) < 10:
        predictions = np.full(len(X_test), float(y_train.mean()), dtype=float)
    else:
        X_train_fit, X_val, y_train_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )

        best_score = np.inf
        best_use_log_target = False

        for use_log_target in (False, True):
            preprocessor = build_preprocessor(numeric_cols, categorical_cols)
            model = make_model(use_log_target)
            pipeline = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("model", model),
                ]
            )

            y_fit = transform_target(y_train_fit, use_log_target)
            pipeline.fit(X_train_fit, y_fit)

            y_val_pred = pipeline.predict(X_val)
            y_val_pred = inverse_transform_target(y_val_pred, use_log_target)
            score = rmse(y_val.to_numpy(dtype=float), y_val_pred)

            if score < best_score:
                best_score = score
                best_use_log_target = use_log_target

        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        model = make_model(best_use_log_target)
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        y_fit = transform_target(y_train, best_use_log_target)
        pipeline.fit(X_train, y_fit)

        predictions = pipeline.predict(X_test)
        predictions = inverse_transform_target(predictions, best_use_log_target)

    output = pd.DataFrame(
        {
            "id": np.arange(len(predictions), dtype=int),
            "shares": predictions.astype(float),
        }
    )
    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()