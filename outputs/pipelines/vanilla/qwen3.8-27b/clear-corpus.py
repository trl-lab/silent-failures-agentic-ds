import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/clear-corpus/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/clear-corpus/data/test.parquet"
TARGET = "BT Easiness"


def add_text_features(df):
    df = df.copy()

    if "Excerpt" in df.columns:
        text = df["Excerpt"].fillna("").astype(str)

        char_count = text.str.len()
        word_count = text.str.split().str.len()
        sentence_count = text.str.count(r"[.!?]+")
        punct_count = text.str.count(r"[^\w\s]")
        digit_count = text.str.count(r"\d")
        upper_count = text.str.count(r"[A-Z]")
        comma_count = text.str.count(",")

        df["excerpt_char_count"] = char_count
        df["excerpt_word_count"] = word_count
        df["excerpt_sentence_count"] = sentence_count
        df["excerpt_punct_count"] = punct_count
        df["excerpt_digit_count"] = digit_count
        df["excerpt_upper_count"] = upper_count
        df["excerpt_comma_count"] = comma_count

        df["excerpt_avg_word_len"] = char_count / word_count.replace(0, np.nan)
        df["excerpt_avg_sentence_len"] = char_count / sentence_count.replace(0, np.nan)
        df["excerpt_punct_ratio"] = punct_count / char_count.replace(0, np.nan)
        df["excerpt_digit_ratio"] = digit_count / char_count.replace(0, np.nan)
        df["excerpt_upper_ratio"] = upper_count / char_count.replace(0, np.nan)
        df["excerpt_comma_ratio"] = comma_count / char_count.replace(0, np.nan)

        df = df.replace([np.inf, -np.inf], np.nan)

    return df


def prepare(df):
    df = add_text_features(df)

    drop_cols = {
        "ID",
        TARGET,
        "Author",
        "Title",
        "Anthology",
        "URL",
        "Excerpt",
        "British Words",
    }

    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    return df


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(numeric_cols, cat_cols):
    transformers = []

    if numeric_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric_cols))

    if cat_cols:
        transformers.append(("cat", make_ohe(), cat_cols))

    if not transformers:
        return "passthrough"

    return ColumnTransformer(transformers=transformers, remainder="drop")


def train_and_refit(model_factory, X_fit, y_fit, X_val, y_val, X_full, y_full, X_test, numeric_cols, cat_cols):
    preprocessor = make_preprocessor(numeric_cols, cat_cols)
    pipeline = Pipeline([("prep", preprocessor), ("model", model_factory())])

    pipeline.fit(X_fit, y_fit)

    val_pred = pipeline.predict(X_val)
    val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
    if not np.isfinite(val_rmse):
        val_rmse = 1.0

    pipeline.fit(X_full, y_full)
    test_pred = pipeline.predict(X_test)

    return np.asarray(test_pred, dtype=float), val_rmse


def main():
    train_raw = pd.read_parquet(TRAIN_PATH)
    test_raw = pd.read_parquet(TEST_PATH)

    y_train = train_raw[TARGET].copy()
    train_df = prepare(train_raw)
    test_df = prepare(test_raw)

    valid = y_train.notna()
    y_train = y_train[valid].reset_index(drop=True)
    train_df = train_df.loc[valid].reset_index(drop=True)

    X_train = train_df
    X_test = test_df.reindex(columns=X_train.columns)

    if len(X_train) == 0:
        pd.DataFrame({
            "id": np.arange(len(X_test), dtype=int),
            TARGET: np.zeros(len(X_test), dtype=float)
        }).to_csv("predictions.csv", index=False)
        return

    if len(X_test) == 0:
        pd.DataFrame({
            "id": np.arange(0, dtype=int),
            TARGET: np.array([], dtype=float)
        }).to_csv("predictions.csv", index=False)
        return

    numeric_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    cat_cols = [c for c in X_train.columns if c not in numeric_cols]

    cat_cols = [c for c in cat_cols if X_train[c].nunique(dropna=True) <= 100]

    for col in numeric_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    for col in cat_cols:
        X_train[col] = X_train[col].where(X_train[col].notna(), "missing").astype(str)
        X_test[col] = X_test[col].where(X_test[col].notna(), "missing").astype(str)

    if len(X_train) >= 10:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=42
        )
    else:
        X_fit, X_val, y_fit, y_val = X_train, X_train, y_train, y_train

    def lgbm_factory():
        return LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    def xgb_factory():
        return XGBRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
        )

    def cat_factory():
        return CatBoostRegressor(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            random_seed=42,
            verbose=0,
            thread_count=-1,
        )

    model_factories = {
        "lgbm": lgbm_factory,
        "xgb": xgb_factory,
        "cat": cat_factory,
    }

    predictions = []
    weights = []

    for factory in model_factories.values():
        try:
            test_pred, val_rmse = train_and_refit(
                factory,
                X_fit,
                y_fit,
                X_val,
                y_val,
                X_train,
                y_train,
                X_test,
                numeric_cols,
                cat_cols,
            )
            test_pred = np.nan_to_num(test_pred, nan=0.0, posinf=0.0, neginf=0.0)
            predictions.append(test_pred)
            weights.append(1.0 / max(val_rmse, 1e-6))
        except Exception:
            continue

    if not predictions:
        try:
            preprocessor = make_preprocessor(numeric_cols, cat_cols)
            pipeline = Pipeline([
                ("prep", preprocessor),
                ("model", LGBMRegressor(n_estimators=200, random_state=42, verbose=-1))
            ])
            pipeline.fit(X_train, y_train)
            final_pred = pipeline.predict(X_test)
        except Exception:
            final_pred = np.full(len(X_test), float(y_train.mean()), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()

        final_pred = np.zeros(len(X_test), dtype=float)
        for weight, pred in zip(weights, predictions):
            final_pred += weight * pred

    final_pred = np.nan_to_num(final_pred, nan=float(y_train.mean()), posinf=0.0, neginf=0.0)

    output = pd.DataFrame({
        "id": np.arange(len(final_pred), dtype=int),
        TARGET: final_pred
    })

    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()