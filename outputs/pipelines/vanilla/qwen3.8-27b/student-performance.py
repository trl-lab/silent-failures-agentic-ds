import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/student-performance/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/student-performance/data/test.parquet"
TARGET = "G3"
ID_COLS = {"id", "Id", "ID", "index", "Index", "row_id", "RowId"}


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(numeric_cols, categorical_cols):
    transformers = []

    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
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
                        ("encoder", make_one_hot_encoder()),
                    ]
                ),
                categorical_cols,
            )
        )

    if not transformers:
        return ColumnTransformer(transformers=[], remainder="passthrough")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def model_factories():
    return {
        "lgbm": lambda: LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": lambda: XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
        ),
        "cat": lambda: CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_state=42,
            verbose=0,
        ),
    }


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET not in train_df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in train data.")

    train_df = train_df.dropna(subset=[TARGET]).reset_index(drop=True)
    y_full = train_df[TARGET].astype(float)

    feature_cols = [
        col
        for col in train_df.columns
        if col != TARGET
        and col not in ID_COLS
        and col in test_df.columns
        and not train_df[col].isna().all()
    ]

    if not feature_cols:
        predictions = np.full(len(test_df), float(y_full.mean()))
        pd.DataFrame({"id": np.arange(len(test_df)), TARGET: predictions}).to_csv(
            "predictions.csv", index=False
        )
        return

    numeric_cols = [
        col
        for col in feature_cols
        if pd.api.types.is_numeric_dtype(train_df[col])
        and not pd.api.types.is_bool_dtype(train_df[col])
    ]
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]

    for col in numeric_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")
        train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan)
        test_df[col] = test_df[col].replace([np.inf, -np.inf], np.nan)

    for col in categorical_cols:
        train_df[col] = train_df[col].astype("object").fillna("missing")
        test_df[col] = test_df[col].astype("object").fillna("missing")

    X_full = train_df[feature_cols]
    X_test = test_df[feature_cols]

    factories = model_factories()
    scores = {}

    if len(train_df) >= 20:
        X_train, X_val, y_train, y_val = train_test_split(
            X_full, y_full, test_size=0.2, random_state=42
        )

        for name, factory in factories.items():
            try:
                pipeline = Pipeline(
                    [
                        ("preprocessor", make_preprocessor(numeric_cols, categorical_cols)),
                        ("model", factory()),
                    ]
                )
                pipeline.fit(X_train, y_train)
                val_pred = pipeline.predict(X_val)
                scores[name] = float(np.sqrt(mean_squared_error(y_val, val_pred)))
            except Exception:
                scores[name] = np.inf
    else:
        scores = {name: 1.0 for name in factories}

    finite_scores = {
        name: score for name, score in scores.items() if np.isfinite(score) and score > 0
    }

    if not finite_scores:
        predictions = np.full(len(test_df), float(y_full.mean()))
    else:
        weights = {name: 1.0 / score for name, score in finite_scores.items()}
        total_weight = sum(weights.values())
        weights = {name: weight / total_weight for name, weight in weights.items()}

        predictions = np.zeros(len(test_df), dtype=float)

        for name, weight in weights.items():
            try:
                pipeline = Pipeline(
                    [
                        ("preprocessor", make_preprocessor(numeric_cols, categorical_cols)),
                        ("model", factories[name]()),
                    ]
                )
                pipeline.fit(X_full, y_full)
                test_pred = pipeline.predict(X_test)
                predictions += weight * np.asarray(test_pred, dtype=float)
            except Exception:
                continue

        if not np.any(predictions):
            predictions = np.full(len(test_df), float(y_full.mean()))

    predictions = np.clip(predictions, 0.0, 20.0)

    pd.DataFrame({"id": np.arange(len(test_df)), TARGET: predictions}).to_csv(
        "predictions.csv", index=False
    )


if __name__ == "__main__":
    main()