import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/connect-4/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/connect-4/data/test.parquet"
TARGET = "class"


def is_categorical_dtype(dtype):
    return isinstance(dtype, pd.CategoricalDtype)


def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str):
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df = train_df.dropna(subset=[target])

    if not is_categorical_dtype(train_df[target].dtype):
        train_df[target] = train_df[target].astype("category")

    y_labels = train_df[target].cat.categories
    y = train_df[target].cat.codes.to_numpy(dtype=int)

    feature_cols = [c for c in train_df.columns if c != target]

    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = np.nan

    categorical_cols = []
    numeric_cols = []

    for col in feature_cols:
        train_col = train_df[col]
        test_col = test_df[col]

        if is_categorical_dtype(train_col.dtype) or train_col.dtype == object:
            categorical_cols.append(col)

            train_df[col] = train_col.astype("category")
            test_df[col] = test_col.astype("category")

            if train_df[col].isna().any():
                train_df[col] = train_df[col].fillna("missing")

            if test_df[col].isna().any():
                test_df[col] = test_df[col].fillna("missing")
        else:
            numeric_cols.append(col)
            train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
            test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

    X = train_df[feature_cols]
    X_test = test_df[feature_cols]

    return X, y, X_test, categorical_cols, numeric_cols, y_labels


def build_preprocessor(categorical_cols, numeric_cols):
    transformers = []

    if categorical_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        )

    if numeric_cols:
        transformers.append(
            ("num", SimpleImputer(strategy="median"), numeric_cols)
        )

    if not transformers:
        return "passthrough"

    if len(transformers) == 1:
        return transformers[0][1]

    return ColumnTransformer(transformers=transformers)


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    X, y, X_test, categorical_cols, numeric_cols, y_labels = prepare_data(
        train_df, test_df, TARGET
    )

    preprocessor = build_preprocessor(categorical_cols, numeric_cols)

    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.1,
        random_state=42,
        stratify=stratify,
    )

    pipeline.fit(X_train, y_train)
    val_pred = pipeline.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    print(f"Validation accuracy: {val_accuracy:.4f}")

    pipeline.fit(X, y)
    pred_codes = pipeline.predict(X_test).astype(int)
    pred_labels = y_labels[pred_codes]

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            "class": pred_labels,
        }
    )
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()