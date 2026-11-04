import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/car-evaluation/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/car-evaluation/data/test.parquet"
TARGET = "car"


def load_data():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    train_df = train_df.dropna(subset=[TARGET])

    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])

    return train_df, test_df


def align_features(df, feature_cols):
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[feature_cols]


def clean_frame(df):
    df = df.copy()
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return df


def build_preprocessor(feature_cols, train_frame):
    numeric_cols = [
        col for col in feature_cols
        if pd.api.types.is_numeric_dtype(train_frame[col])
    ]
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]

    transformers = []

    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                numeric_cols,
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_cols,
            )
        )

    if not transformers:
        return Pipeline([("passthrough", "passthrough")])

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_model():
    return LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def main():
    train_df, test_df = load_data()

    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET].astype(str)

    feature_cols = X.columns.tolist()

    X = clean_frame(X)
    X_test = align_features(test_df, feature_cols)
    X_test = clean_frame(X_test)

    if len(X) > 1:
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        preprocessor = build_preprocessor(feature_cols, X_train)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", make_model()),
        ])

        pipeline.fit(X_train, y_train)
        val_pred = pipeline.predict(X_val)
        _ = accuracy_score(y_val, val_pred)

        pipeline.fit(X, y)
    else:
        preprocessor = build_preprocessor(feature_cols, X)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", make_model()),
        ])
        pipeline.fit(X, y)

    predictions = pipeline.predict(X_test)
    predictions = np.asarray(predictions, dtype=object)

    output_df = pd.DataFrame({
        "id": np.arange(len(predictions), dtype=int),
        "car": predictions,
    })

    output_df.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()