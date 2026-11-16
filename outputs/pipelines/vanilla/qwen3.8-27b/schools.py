import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/schools/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/schools/data/test.parquet"
TARGET = "Enrollment_FT_Teachers_Ratio"

DROP_COLS = [
    "OBJECTID",
    "NCES ID",
    "Name",
    "Address",
    "City",
    "Telephone",
    "Source",
    "Website",
    "Shelter ID",
    "Validation Date",
    "Zip",
    "Zip4",
    "District ID",
]

MISSING_TOKENS = ["NOT AVAILABLE", "Not Available", "N/A", "NA", ""]


def build_preprocessor(numeric_cols, categorical_cols):
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
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="__missing__"),
                        ),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )

    if not transformers:
        return None

    return ColumnTransformer(transformers=transformers, remainder="drop")


def main():
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    test = test.drop(columns=[TARGET], errors="ignore")

    y = train[TARGET]
    train = train.drop(columns=[TARGET])

    valid = y.notna() & np.isfinite(y)
    train = train.loc[valid]
    y = y.loc[valid]

    drop_cols = [c for c in DROP_COLS if c in train.columns or c in test.columns]
    train = train.drop(columns=drop_cols, errors="ignore")
    test = test.drop(columns=drop_cols, errors="ignore")

    common_cols = [c for c in train.columns if c in test.columns]
    train = train[common_cols]
    test = test[common_cols]

    common_cols = [c for c in common_cols if train[c].notna().sum() > 0]
    train = train[common_cols]
    test = test[common_cols]

    common_cols = [c for c in common_cols if train[c].nunique(dropna=True) > 1]
    train = train[common_cols]
    test = test[common_cols]

    categorical_cols = [
        c for c in train.columns if not pd.api.types.is_numeric_dtype(train[c])
    ]

    for col in categorical_cols:
        train[col] = train[col].astype("object")
        test[col] = test[col].astype("object")
        train[col] = train[col].replace(MISSING_TOKENS, np.nan)
        test[col] = test[col].replace(MISSING_TOKENS, np.nan)

    max_cardinality = 5000
    keep_categorical = []
    for col in categorical_cols:
        if train[col].nunique(dropna=True) <= max_cardinality:
            keep_categorical.append(col)

    drop_categorical = [c for c in categorical_cols if c not in keep_categorical]
    train = train.drop(columns=drop_categorical, errors="ignore")
    test = test.drop(columns=drop_categorical, errors="ignore")

    missing_frac = train.isna().mean()
    drop_missing = [c for c in train.columns if missing_frac[c] > 0.99]
    train = train.drop(columns=drop_missing, errors="ignore")
    test = test.drop(columns=drop_missing, errors="ignore")

    numeric_cols = [
        c for c in train.columns if pd.api.types.is_numeric_dtype(train[c])
    ]
    categorical_cols = [
        c for c in train.columns if not pd.api.types.is_numeric_dtype(train[c])
    ]

    if len(train) == 0:
        predictions = np.zeros(len(test), dtype=float)
    elif not numeric_cols and not categorical_cols:
        predictions = np.full(len(test), float(y.mean()), dtype=float)
    else:
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)

        if preprocessor is None:
            predictions = np.full(len(test), float(y.mean()), dtype=float)
        else:
            model = LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                num_leaves=63,
                max_depth=-1,
                min_child_samples=20,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )

            pipeline = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("model", model),
                ]
            )

            pipeline.fit(train, y)
            predictions = pipeline.predict(test)
            predictions = np.asarray(predictions, dtype=float)
            predictions = np.where(np.isfinite(predictions), predictions, float(y.mean()))

            if y.min() >= 0:
                predictions = np.clip(predictions, 0.0, None)

    output = pd.DataFrame(
        {
            "id": np.arange(len(test), dtype=int),
            TARGET: predictions,
        }
    )
    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()