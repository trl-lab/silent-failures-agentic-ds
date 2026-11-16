import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/historical-earthquake-locations/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/historical-earthquake-locations/data/test.parquet"
TARGET = "Damage Description (total)"

DROP_COLS = [
    TARGET,
    "Date (YYYY/MM/DD)",
    "Location Name",
    "Area",
    "Comments",
    "URL",
    "OBJECTID",
    "ID",
    "DRAWING_ORDER",
    "x",
    "y",
]

MISSING_TOKEN = "__MISSING__"


def clean_target(y: pd.Series) -> pd.Series:
    y = y.astype("string")
    y = y.str.strip()
    y = y.where(y.notna() & (y != ""), other=pd.NA)
    return y.dropna()


def clean_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    return df


def clean_categorical(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col].astype("string")
        s = s.str.strip()
        s = s.where(s.notna() & (s != ""), other=MISSING_TOKEN)
        df[col] = s
    return df


def encode_categoricals(train_df: pd.DataFrame, test_df: pd.DataFrame, cat_cols: list):
    train_df = train_df.copy()
    test_df = test_df.copy()

    for col in cat_cols:
        if col not in train_df.columns:
            continue

        train_values = sorted(train_df[col].dropna().unique().tolist())
        mapping = {value: idx for idx, value in enumerate(train_values)}
        unknown_code = len(train_values)

        train_df[col] = (
            train_df[col]
            .map(mapping)
            .fillna(unknown_code)
            .astype(int)
        )
        test_df[col] = (
            test_df[col]
            .map(lambda value: mapping.get(value, unknown_code))
            .astype(int)
        )

    return train_df, test_df


def make_lgbm(categorical_feature, n_samples: int) -> LGBMClassifier:
    min_child_samples = 20 if n_samples >= 100 else 1
    return LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=min_child_samples,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        categorical_feature=categorical_feature,
    )


def fallback_predict(X_train: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier

        min_leaf = 20 if len(X_train) >= 100 else 1
        model = HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=min_leaf,
            random_state=42,
        )
        model.fit(X_train, y)
        return model.predict(X_test)
    except Exception:
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline

            pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=200,
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
            pipe.fit(X_train, y)
            return pipe.predict(X_test)
        except Exception:
            majority = int(np.bincount(y).argmax())
            return np.full(len(X_test), majority, dtype=int)


def write_predictions(test_len: int, labels) -> None:
    predictions = pd.DataFrame(
        {
            "id": np.arange(test_len, dtype=int),
            TARGET: labels,
        }
    )
    predictions.to_csv("predictions.csv", index=False)


def main() -> None:
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])

    if TARGET not in train_df.columns or len(train_df) == 0:
        write_predictions(len(test_df), ["None"] * len(test_df))
        return

    y_str = clean_target(train_df[TARGET])

    if len(y_str) == 0:
        write_predictions(len(test_df), ["None"] * len(test_df))
        return

    train_df = train_df.loc[y_str.index].copy()
    train_df = train_df.drop(columns=[TARGET])

    drop_cols = [col for col in DROP_COLS if col in train_df.columns]
    train_df = train_df.drop(columns=drop_cols, errors="ignore")

    drop_cols_test = [col for col in DROP_COLS if col in test_df.columns]
    test_df = test_df.drop(columns=drop_cols_test, errors="ignore")

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    test_df = test_df.reindex(columns=train_df.columns)

    numeric_cols = [
        col for col in train_df.columns if pd.api.types.is_numeric_dtype(train_df[col])
    ]
    categorical_cols = [col for col in train_df.columns if col not in numeric_cols]

    train_df = clean_numeric(train_df, numeric_cols)
    test_df = clean_numeric(test_df, numeric_cols)

    train_df = clean_categorical(train_df, categorical_cols)
    test_df = clean_categorical(test_df, categorical_cols)

    train_df, test_df = encode_categoricals(train_df, test_df, categorical_cols)

    feature_cols = [
        col for col in train_df.columns if train_df[col].nunique(dropna=True) > 1
    ]

    classes = sorted(y_str.unique().tolist())
    class_to_code = {cls: idx for idx, cls in enumerate(classes)}
    code_to_class = {idx: cls for cls, idx in class_to_code.items()}
    y = y_str.map(class_to_code).astype(int)

    if len(feature_cols) == 0 or len(train_df) == 0:
        majority = int(np.bincount(y).argmax())
        write_predictions(len(test_df), [classes[majority]] * len(test_df))
        return

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    categorical_feature = [
        f"name:{col}" for col in categorical_cols if col in feature_cols
    ]
    if not categorical_feature:
        categorical_feature = None

    if len(X_train) >= 100 and y.nunique() > 1:
        try:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y,
            )
            val_model = make_lgbm(categorical_feature, len(X_tr))
            val_model.fit(X_tr, y_tr)
            val_pred = val_model.predict(X_val)
            print(f"Validation accuracy: {accuracy_score(y_val, val_pred):.6f}")
        except Exception:
            pass

    try:
        model = make_lgbm(categorical_feature, len(X_train))
        model.fit(X_train, y)
        pred_codes = model.predict(X_test)
    except Exception:
        pred_codes = fallback_predict(X_train, y, X_test)

    pred_labels = [code_to_class.get(int(code), classes[0]) for code in pred_codes]
    write_predictions(len(test_df), pred_labels)


if __name__ == "__main__":
    main()