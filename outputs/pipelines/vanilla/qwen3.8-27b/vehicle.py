import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/vehicle/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/vehicle/data/test.parquet"
TARGET = "class"
RANDOM_STATE = 42
MISSING_TOKEN = "__MISSING__"


def _make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _load_data():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])

    if TARGET in train_df.columns:
        train_df = train_df.dropna(subset=[TARGET])

    return train_df, test_df


def _prepare_data(train_df, test_df):
    X_train = train_df.drop(columns=[TARGET]).copy().reset_index(drop=True)
    y_train = train_df[TARGET].astype(str).reset_index(drop=True)

    X_test = test_df.copy()
    X_test = X_test.reindex(columns=X_train.columns).reset_index(drop=True)

    numeric_cols = []
    categorical_cols = []

    for col in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    for col in numeric_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
        X_train[col] = X_train[col].replace([np.inf, -np.inf], np.nan)
        X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan)

    for col in categorical_cols:
        X_train[col] = X_train[col].astype("object")
        X_test[col] = X_test[col].astype("object")
        X_train[col] = X_train[col].where(X_train[col].notna(), MISSING_TOKEN)
        X_test[col] = X_test[col].where(X_test[col].notna(), MISSING_TOKEN)

    label_encoder = LabelEncoder()
    y_train_int = label_encoder.fit_transform(y_train)

    return X_train, y_train_int, X_test, numeric_cols, categorical_cols, label_encoder


def _build_preprocessor(numeric_cols, categorical_cols):
    transformers = []

    if numeric_cols:
        transformers.append(
            ("num", SimpleImputer(strategy="median"), numeric_cols)
        )

    if categorical_cols:
        transformers.append(
            ("cat", _make_one_hot_encoder(), categorical_cols)
        )

    if not transformers:
        return ColumnTransformer(transformers=[], remainder="drop")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _make_models():
    return {
        "lgbm": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=5,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
        "xgb": XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="mlogloss",
        ),
        "cat": CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_seed=RANDOM_STATE,
            verbose=0,
        ),
    }


def _save_predictions(predictions):
    output = pd.DataFrame(
        {
            "id": np.arange(len(predictions), dtype=int),
            "class": np.asarray(predictions, dtype=str),
        }
    )
    output.to_csv("predictions.csv", index=False)


def main():
    train_df, test_df = _load_data()
    X_train, y_train, X_test, numeric_cols, categorical_cols, label_encoder = (
        _prepare_data(train_df, test_df)
    )

    if len(X_test) == 0:
        _save_predictions([])
        return

    if X_train.shape[1] == 0 or len(X_train) == 0:
        default_class = (
            label_encoder.classes_[np.argmax(np.bincount(y_train))]
            if len(y_train) > 0
            else "unknown"
        )
        _save_predictions([default_class] * len(X_test))
        return

    try:
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y_train,
        )
    except ValueError:
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )

    validation_probabilities = []
    validation_accuracy = {}

    for name, model in _make_models().items():
        pipeline = Pipeline(
            [
                ("preprocessor", _build_preprocessor(numeric_cols, categorical_cols)),
                ("model", model),
            ]
        )
        pipeline.fit(X_train_split, y_train_split)
        proba = pipeline.predict_proba(X_val_split)
        validation_probabilities.append(proba)
        validation_accuracy[name] = accuracy_score(
            y_val_split, np.argmax(proba, axis=1)
        )

    ensemble_proba = np.mean(validation_probabilities, axis=0)
    ensemble_accuracy = accuracy_score(
        y_val_split, np.argmax(ensemble_proba, axis=1)
    )

    best_model_name = max(validation_accuracy, key=validation_accuracy.get)
    use_ensemble = ensemble_accuracy >= validation_accuracy[best_model_name]

    if use_ensemble:
        final_pipelines = []
        for name, model in _make_models().items():
            pipeline = Pipeline(
                [
                    (
                        "preprocessor",
                        _build_preprocessor(numeric_cols, categorical_cols),
                    ),
                    ("model", model),
                ]
            )
            pipeline.fit(X_train, y_train)
            final_pipelines.append(pipeline)

        test_proba = np.mean(
            [pipeline.predict_proba(X_test) for pipeline in final_pipelines],
            axis=0,
        )
        classes = final_pipelines[0].named_steps["model"].classes_
    else:
        pipeline = Pipeline(
            [
                (
                    "preprocessor",
                    _build_preprocessor(numeric_cols, categorical_cols),
                ),
                ("model", _make_models()[best_model_name]),
            ]
        )
        pipeline.fit(X_train, y_train)
        test_proba = pipeline.predict_proba(X_test)
        classes = pipeline.named_steps["model"].classes_

    predicted_labels = classes[np.argmax(test_proba, axis=1)]
    _save_predictions(predicted_labels)


if __name__ == "__main__":
    main()