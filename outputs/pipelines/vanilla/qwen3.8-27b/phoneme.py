import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.metrics import accuracy_score

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/phoneme/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/phoneme/data/test.parquet"
TARGET = "class"
RANDOM_STATE = 42


def is_numeric_column(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=100)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, max_categories=100)


def main() -> None:
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    train_df = train_df.dropna(subset=[TARGET])

    feature_cols = [col for col in train_df.columns if col != TARGET]

    numeric_cols = [col for col in feature_cols if is_numeric_column(train_df[col])]
    categorical_cols = [col for col in feature_cols if not is_numeric_column(train_df[col])]

    for col in categorical_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].where(train_df[col].notna(), "missing").astype(str)
        if col in test_df.columns:
            test_df[col] = test_df[col].where(test_df[col].notna(), "missing").astype(str)

    X_train_full = train_df[feature_cols].copy()
    y_raw = train_df[TARGET].copy()

    y_codes, label_values = pd.factorize(y_raw, sort=False)
    y_train_full = y_codes.astype(int)

    X_test = test_df.reindex(columns=feature_cols)

    transformers = []

    if numeric_cols:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_cols,
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical_cols,
            )
        )

    if transformers:
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    else:
        preprocessor = FunctionTransformer(lambda x: x)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    models = {
        "lightgbm": LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "catboost": CatBoostClassifier(
            iterations=600,
            learning_rate=0.05,
            depth=6,
            subsample=0.9,
            random_state=RANDOM_STATE,
            verbose=0,
        ),
    }

    val_scores = {}
    val_proba = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        val_pred = pipeline.predict(X_val)
        val_proba[name] = pipeline.predict_proba(X_val)
        val_scores[name] = accuracy_score(y_val, val_pred)

    ensemble_proba = np.mean(list(val_proba.values()), axis=0)
    ensemble_pred = np.argmax(ensemble_proba, axis=1)
    val_scores["ensemble"] = accuracy_score(y_val, ensemble_pred)

    best_name = max(val_scores, key=val_scores.get)

    if best_name == "ensemble":
        fitted_models = {}
        for name, model in models.items():
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
            pipeline.fit(X_train_full, y_train_full)
            fitted_models[name] = pipeline

        test_proba = np.mean(
            [pipeline.predict_proba(X_test) for pipeline in fitted_models.values()],
            axis=0,
        )
        test_pred_codes = np.argmax(test_proba, axis=1)
    else:
        final_pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", models[best_name])]
        )
        final_pipeline.fit(X_train_full, y_train_full)
        test_pred_codes = final_pipeline.predict(X_test)

    test_pred_labels = np.asarray(label_values)[test_pred_codes]

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_pred_labels), dtype=int),
            "class": test_pred_labels,
        }
    )

    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()