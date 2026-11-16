import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/qsar-biodegradation/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/qsar-biodegradation/data/test.parquet"
TARGET = "Class"
RANDOM_STATE = 42


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(numeric_cols, categorical_cols):
    transformers = []

    if numeric_cols:
        transformers.append(
            (
                "num",
                SimpleImputer(strategy="median"),
                numeric_cols,
            )
        )

    if categorical_cols:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="constant", fill_value="missing"),
                ),
                ("onehot", make_one_hot_encoder()),
            ]
        )
        transformers.append(
            (
                "cat",
                categorical_pipeline,
                categorical_cols,
            )
        )

    if not transformers:
        return "passthrough"

    return ColumnTransformer(transformers=transformers, remainder="drop")


def prepare_data(train_df, test_df):
    train_df = train_df.dropna(subset=[TARGET]).copy()

    feature_cols = [col for col in train_df.columns if col != TARGET]

    X_train = train_df[feature_cols].copy()
    X_test = test_df.reindex(columns=feature_cols).copy()

    numeric_cols = []
    categorical_cols = []

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    for col in numeric_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce").astype(float)
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce").astype(float)

    for col in categorical_cols:
        X_train[col] = X_train[col].fillna("missing").astype(str)
        X_test[col] = X_test[col].fillna("missing").astype(str)

    y_raw = train_df[TARGET].astype(str)
    classes = sorted(y_raw.unique())

    if len(classes) != 2:
        raise ValueError(f"Expected a binary target, found classes: {classes}")

    positive_class = classes[-1]
    y = (y_raw == positive_class).astype(int)

    return X_train, X_test, y, numeric_cols, categorical_cols


def make_model(name):
    if name == "lgbm":
        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )

    if name == "xgb":
        return XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
        )

    if name == "catboost":
        return CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            subsample=0.9,
            colsample_bylevel=0.9,
            random_state=RANDOM_STATE,
            verbose=0,
        )

    raise ValueError(f"Unknown model: {name}")


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    X_train, X_test, y, numeric_cols, categorical_cols = prepare_data(train_df, test_df)

    if len(np.unique(y)) == 2 and y.value_counts().min() >= 2:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    else:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )

    model_names = ["lgbm", "xgb", "catboost"]

    val_probs = []
    for name in model_names:
        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(numeric_cols, categorical_cols),
                ),
                ("model", make_model(name)),
            ]
        )
        pipeline.fit(X_fit, y_fit)
        val_probs.append(pipeline.predict_proba(X_val)[:, 1])

    val_score = np.mean(val_probs, axis=0)
    if len(np.unique(y_val)) == 2:
        val_auc = roc_auc_score(y_val, val_score)
    else:
        val_auc = np.nan

    test_probs = []
    for name in model_names:
        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(numeric_cols, categorical_cols),
                ),
                ("model", make_model(name)),
            ]
        )
        pipeline.fit(X_train, y)
        test_probs.append(pipeline.predict_proba(X_test)[:, 1])

    final_probs = np.mean(test_probs, axis=0)
    final_probs = np.clip(final_probs, 0.0, 1.0)

    predictions = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            "Class": final_probs,
        }
    )

    predictions.to_csv("predictions.csv", index=False)

    if not np.isnan(val_auc):
        print(f"Validation ROC AUC: {val_auc:.4f}")


if __name__ == "__main__":
    main()