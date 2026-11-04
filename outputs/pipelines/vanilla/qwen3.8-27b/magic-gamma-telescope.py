import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/magic-gamma-telescope/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/magic-gamma-telescope/data/test.parquet"
TARGET = "class"
POSITIVE_CLASS = "g"
RANDOM_STATE = 42


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(numeric_cols, categorical_cols):
    transformers = []

    if numeric_cols:
        transformers.append(
            ("num", SimpleImputer(strategy="median"), numeric_cols)
        )

    if categorical_cols:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", make_one_hot_encoder()),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    if not transformers:
        return "passthrough"

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_models():
    return {
        "lightgbm": LGBMClassifier(
            n_estimators=700,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=700,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
        ),
        "catboost": CatBoostClassifier(
            iterations=700,
            learning_rate=0.03,
            depth=6,
            random_seed=RANDOM_STATE,
            verbose=0,
            thread_count=-1,
        ),
    }


def fit_models(X, y, numeric_cols, categorical_cols):
    preprocessor = make_preprocessor(numeric_cols, categorical_cols)
    fitted = {}

    for name, model in make_models().items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X, y)
        fitted[name] = pipeline

    return fitted


def predict_positive_probability(pipelines, X):
    probabilities = []

    for pipeline in pipelines.values():
        proba = pipeline.predict_proba(X)
        if proba.shape[1] == 2:
            probabilities.append(proba[:, 1])
        else:
            probabilities.append(np.full(X.shape[0], 0.5))

    return np.mean(probabilities, axis=0)


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET not in train_df.columns:
        predictions = pd.DataFrame({"id": range(len(test_df)), "class": 0.5})
        predictions.to_csv("predictions.csv", index=False)
        return

    train_df = train_df.copy()
    train_df[TARGET] = train_df[TARGET].astype(str).str.strip().str.lower()
    train_df = train_df.dropna(subset=[TARGET])

    feature_cols = [col for col in train_df.columns if col != TARGET]

    if not feature_cols:
        predictions = pd.DataFrame({"id": range(len(test_df)), "class": 0.5})
        predictions.to_csv("predictions.csv", index=False)
        return

    X_all = train_df[feature_cols].copy()
    y_all = (train_df[TARGET] == POSITIVE_CLASS).astype(int)

    if y_all.nunique() < 2:
        score = float(y_all.iloc[0]) if len(y_all) else 0.5
        predictions = pd.DataFrame({"id": range(len(test_df)), "class": score})
        predictions.to_csv("predictions.csv", index=False)
        return

    numeric_cols = [
        col
        for col in feature_cols
        if pd.api.types.is_numeric_dtype(X_all[col])
    ]
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]

    for col in categorical_cols:
        X_all[col] = X_all[col].astype("object")

    X_test = test_df.reindex(columns=feature_cols)
    for col in categorical_cols:
        if col in X_test.columns:
            X_test[col] = X_test[col].astype("object")

    X_train, X_val, y_train, y_val = train_test_split(
        X_all,
        y_all,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )

    val_pipelines = fit_models(X_train, y_train, numeric_cols, categorical_cols)
    val_score = predict_positive_probability(val_pipelines, X_val)
    val_auc = roc_auc_score(y_val, val_score)

    final_pipelines = fit_models(X_all, y_all, numeric_cols, categorical_cols)
    test_score = predict_positive_probability(final_pipelines, X_test)
    test_score = np.clip(test_score, 0.0, 1.0)

    predictions = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            "class": test_score,
        }
    )
    predictions.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()