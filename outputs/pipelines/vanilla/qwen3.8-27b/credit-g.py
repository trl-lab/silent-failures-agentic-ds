import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/credit-g/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/credit-g/data/test.parquet"
TARGET = "class"
RANDOM_STATE = 42


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def get_feature_columns(X: pd.DataFrame):
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols):
    transformers = []

    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical_cols,
            )
        )

    if not transformers:
        transformers.append(("passthrough", "passthrough", []))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def prepare_frames(train_df: pd.DataFrame, test_df: pd.DataFrame):
    X = train_df.drop(columns=[TARGET]).copy()
    y_raw = train_df[TARGET].astype(str).str.strip().str.lower()

    if set(y_raw.unique()) <= {"good", "bad"}:
        y = (y_raw == "good").astype(int)
        positive_label = "good"
    else:
        y = y_raw.astype(int)
        positive_label = None

    X_test = test_df.copy()
    if TARGET in X_test.columns:
        X_test = X_test.drop(columns=[TARGET])

    numeric_cols, categorical_cols = get_feature_columns(X)

    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype(object)
        if col in X_test.columns:
            X_test[col] = X_test[col].astype(object)

    for col in numeric_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        if col in X_test.columns:
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    missing_test_cols = [c for c in X.columns if c not in X_test.columns]
    for col in missing_test_cols:
        X_test[col] = np.nan

    X_test = X_test[X.columns]

    return X, y, X_test, numeric_cols, categorical_cols, positive_label


def make_models(n_estimators: int):
    return {
        "lightgbm": LGBMClassifier(
            n_estimators=n_estimators,
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
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
        ),
        "catboost": CatBoostClassifier(
            iterations=n_estimators,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            verbose=0,
        ),
    }


def predict_proba(model, X):
    proba = model.predict_proba(X)
    if proba.shape[1] == 2:
        return proba[:, 1]
    return proba[:, -1]


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    X, y, X_test, numeric_cols, categorical_cols, positive_label = prepare_frames(
        train_df, test_df
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_train_p = preprocessor.fit_transform(X_train)
    X_val_p = preprocessor.transform(X_val)

    candidate_models = make_models(700)
    val_proba = {}
    val_auc = {}

    for name, model in candidate_models.items():
        model.fit(X_train_p, y_train)
        proba = predict_proba(model, X_val_p)
        val_proba[name] = proba
        val_auc[name] = roc_auc_score(y_val, proba)

    ensemble_proba = np.mean(list(val_proba.values()), axis=0)
    val_auc["ensemble"] = roc_auc_score(y_val, ensemble_proba)

    best_strategy = max(val_auc, key=val_auc.get)

    preprocessor_final = build_preprocessor(numeric_cols, categorical_cols)
    X_full_p = preprocessor_final.fit_transform(X)
    X_test_p = preprocessor_final.transform(X_test)

    if best_strategy == "ensemble":
        final_models = {}
        for name in candidate_models:
            final_model = make_models(700)[name]
            final_model.fit(X_full_p, y)
            final_models[name] = final_model
        test_proba = np.mean(
            [predict_proba(model, X_test_p) for model in final_models.values()],
            axis=0,
        )
    else:
        final_model = make_models(700)[best_strategy]
        final_model.fit(X_full_p, y)
        test_proba = predict_proba(final_model, X_test_p)

    test_proba = np.clip(test_proba, 0.0, 1.0)

    predictions = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            "class": test_proba,
        }
    )
    predictions.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()