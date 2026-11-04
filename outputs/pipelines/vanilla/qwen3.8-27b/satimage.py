import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/satimage/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/satimage/data/test.parquet"
TARGET = "class"
RANDOM_STATE = 42


def encode_target(y):
    y = y.copy()

    if isinstance(y.dtype, pd.CategoricalDtype):
        try:
            cats = y.cat.categories
            cats_numeric = pd.to_numeric(cats, errors="raise")
            if np.all(np.isfinite(cats_numeric)):
                return y.astype(int), None
        except Exception:
            pass

    try:
        return y.astype(int), None
    except Exception:
        le = LabelEncoder()
        y_model = le.fit_transform(y.astype(str))
        return y_model, le.classes_


def align_proba(proba, model_classes, all_classes):
    if proba.shape[1] == len(all_classes) and np.array_equal(
        np.asarray(model_classes), np.asarray(all_classes)
    ):
        return proba

    aligned = np.zeros((proba.shape[0], len(all_classes)), dtype=float)
    all_classes_list = list(all_classes)

    for i, c in enumerate(model_classes):
        try:
            j = all_classes_list.index(c)
            aligned[:, j] = proba[:, i]
        except ValueError:
            pass

    return aligned


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    train_df = train_df.dropna(subset=[TARGET])

    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])

    X_train = train_df.drop(columns=[TARGET])
    y_raw = train_df[TARGET]
    y_train, label_map = encode_target(y_raw)

    common_cols = [c for c in X_train.columns if c in test_df.columns]
    X_train = X_train[common_cols]
    X_test = test_df[common_cols]

    numeric_cols = [
        c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])
    ]
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

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
                            SimpleImputer(strategy="constant", fill_value="missing"),
                        ),
                        ("encoder", OrdinalEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers, remainder="drop")

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

    models = {
        "lgbm": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
        "catboost": CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_seed=RANDOM_STATE,
            verbose=0,
        ),
    }

    all_classes = np.unique(y_train)
    val_weights = []

    for name, model in models.items():
        pipe = Pipeline([("pre", clone(preprocessor)), ("model", model)])
        pipe.fit(X_train_split, y_train_split)

        proba = pipe.predict_proba(X_val_split)
        model_classes = pipe.named_steps["model"].classes_
        proba = align_proba(proba, model_classes, all_classes)

        pred = all_classes[np.argmax(proba, axis=1)]
        acc = accuracy_score(y_val_split, pred)
        val_weights.append(acc)

    weights = np.array(val_weights, dtype=float)
    if weights.sum() <= 0:
        weights = np.ones(len(weights), dtype=float)

    weights = np.clip(weights, 1e-6, None)
    weights /= weights.sum()

    test_probs = []

    for name, model in models.items():
        pipe = Pipeline([("pre", clone(preprocessor)), ("model", model)])
        pipe.fit(X_train, y_train)

        proba = pipe.predict_proba(X_test)
        model_classes = pipe.named_steps["model"].classes_
        proba = align_proba(proba, model_classes, all_classes)

        test_probs.append(proba)

    test_probs = np.array(test_probs)
    weighted_proba = np.average(test_probs, axis=0, weights=weights)

    pred_idx = np.argmax(weighted_proba, axis=1)
    pred_labels = all_classes[pred_idx]

    if label_map is not None:
        pred_labels = label_map[pred_labels]

    output = pd.DataFrame(
        {
            "id": np.arange(len(pred_labels), dtype=int),
            "class": pred_labels,
        }
    )
    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()