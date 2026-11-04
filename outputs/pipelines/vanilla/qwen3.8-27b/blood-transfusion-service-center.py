import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/blood-transfusion-service-center/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/blood-transfusion-service-center/data/test.parquet"
TARGET = "Class"
RANDOM_STATE = 42


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_models():
    return {
        "lgbm": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="logloss",
        ),
        "cat": CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=4,
            random_seed=RANDOM_STATE,
            verbose=0,
        ),
    }


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    X_train = train_df.drop(columns=[TARGET]).copy()
    y_raw = train_df[TARGET].copy()
    X_test = test_df.drop(columns=[TARGET], errors="ignore").copy()
    X_test = X_test.reindex(columns=X_train.columns)

    categorical_cols = [
        col
        for col in X_train.columns
        if pd.api.types.is_categorical_dtype(X_train[col]) or X_train[col].dtype == object
    ]
    numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

    for col in categorical_cols:
        X_train[col] = X_train[col].astype("object").where(X_train[col].notna(), "missing")
        X_test[col] = X_test[col].astype("object").where(X_test[col].notna(), "missing")

    for col in numeric_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    y_numeric = pd.to_numeric(y_raw, errors="coerce")
    if y_numeric.notna().all():
        y_values = y_numeric.astype(int)
    else:
        y_values = y_raw.astype(str)

    unique_labels = set(y_values.unique())
    if 2 in unique_labels and 1 in unique_labels:
        positive_label = 2
    elif 1 in unique_labels:
        positive_label = 1
    elif 0 in unique_labels:
        positive_label = 1
    else:
        positive_label = max(unique_labels)

    y_train = (y_values == positive_label).astype(int)

    transformers = []
    if numeric_cols:
        transformers.append(("num", "passthrough", numeric_cols))
    if categorical_cols:
        transformers.append(("cat", make_one_hot_encoder(), categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    preprocessor.fit(X_fit)
    X_fit_processed = preprocessor.transform(X_fit)
    X_val_processed = preprocessor.transform(X_val)

    model_weights = {}
    model_flips = {}

    if y_val.nunique() > 1:
        for name, model in make_models().items():
            model.fit(X_fit_processed, y_fit)
            val_proba = model.predict_proba(X_val_processed)[:, 1]
            auc = roc_auc_score(y_val, val_proba)
            flip = auc < 0.5
            model_weights[name] = max(abs(auc - 0.5), 1e-6)
            model_flips[name] = flip
    else:
        for name in make_models():
            model_weights[name] = 1.0
            model_flips[name] = False

    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    test_probabilities = []
    total_weight = 0.0

    for name, model in make_models().items():
        model.fit(X_train_processed, y_train)
        proba = model.predict_proba(X_test_processed)[:, 1]
        if model_flips.get(name, False):
            proba = 1.0 - proba
        weight = model_weights.get(name, 1.0)
        test_probabilities.append(proba * weight)
        total_weight += weight

    if total_weight > 0:
        final_proba = np.sum(test_probabilities, axis=0) / total_weight
    else:
        final_proba = np.full(len(X_test), 0.5)

    final_proba = np.clip(final_proba, 0.0, 1.0)

    predictions = pd.DataFrame(
        {
            "id": np.arange(len(final_proba), dtype=int),
            "Class": final_proba,
        }
    )
    predictions.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()