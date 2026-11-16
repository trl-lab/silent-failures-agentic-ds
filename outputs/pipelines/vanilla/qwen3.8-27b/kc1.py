import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/kc1/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/kc1/data/test.parquet"
TARGET = "defects"


def make_encoder():
    try:
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    except TypeError:
        return OrdinalEncoder(handle_unknown="ignore")


def make_preprocessor(num_cols, cat_cols):
    transformers = []

    if num_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), num_cols))

    if cat_cols:
        transformers.append(("cat", make_encoder(), cat_cols))

    if not transformers:
        from sklearn.base import PassThrough
        return PassThrough()

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_models():
    return {
        "lgbm": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist",
        ),
        "cat": CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_state=42,
            verbose=0,
        ),
    }


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET not in train_df.columns:
        raise ValueError("Target column not found in train data")

    y_raw = train_df[TARGET]
    if y_raw.dtype == bool:
        y = y_raw.astype(int)
    else:
        y_str = y_raw.astype(str).str.strip().str.lower()
        mapping = {"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0}
        y = y_str.map(mapping)
        if y.isna().any():
            y = y_raw.astype(float).round().astype(int)
        y = y.astype(int)

    feature_cols = [c for c in train_df.columns if c != TARGET]
    X_train_full = train_df[feature_cols].copy()
    X_test = test_df.reindex(columns=feature_cols).copy()

    cat_cols = []
    for col in feature_cols:
        if col not in X_train_full.columns:
            continue
        dtype = X_train_full[col].dtype
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
            cat_cols.append(col)

    num_cols = [c for c in feature_cols if c not in cat_cols]

    for col in num_cols:
        if col in X_train_full.columns:
            X_train_full[col] = pd.to_numeric(X_train_full[col], errors="coerce")
        if col in X_test.columns:
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    for df in (X_train_full, X_test):
        for col in num_cols:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    for col in cat_cols:
        if col in X_train_full.columns:
            X_train_full[col] = X_train_full[col].astype(object).fillna("missing").astype(str)
        if col in X_test.columns:
            X_test[col] = X_test[col].astype(object).fillna("missing").astype(str)

    if y.nunique() < 2:
        value = float(y.iloc[0]) if len(y) > 0 else 0.5
        submission = pd.DataFrame(
            {
                "id": np.arange(len(X_test), dtype=int),
                "defects": np.full(len(X_test), value, dtype=float),
            }
        )
        submission.to_csv("predictions.csv", index=False)
        return

    stratify = y if y.value_counts().min() > 1 else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    val_scores = {}
    val_probs = {}

    if len(X_val) > 0 and y_val.nunique() > 1:
        for name, model in make_models().items():
            pipe = Pipeline(
                [
                    ("preprocessor", make_preprocessor(num_cols, cat_cols)),
                    ("model", model),
                ]
            )
            pipe.fit(X_tr, y_tr)
            prob = pipe.predict_proba(X_val)[:, 1]
            val_probs[name] = prob
            val_scores[name] = roc_auc_score(y_val, prob)

        if val_probs:
            ensemble_prob = np.mean(list(val_probs.values()), axis=0)
            val_scores["ensemble"] = roc_auc_score(y_val, ensemble_prob)

        best_name = max(val_scores, key=val_scores.get)
    else:
        best_name = "ensemble"

    if best_name == "ensemble":
        final_probs = []
        for name, model in make_models().items():
            pipe = Pipeline(
                [
                    ("preprocessor", make_preprocessor(num_cols, cat_cols)),
                    ("model", model),
                ]
            )
            pipe.fit(X_train_full, y)
            prob = pipe.predict_proba(X_test)[:, 1]
            final_probs.append(prob)
        test_prob = np.mean(final_probs, axis=0)
    else:
        pipe = Pipeline(
            [
                ("preprocessor", make_preprocessor(num_cols, cat_cols)),
                ("model", make_models()[best_name]),
            ]
        )
        pipe.fit(X_train_full, y)
        test_prob = pipe.predict_proba(X_test)[:, 1]

    test_prob = np.clip(np.asarray(test_prob, dtype=float), 0.0, 1.0)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_prob), dtype=int),
            "defects": test_prob,
        }
    )
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()