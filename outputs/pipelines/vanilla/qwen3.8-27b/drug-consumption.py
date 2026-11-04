import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/drug-consumption/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/drug-consumption/data/test.parquet"
TARGET = "cannabis"
RANDOM_STATE = 42


def make_one_hot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_ordinal():
    try:
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    except TypeError:
        return OrdinalEncoder()


def align_proba(proba, model_classes, all_classes):
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1:
        if len(model_classes) == 2:
            proba = np.column_stack((1.0 - proba, proba))
        else:
            proba = proba.reshape(-1, 1)

    out = np.zeros((proba.shape[0], len(all_classes)), dtype=float)
    all_list = list(all_classes)

    for i, cls in enumerate(model_classes):
        if cls in all_list:
            out[:, all_list.index(cls)] = proba[:, i]

    row_sums = out.sum(axis=1, keepdims=True)
    zero_rows = row_sums[:, 0] == 0
    if np.any(zero_rows):
        out[zero_rows] = 1.0 / len(all_classes)
        row_sums[zero_rows] = 1.0

    return out / row_sums


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
            tree_method="hist",
        )
    if name == "catboost":
        return CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_seed=RANDOM_STATE,
            verbose=0,
        )
    raise ValueError(f"Unknown model: {name}")


def is_categorical_series(series):
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
        or pd.api.types.is_string_dtype(series)
    )


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET not in train_df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in train data.")

    train_df = train_df.dropna(subset=[TARGET])
    train_df[TARGET] = train_df[TARGET].astype(str)

    X = train_df.drop(columns=[TARGET]).copy()
    y = train_df[TARGET].copy()

    test_X = test_df.copy()
    if TARGET in test_X.columns:
        test_X = test_X.drop(columns=[TARGET])

    id_cols = [c for c in X.columns if str(c).lower() in {"id", "passengerid"}]
    X = X.drop(columns=id_cols, errors="ignore")
    test_X = test_X.drop(columns=id_cols, errors="ignore")

    common_cols = [c for c in X.columns if c in test_X.columns]
    X = X[common_cols].copy()
    test_X = test_X[common_cols].copy()

    mask = ~X.isna().all(axis=1)
    X = X[mask].copy()
    y = y[mask].copy()

    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
        if col in test_X.columns and test_X[col].dtype == bool:
            test_X[col] = test_X[col].astype(int)

    cat_cols = [c for c in common_cols if is_categorical_series(X[c])]
    num_cols = [c for c in common_cols if c not in cat_cols]

    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        if col in test_X.columns:
            test_X[col] = pd.to_numeric(test_X[col], errors="coerce")

    X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)
    test_X[num_cols] = test_X[num_cols].replace([np.inf, -np.inf], np.nan)

    low_card_cols = [c for c in cat_cols if X[c].nunique(dropna=True) <= 20]
    high_card_cols = [c for c in cat_cols if c not in low_card_cols]

    transformers = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median", fill_value=0.0))]),
                num_cols,
            )
        )
    if low_card_cols:
        transformers.append(
            (
                "cat_low",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", make_one_hot()),
                    ]
                ),
                low_card_cols,
            )
        )
    if high_card_cols:
        transformers.append(
            (
                "cat_high",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("ordinal", make_ordinal()),
                    ]
                ),
                high_card_cols,
            )
        )

    if transformers:
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    else:
        preprocessor = "passthrough"

    le = LabelEncoder()
    y = le.fit_transform(y)
    all_classes = le.classes_

    if len(X) > 1 and len(np.unique(y)) > 1:
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
            )
        except ValueError:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE
            )
    else:
        X_train, X_val, y_train, y_val = X, X, y, y

    model_names = ["lgbm", "xgb", "catboost"]
    val_probs = []
    val_names = []

    for name in model_names:
        model = make_model(name)
        pipe = Pipeline([("prep", preprocessor), ("clf", model)])
        try:
            pipe.fit(X_train, y_train)
            proba = pipe.predict_proba(X_val)
            proba = align_proba(proba, pipe.named_steps["clf"].classes_, all_classes)
            val_probs.append(proba)
            val_names.append(name)
        except Exception:
            continue

    if val_probs:
        ensemble_proba = np.mean(val_probs, axis=0)
        scores = [(accuracy_score(y_val, np.argmax(ensemble_proba, axis=1)), "ensemble")]
        for proba, name in zip(val_probs, val_names):
            scores.append((accuracy_score(y_val, np.argmax(proba, axis=1)), name))
        best_name = max(scores, key=lambda x: x[0])[1]
    else:
        best_name = "fallback"

    def fallback_predictions():
        if len(y) > 0:
            majority_idx = int(np.bincount(y).argmax())
            majority_label = le.inverse_transform(np.array([majority_idx]))[0]
        else:
            majority_label = "unknown"
        return np.array([majority_label] * len(test_X), dtype=object)

    if best_name == "fallback":
        test_pred = fallback_predictions()
    elif best_name == "ensemble":
        final_probs = []
        for name in val_names:
            model = make_model(name)
            pipe = Pipeline([("prep", preprocessor), ("clf", model)])
            try:
                pipe.fit(X, y)
                proba = pipe.predict_proba(test_X)
                proba = align_proba(proba, pipe.named_steps["clf"].classes_, all_classes)
                final_probs.append(proba)
            except Exception:
                continue

        if final_probs:
            test_proba = np.mean(final_probs, axis=0)
            test_pred = le.inverse_transform(np.argmax(test_proba, axis=1))
        else:
            test_pred = fallback_predictions()
    else:
        model = make_model(best_name)
        pipe = Pipeline([("prep", preprocessor), ("clf", model)])
        try:
            pipe.fit(X, y)
            proba = pipe.predict_proba(test_X)
            proba = align_proba(proba, pipe.named_steps["clf"].classes_, all_classes)
            test_pred = le.inverse_transform(np.argmax(proba, axis=1))
        except Exception:
            test_pred = fallback_predictions()

    pd.DataFrame(
        {
            "id": np.arange(len(test_pred), dtype=int),
            "cannabis": test_pred,
        }
    ).to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()