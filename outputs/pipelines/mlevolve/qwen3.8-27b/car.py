

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
import lightgbm as lgb

TARGET = "class"


ORDINAL_ORDERS = {
    "buying": ["low", "med", "high", "vhigh"],
    "maint": ["low", "med", "high", "vhigh"],
    "doors": ["2", "3", "4", "5more"],
    "persons": ["2", "4", "more"],
    "lug_boot": ["small", "med", "big"],
    "safety": ["low", "med", "high"],
}


def _ordinal_key(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def encode_features(X_train: pd.DataFrame, X_test: pd.DataFrame, feature_cols: list) -> tuple:
    X_train = X_train.copy()
    X_test = X_test.copy()

    for col in feature_cols:
        if col not in X_test.columns:
            X_test[col] = np.nan

    X_train = X_train[feature_cols].copy()
    X_test = X_test[feature_cols].copy()

    for col in feature_cols:
        train_values = X_train[col].dropna().map(_ordinal_key)
        train_value_set = set(train_values.unique())

        if col in ORDINAL_ORDERS:
            ordered = [_ordinal_key(value) for value in ORDINAL_ORDERS[col]]
            mapping = {value: rank for rank, value in enumerate(ordered) if value in train_value_set}
        else:
            mapping = {value: rank for rank, value in enumerate(sorted(train_values.unique()))}

        X_train[col] = X_train[col].map(lambda value: -1 if pd.isna(value) else mapping.get(_ordinal_key(value), -1)).astype(int)
        X_test[col] = X_test[col].map(lambda value: -1 if pd.isna(value) else mapping.get(_ordinal_key(value), -1)).astype(int)

    for col in ("buying", "maint", "doors", "persons", "lug_boot", "safety"):
        if col not in X_train.columns:
            X_train[col] = -1
        if col not in X_test.columns:
            X_test[col] = -1

    X_train["price_score"] = -(X_train["buying"] + X_train["maint"])
    X_train["comfort_score"] = X_train["doors"] + X_train["persons"] + X_train["lug_boot"]
    X_train["tech_score"] = X_train["safety"]
    X_train["total_score"] = X_train["price_score"] + X_train["comfort_score"] + X_train["tech_score"]
    X_train["price_comfort"] = X_train["price_score"] * X_train["comfort_score"]
    X_train["price_tech"] = X_train["price_score"] * X_train["tech_score"]
    X_train["comfort_tech"] = X_train["comfort_score"] * X_train["tech_score"]

    X_test["price_score"] = -(X_test["buying"] + X_test["maint"])
    X_test["comfort_score"] = X_test["doors"] + X_test["persons"] + X_test["lug_boot"]
    X_test["tech_score"] = X_test["safety"]
    X_test["total_score"] = X_test["price_score"] + X_test["comfort_score"] + X_test["tech_score"]
    X_test["price_comfort"] = X_test["price_score"] * X_test["comfort_score"]
    X_test["price_tech"] = X_test["price_score"] * X_test["tech_score"]
    X_test["comfort_tech"] = X_test["comfort_score"] * X_test["tech_score"]

    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    return X_train.astype(float), X_test.astype(float)


def make_models() -> list:
    models = []
    for seed in (42, 123, 456):
        models.append(
            lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=5,
                subsample=0.9,
                subsample_freq=1,
                colsample_bytree=0.9,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            )
        )
    models.append(
        HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=5,
            l2_regularization=1.0,
            random_state=42,
        )
    )
    return models


def train_ensemble(models: list, X: pd.DataFrame, y: pd.Series) -> list:
    for model in models:
        model.fit(X, y)
    return models


def predict_ensemble(models: list, X: pd.DataFrame, label_order: list) -> np.ndarray:
    label_array = np.asarray(label_order)
    averaged = np.zeros((len(X), len(label_array)), dtype=float)
    for model in models:
        proba = model.predict_proba(X)
        aligned = np.zeros((len(X), len(label_array)), dtype=float)
        for idx, cls in enumerate(model.classes_):
            if cls in label_array:
                aligned[:, np.where(label_array == cls)[0][0]] = proba[:, idx]
        averaged += aligned
    averaged /= max(len(models), 1)
    return label_array[np.argmax(averaged, axis=1)]


def main():
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    train_df = train_df.dropna(subset=[TARGET]).reset_index(drop=True)
    y = train_df[TARGET].astype(str)
    feature_cols = [col for col in train_df.columns if col != TARGET]

    try:
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            train_df, y, test_size=0.2, stratify=y, random_state=42
        )
    except ValueError:
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            train_df, y, test_size=0.2, random_state=42
        )

    X_train, X_val = encode_features(X_train_raw, X_val_raw, feature_cols)
    models = make_models()
    train_ensemble(models, X_train, y_train)
    val_label_order = sorted(y_train.unique())
    val_pred = predict_ensemble(models, X_val, val_label_order)
    score = accuracy_score(y_val, val_pred)

    X_train_full, X_test = encode_features(train_df, test_df, feature_cols)
    final_models = make_models()
    train_ensemble(final_models, X_train_full, y)
    test_label_order = sorted(y.unique())
    test_pred = predict_ensemble(final_models, X_test, test_label_order)

    os.makedirs("./submission", exist_ok=True)
    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            "class": np.asarray(test_pred, dtype=str),
        }
    )
    submission.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()
