import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/KDDCup09_appetency/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/KDDCup09_appetency/data/test.parquet"
TARGET = "APPETENCY"
RANDOM_STATE = 42


def convert_target(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return (numeric > 0).astype(int)

    text = series.astype(str).str.strip().str.lower()
    positive = text.isin(["1", "1.0", "true", "yes", "y", "t"])
    return positive.astype(int)


def encode_series(series: pd.Series, mapping: dict) -> pd.Series:
    encoded = series.astype("string").map(mapping)
    return encoded.fillna(-1).astype(int)


def prepare_features(train_df: pd.DataFrame, other_df: pd.DataFrame):
    feature_cols = [
        col for col in train_df.columns
        if col != TARGET and col in other_df.columns
    ]

    numeric_cols = []
    categorical_cols = []

    for col in feature_cols:
        series = train_df[col]

        if series.dtype == object or series.dtype.name == "category":
            if series.dtype.name == "category":
                categorical_cols.append(col)
            else:
                non_null = series.dropna()
                if len(non_null) == 0:
                    continue

                converted = pd.to_numeric(non_null, errors="coerce")
                if converted.notna().mean() > 0.8:
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
        else:
            numeric_cols.append(col)

    kept_numeric = []
    for col in numeric_cols:
        series = pd.to_numeric(train_df[col], errors="coerce")
        if series.notna().any() and series.nunique(dropna=True) > 1:
            kept_numeric.append(col)

    kept_categorical = []
    for col in categorical_cols:
        series = train_df[col]
        if series.notna().any() and series.nunique(dropna=True) > 1:
            kept_categorical.append(col)

    numeric_cols = kept_numeric
    categorical_cols = kept_categorical

    X_train = pd.DataFrame(index=train_df.index)
    X_other = pd.DataFrame(index=other_df.index)

    for col in numeric_cols:
        X_train[col] = pd.to_numeric(train_df[col], errors="coerce")
        X_other[col] = pd.to_numeric(other_df[col], errors="coerce")

    for col in categorical_cols:
        train_values = train_df[col].dropna()
        categories = sorted(train_values.astype(str).unique())
        mapping = {category: idx for idx, category in enumerate(categories)}

        X_train[col] = encode_series(train_df[col], mapping)
        X_other[col] = encode_series(other_df[col], mapping)

    if X_train.shape[1] == 0:
        X_train["constant"] = 0.0
        X_other["constant"] = 0.0
        categorical_cols = []

    return X_train, X_other, categorical_cols


def make_models(categorical_cols):
    models = {}

    if categorical_cols:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
            categorical_features=categorical_cols,
        )
    else:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )

    models["xgboost"] = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="auc",
        use_label_encoder=False,
    )

    models["catboost"] = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_seed=RANDOM_STATE,
        verbose=0,
    )

    return models


def predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba[:, -1]

    decision = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-decision))


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    y = convert_target(train_df[TARGET])

    if y.nunique() < 2:
        X_train, X_test, categorical_cols = prepare_features(train_df, test_df)
        model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_train, y)
        predictions = predict_proba(model, X_test)
    else:
        train_part, val_part, y_train, y_val = train_test_split(
            train_df,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        X_train, X_val, categorical_cols = prepare_features(train_part, val_part)

        validation_scores = {}
        for name, model in make_models(categorical_cols).items():
            try:
                model.fit(X_train, y_train)
                val_proba = predict_proba(model, X_val)
                validation_scores[name] = roc_auc_score(y_val, val_proba)
            except Exception:
                continue

        if validation_scores:
            selected_models = [
                name for name, score in validation_scores.items()
                if score > 0.5
            ]
            if not selected_models:
                selected_models = [max(validation_scores, key=validation_scores.get)]
        else:
            selected_models = ["lightgbm"]

        X_train_full, X_test, categorical_cols = prepare_features(train_df, test_df)

        final_predictions = []
        for name in selected_models:
            try:
                model = make_models(categorical_cols)[name]
                model.fit(X_train_full, y)
                final_predictions.append(predict_proba(model, X_test))
            except Exception:
                continue

        if not final_predictions:
            model = LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(X_train_full, y)
            final_predictions = [predict_proba(model, X_test)]

        predictions = np.mean(final_predictions, axis=0)

    output = pd.DataFrame({
        "id": np.arange(len(test_df), dtype=int),
        "APPETENCY": predictions,
    })
    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()