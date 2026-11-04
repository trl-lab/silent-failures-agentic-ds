import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/internet_advertisement/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/internet_advertisement/data/test.parquet"
TARGET = "class"
OUTPUT_PATH = "predictions.csv"
RANDOM_STATE = 42
MISSING_TOKEN = "__MISSING__"


def load_data():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    test_target = None
    if TARGET in test_df.columns:
        test_target = test_df[TARGET].copy()
        test_df = test_df.drop(columns=[TARGET])

    train_df = train_df.dropna(subset=[TARGET])

    y_train_raw = train_df[TARGET].astype(str)
    X_train = train_df.drop(columns=[TARGET])

    test_df = test_df.reindex(columns=X_train.columns)

    return X_train, y_train_raw, test_df, test_target


def encode_features(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    continuous_cols = [col for col in ("height", "width", "aratio") if col in X_train.columns]

    for col in X_train.columns:
        train_series = X_train[col]
        test_series = X_test[col]

        if pd.api.types.is_numeric_dtype(train_series):
            train_series = pd.to_numeric(train_series, errors="coerce").replace([np.inf, -np.inf], np.nan)
            test_series = pd.to_numeric(test_series, errors="coerce").replace([np.inf, -np.inf], np.nan)

            if col in continuous_cols:
                train_series = train_series.replace(0, np.nan)
                test_series = test_series.replace(0, np.nan)

            X_train[col] = train_series.astype(float)
            X_test[col] = test_series.astype(float)
            continue

        train_series = (
            train_series.astype(object)
            .where(train_series.notna(), MISSING_TOKEN)
            .astype(str)
        )
        test_series = (
            test_series.astype(object)
            .where(test_series.notna(), MISSING_TOKEN)
            .astype(str)
        )

        try:
            train_numeric = pd.to_numeric(train_series, errors="raise")
            test_numeric = pd.to_numeric(test_series, errors="raise")
            X_train[col] = train_numeric.astype(float)
            X_test[col] = test_numeric.astype(float)
        except (ValueError, TypeError):
            mapping = {value: idx for idx, value in enumerate(train_series.unique())}
            X_train[col] = train_series.map(mapping).astype(float)
            X_test[col] = test_series.map(mapping).fillna(-1.0).astype(float)

    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    return X_train, X_test


def choose_positive_label(y_train_raw, test_target=None):
    labels = set(y_train_raw.astype(str).unique())
    if test_target is not None:
        labels.update(test_target.dropna().astype(str).unique())

    if "ad" in labels and "noad" in labels:
        return "noad"

    if "ad" in labels:
        return "ad"

    if "noad" in labels:
        return "noad"

    sorted_labels = sorted(labels)
    return sorted_labels[-1] if sorted_labels else "1"


def make_model():
    return LGBMClassifier(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=-1,
    )


def save_predictions(scores):
    scores = np.asarray(scores, dtype=float)
    scores = np.clip(scores, 0.0, 1.0)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(scores), dtype=int),
            "class": scores,
        }
    )
    submission.to_csv(OUTPUT_PATH, index=False)


def main():
    X_train, y_train_raw, X_test, test_target = load_data()

    positive_label = choose_positive_label(y_train_raw, test_target)
    y_train = (y_train_raw.astype(str) == positive_label).astype(int)

    if len(np.unique(y_train)) < 2:
        fallback = float(y_train.iloc[0]) if len(y_train) > 0 else 0.0
        save_predictions(np.full(len(X_test), fallback, dtype=float))
        return

    X_train, X_test = encode_features(X_train, X_test)

    if len(X_train) > 1 and y_train.value_counts().min() > 1:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y_train,
        )

        validation_model = make_model()
        validation_model.fit(X_fit, y_fit)

        if len(np.unique(y_val)) > 2:
            val_score = validation_model.predict_proba(X_val)[:, 1]
            _ = roc_auc_score(y_val, val_score)

    final_model = make_model()
    final_model.fit(X_train, y_train)

    if len(final_model.classes_) == 2:
        scores = final_model.predict_proba(X_test)[:, 1]
    else:
        scores = final_model.predict_proba(X_test).max(axis=1)

    save_predictions(scores)


if __name__ == "__main__":
    main()