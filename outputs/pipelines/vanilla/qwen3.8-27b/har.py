import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/har/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/har/data/test.parquet"
TARGET = "class"
RANDOM_STATE = 42


class FeatureTransformer:
    def __init__(self):
        self.columns_ = []
        self.numeric_cols = []
        self.categorical_cols = []
        self.medians = {}
        self.mappings = {}

    def fit(self, X):
        X = X.copy()
        self.columns_ = list(X.columns)
        self.numeric_cols = []
        self.categorical_cols = []

        for col in self.columns_:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.numeric_cols.append(col)
            else:
                self.categorical_cols.append(col)

        for col in self.numeric_cols:
            series = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            median = series.median()
            self.medians[col] = 0.0 if pd.isna(median) else float(median)

        for col in self.categorical_cols:
            values = X[col].fillna("missing").astype(str)
            categories = sorted(values.unique())
            self.mappings[col] = {category: idx for idx, category in enumerate(categories)}

        return self

    def transform(self, X):
        X = X.copy()

        for col in self.numeric_cols:
            if col in X.columns:
                series = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            else:
                series = pd.Series(np.nan, index=X.index)
            X[col] = series.fillna(self.medians[col])

        for col in self.categorical_cols:
            mapping = self.mappings[col]
            if col in X.columns:
                values = X[col].fillna("missing").astype(str)
            else:
                values = pd.Series("missing", index=X.index)
            X[col] = values.map(mapping).fillna(-1).astype(int)

        X = X.reindex(columns=self.columns_)
        return X.astype(float)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def make_models(n_classes):
    models = {}

    models["lightgbm"] = LGBMClassifier(
        n_estimators=300,
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

    if n_classes == 2:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
            objective="binary:logistic",
        )
    else:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
            objective="multi:softprob",
            num_class=n_classes,
        )

    models["catboost"] = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=RANDOM_STATE,
        verbose=0,
    )

    return models


def align_proba(proba, model_classes, target_classes):
    proba = np.asarray(proba, dtype=float)

    if list(model_classes) == list(target_classes):
        return proba

    aligned = np.zeros((proba.shape[0], len(target_classes)), dtype=float)
    for idx, cls in enumerate(model_classes):
        if cls in target_classes:
            aligned[:, target_classes.index(cls)] = proba[:, idx]

    return aligned


def sanitize_proba(proba):
    proba = np.nan_to_num(proba, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = proba.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return proba / row_sums


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    n_test = len(test_df)

    if TARGET not in train_df.columns:
        predictions = pd.DataFrame({
            "id": np.arange(n_test, dtype=int),
            "class": np.array([], dtype=object),
        })
        predictions.to_csv("predictions.csv", index=False)
        return

    train_df = train_df.dropna(subset=[TARGET])
    X_train = train_df.drop(columns=[TARGET], errors="ignore")
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=[TARGET], errors="ignore")

    if len(y_train) == 0:
        predictions = pd.DataFrame({
            "id": np.arange(n_test, dtype=int),
            "class": np.array([], dtype=object),
        })
        predictions.to_csv("predictions.csv", index=False)
        return

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    target_classes = np.arange(len(label_encoder.classes_))

    if len(target_classes) == 1:
        predictions = pd.DataFrame({
            "id": np.arange(n_test, dtype=int),
            "class": np.full(n_test, label_encoder.classes_[0]),
        })
        predictions.to_csv("predictions.csv", index=False)
        return

    try:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train_encoded,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y_train_encoded,
        )
    except ValueError:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train_encoded,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )

    transformer = FeatureTransformer().fit(X_fit)
    X_fit_processed = transformer.transform(X_fit)
    X_val_processed = transformer.transform(X_val)

    validation_models = make_models(len(target_classes))
    model_names = list(validation_models.keys())
    validation_scores = {}

    for name, model in validation_models.items():
        try:
            model.fit(X_fit_processed, y_fit)
            validation_scores[name] = accuracy_score(y_val, model.predict(X_val_processed))
        except Exception:
            validation_scores[name] = 0.0

    scores = np.array([validation_scores[name] for name in model_names], dtype=float)
    weights = scores - scores.min()
    if weights.sum() <= 0:
        weights = np.ones(len(model_names), dtype=float)
    weights = weights / weights.sum()

    transformer_final = FeatureTransformer().fit(X_train)
    X_train_processed = transformer_final.transform(X_train)
    X_test_processed = transformer_final.transform(X_test)

    final_models = make_models(len(target_classes))
    final_probas = []
    final_weights = []

    for name, model in final_models.items():
        try:
            model.fit(X_train_processed, y_train_encoded)
            proba = model.predict_proba(X_test_processed)
            proba = align_proba(proba, model.classes_, target_classes)
            proba = sanitize_proba(proba)
            final_probas.append(proba)
            final_weights.append(weights[model_names.index(name)])
        except Exception:
            continue

    if not final_probas:
        majority_class = int(np.bincount(y_train_encoded).argmax())
        predicted_labels = label_encoder.inverse_transform(
            np.full(n_test, majority_class, dtype=int)
        )
    else:
        final_weights = np.array(final_weights, dtype=float)
        if final_weights.sum() <= 0:
            final_weights = np.ones(len(final_probas), dtype=float)
        final_weights = final_weights / final_weights.sum()

        ensemble_proba = np.zeros((n_test, len(target_classes)), dtype=float)
        for weight, proba in zip(final_weights, final_probas):
            ensemble_proba += weight * proba

        ensemble_proba = sanitize_proba(ensemble_proba)
        predicted_encoded = np.argmax(ensemble_proba, axis=1)
        predicted_labels = label_encoder.inverse_transform(predicted_encoded)

    predictions = pd.DataFrame({
        "id": np.arange(n_test, dtype=int),
        "class": predicted_labels,
    })
    predictions.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()