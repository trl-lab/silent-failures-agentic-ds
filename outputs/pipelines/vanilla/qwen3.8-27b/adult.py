import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/adult/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/adult/data/test.parquet"
TARGET = "income"

KNOWN_NUMERIC = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]


class Preprocessor:
    def __init__(self, numeric_cols, categorical_cols):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.numeric_medians = None
        self.categories = None

    def _prepare(self, X):
        X = X.copy()
        for col in self.numeric_cols + self.categorical_cols:
            if col not in X.columns:
                X[col] = np.nan
        return X

    @staticmethod
    def _clean_categorical(s):
        s = s.where(s.notna(), "missing")
        s = s.astype(str).str.strip()
        s = s.replace(
            {
                "?": "missing",
                "": "missing",
                "nan": "missing",
                "NaN": "missing",
                "None": "missing",
            }
        )
        return s

    def fit(self, X, y=None):
        X = self._prepare(X)

        numeric = X[self.numeric_cols].apply(pd.to_numeric, errors="coerce")
        self.numeric_medians = numeric.median().fillna(0)

        self.categories = {}
        for col in self.categorical_cols:
            cleaned = self._clean_categorical(X[col])
            self.categories[col] = sorted(cleaned.unique().tolist())

        return self

    def transform(self, X):
        X = self._prepare(X)
        out = pd.DataFrame(index=X.index)

        for col in self.numeric_cols:
            out[col] = pd.to_numeric(X[col], errors="coerce").fillna(self.numeric_medians[col])

        for col in self.categorical_cols:
            cleaned = self._clean_categorical(X[col])
            for cat in self.categories.get(col, []):
                feature_name = f"{col}_{cat}".replace(" ", "_")
                out[feature_name] = (cleaned == cat).astype(int)

        return out

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


def make_models():
    return {
        "lgbm": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        ),
        "cat": CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=0,
        ),
    }


def align_proba(proba, model_classes, all_classes):
    if list(model_classes) == list(all_classes):
        return proba

    out = np.zeros((proba.shape[0], len(all_classes)), dtype=float)
    all_classes = list(all_classes)

    for i, cls in enumerate(model_classes):
        if cls in all_classes:
            out[:, all_classes.index(cls)] = proba[:, i]

    return out


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    features = [c for c in train_df.columns if c != TARGET]

    numeric_cols = [
        c
        for c in features
        if c in KNOWN_NUMERIC or pd.api.types.is_numeric_dtype(train_df[c])
    ]
    categorical_cols = [c for c in features if c not in numeric_cols]

    y = train_df[TARGET].astype(str).str.strip()
    valid = y.notna() & (y != "") & (y != "?")

    X = train_df.loc[valid, features].copy()
    y = y.loc[valid].copy()

    if len(y) == 0:
        predictions = pd.DataFrame(
            {
                "id": np.arange(len(test_df)),
                "income": np.array(["<=50K." for _ in range(len(test_df))]),
            }
        )
        predictions.to_csv("predictions.csv", index=False)
        return

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    all_classes = np.arange(len(label_encoder.classes_))

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    preprocessor = Preprocessor(numeric_cols, categorical_cols)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    models = make_models()
    val_probs = []
    model_scores = {}

    for name, model in models.items():
        model.fit(X_train_processed, y_train)
        proba = model.predict_proba(X_val_processed)
        val_probs.append(align_proba(proba, model.classes_, all_classes))
        model_scores[name] = accuracy_score(y_val, model.predict(X_val_processed))

    ensemble_proba = np.mean(val_probs, axis=0)
    ensemble_pred = np.argmax(ensemble_proba, axis=1)
    ensemble_score = accuracy_score(y_val, ensemble_pred)

    best_single_name = max(model_scores, key=model_scores.get)
    use_ensemble = ensemble_score >= model_scores[best_single_name]

    preprocessor_final = Preprocessor(numeric_cols, categorical_cols)
    X_train_final = preprocessor_final.fit_transform(X)
    X_test_final = preprocessor_final.transform(test_df[features])

    if use_ensemble:
        final_probs = []
        for model in models.values():
            model.fit(X_train_final, y_encoded)
            proba = model.predict_proba(X_test_final)
            final_probs.append(align_proba(proba, model.classes_, all_classes))

        final_proba = np.mean(final_probs, axis=0)
        pred_encoded = np.argmax(final_proba, axis=1)
    else:
        models[best_single_name].fit(X_train_final, y_encoded)
        pred_encoded = models[best_single_name].predict(X_test_final)

    pred_labels = label_encoder.inverse_transform(pred_encoded)

    predictions = pd.DataFrame(
        {
            "id": np.arange(len(test_df)),
            "income": pred_labels,
        }
    )
    predictions.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()