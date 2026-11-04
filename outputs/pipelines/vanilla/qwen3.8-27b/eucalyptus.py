import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/eucalyptus/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/eucalyptus/data/test.parquet"
TARGET = "Utility"


class FeatureTransformer:
    def __init__(self):
        self.numeric_cols_ = []
        self.categorical_cols_ = []
        self.medians_ = {}
        self.categories_ = {}
        self.feature_order_ = []

    @staticmethod
    def _clean_categorical_series(series):
        series = series.astype(str).str.strip()
        series = series.replace({"nan": "missing", "None": "missing", "": "missing"})
        return series

    @staticmethod
    def _safe_name(value):
        return "".join(ch if ch.isalnum() else "_" for ch in str(value))

    def fit(self, X, y=None):
        X = X.copy()

        self.numeric_cols_ = [
            col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])
        ]
        self.categorical_cols_ = [
            col for col in X.columns if col not in self.numeric_cols_
        ]

        self.medians_ = {}
        for col in self.numeric_cols_:
            values = pd.to_numeric(X[col], errors="coerce")
            values = values.replace([np.inf, -np.inf], np.nan)
            median = values.median()
            if pd.isna(median):
                median = 0.0
            self.medians_[col] = float(median)

        self.categories_ = {}
        for col in self.categorical_cols_:
            values = self._clean_categorical_series(X[col])
            self.categories_[col] = sorted(values.unique().tolist())

        self.feature_order_ = list(self.numeric_cols_)
        for col in self.categorical_cols_:
            for cat in self.categories_[col]:
                self.feature_order_.append(f"{col}__{self._safe_name(cat)}")

        return self

    def transform(self, X):
        X = X.copy()
        out = {}

        for col in self.numeric_cols_:
            if col in X.columns:
                values = pd.to_numeric(X[col], errors="coerce")
            else:
                values = pd.Series(np.nan, index=X.index)

            values = values.replace([np.inf, -np.inf], np.nan)
            values = values.fillna(self.medians_[col])
            out[col] = values.to_numpy(dtype=float)

        for col in self.categorical_cols_:
            if col in X.columns:
                values = self._clean_categorical_series(X[col])
            else:
                values = pd.Series("missing", index=X.index)

            for cat in self.categories_[col]:
                out[f"{col}__{self._safe_name(cat)}"] = (values == cat).to_numpy(dtype=int)

        if not self.feature_order_:
            return pd.DataFrame(index=X.index)

        return pd.DataFrame(out, index=X.index)[self.feature_order_]

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


def make_models():
    models = {}

    try:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=5,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception:
        pass

    try:
        models["XGBoost"] = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
    except Exception:
        pass

    try:
        models["RandomForest"] = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        )
    except Exception:
        pass

    return models


def make_validation_split(X, y):
    if len(X) < 10:
        return X, y, None, None

    try:
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except Exception:
        return train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate_model(model, X_train, y_train, X_val, y_val):
    try:
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        return float(accuracy_score(y_val, pred))
    except Exception:
        return -1.0


def save_predictions(predictions):
    predictions = np.asarray(predictions, dtype=str)
    predictions = pd.Series(predictions).str.strip().to_numpy()

    output = pd.DataFrame(
        {
            "id": np.arange(len(predictions), dtype=int),
            "Utility": predictions,
        }
    )
    output.to_csv("predictions.csv", index=False)


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])

    train_df = train_df.dropna(subset=[TARGET]).copy()

    if len(train_df) == 0:
        save_predictions(np.array(["unknown"] * len(test_df), dtype=str))
        return

    y = train_df[TARGET].astype(str).str.strip().reset_index(drop=True)
    X = train_df.drop(columns=[TARGET]).reset_index(drop=True)
    X_test = test_df.reset_index(drop=True)

    if len(X_test) == 0:
        pd.DataFrame({"id": [], "Utility": []}).to_csv("predictions.csv", index=False)
        return

    if y.nunique() <= 1:
        save_predictions(np.full(len(X_test), y.iloc[0], dtype=str))
        return

    X_train, X_val, y_train, y_val = make_validation_split(X, y)
    models = make_models()

    best_name = None
    best_score = -1.0

    if X_val is not None and len(X_val) > 0 and models:
        preprocessor = FeatureTransformer()
        X_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val)

        for name, model in models.items():
            score = evaluate_model(model, X_train_proc, y_train, X_val_proc, y_val)
            if score > best_score:
                best_score = score
                best_name = name

    if best_name is None:
        if "LightGBM" in models:
            best_name = "LightGBM"
        elif models:
            best_name = next(iter(models))

    if best_name is None:
        save_predictions(np.full(len(X_test), y.value_counts().index[0], dtype=str))
        return

    final_preprocessor = FeatureTransformer()
    X_all_proc = final_preprocessor.fit_transform(X)
    X_test_proc = final_preprocessor.transform(X_test)

    final_model = make_models()[best_name]

    try:
        final_model.fit(X_all_proc, y)
        predictions = final_model.predict(X_test_proc)
    except Exception:
        predictions = np.full(len(X_test), y.value_counts().index[0], dtype=str)

    save_predictions(predictions)


if __name__ == "__main__":
    main()