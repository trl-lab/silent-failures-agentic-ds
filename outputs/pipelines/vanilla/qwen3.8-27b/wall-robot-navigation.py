import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/wall-robot-navigation/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/wall-robot-navigation/data/test.parquet"
TARGET = "class"


class FeaturePreprocessor:
    def __init__(self):
        self.numeric_cols = []
        self.categorical_cols = []
        self.medians = {}
        self.categories = {}
        self.output_cols = []

    def _is_numeric_like(self, series):
        if pd.api.types.is_numeric_dtype(series):
            return True

        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or series.dtype == "category"
        ):
            sample = series.dropna()
            if len(sample) == 0:
                return False
            sample = sample.head(1000)
            converted = pd.to_numeric(sample, errors="coerce")
            return converted.notna().mean() >= 0.9

        return False

    def fit(self, X):
        X = X.copy()
        self.numeric_cols = []
        self.categorical_cols = []
        self.medians = {}
        self.categories = {}

        for col in X.columns:
            if self._is_numeric_like(X[col]):
                self.numeric_cols.append(col)
                s = pd.to_numeric(X[col], errors="coerce")
                s = s.replace([np.inf, -np.inf], np.nan)
                med = s.median()
                self.medians[col] = 0.0 if pd.isna(med) else float(med)
            else:
                self.categorical_cols.append(col)
                s = X[col].astype(str)
                s = s.replace({"nan": "missing", "None": "missing", "": "missing"})
                cats = sorted(s.unique().tolist())
                if not cats:
                    cats = ["missing"]
                self.categories[col] = cats

        self.output_cols = list(self.numeric_cols)
        for col in self.categorical_cols:
            for cat in self.categories[col]:
                self.output_cols.append(f"cat__{col}__{cat}")

        return self

    def transform(self, X):
        X = X.copy()

        for col in self.numeric_cols + self.categorical_cols:
            if col not in X.columns:
                X[col] = np.nan if col in self.numeric_cols else "missing"

        for col in self.numeric_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            X[col] = X[col].fillna(self.medians[col])

        for col in self.categorical_cols:
            s = X[col].astype(str)
            s = s.replace({"nan": "missing", "None": "missing", "": "missing"})
            for cat in self.categories[col]:
                X[f"cat__{col}__{cat}"] = (s == cat).astype(int)
            X = X.drop(columns=[col])

        for col in self.output_cols:
            if col not in X.columns:
                X[col] = 0

        X = X[self.output_cols]
        return X.astype(float)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


def make_models():
    return {
        "lgbm": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
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
        ),
        "catboost": CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            random_state=42,
            verbose=0,
        ),
    }


def cast_to_original_dtype(values, original_series):
    values = np.asarray(values, dtype=object)
    try:
        if original_series.dtype == "category":
            return pd.Categorical(values, categories=original_series.cat.categories)
        if pd.api.types.is_integer_dtype(original_series):
            return values.astype(original_series.dtype)
        if pd.api.types.is_float_dtype(original_series):
            return values.astype(original_series.dtype)
    except Exception:
        pass
    return values


def save_predictions(values, original_series, n_rows):
    values = cast_to_original_dtype(values, original_series)
    output = pd.DataFrame(
        {
            "id": np.arange(n_rows, dtype=int),
            "class": values,
        }
    )
    output.to_csv("predictions.csv", index=False)


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET not in train_df.columns:
        raise ValueError("Target column not found in train data")

    train_df = train_df.dropna(subset=[TARGET]).reset_index(drop=True)

    if train_df.empty:
        save_predictions([], pd.Series(dtype=object), len(test_df))
        return

    y_original = train_df[TARGET].copy()
    X = train_df.drop(columns=[TARGET])

    X_test = test_df.drop(columns=[TARGET], errors="ignore")
    for col in X.columns:
        if col not in X_test.columns:
            X_test[col] = np.nan
    X_test = X_test[X.columns].reset_index(drop=True)

    y_values = y_original.astype(object)
    classes = list(pd.unique(y_values))

    if len(classes) == 1:
        save_predictions([classes[0]] * len(X_test), y_original, len(X_test))
        return

    class_to_enc = {c: i for i, c in enumerate(classes)}
    y_enc = y_values.map(class_to_enc).astype(int)

    use_validation = len(y_enc) >= 2 and len(classes) >= 2
    if use_validation:
        class_counts = y_enc.value_counts()
        stratify = y_enc if class_counts.min() >= 2 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_enc, test_size=0.2, random_state=42, stratify=stratify
        )
    else:
        X_train, X_val, y_train, y_val = X, X, y_enc, y_enc

    best_pipeline = None
    best_score = -1.0

    for _, model in make_models().items():
        pipeline = Pipeline([("preprocessor", FeaturePreprocessor()), ("model", model)])
        pipeline.fit(X_train, y_train)

        if use_validation:
            val_pred = pipeline.predict(X_val)
            score = accuracy_score(y_val, val_pred)
        else:
            score = 1.0

        if score > best_score:
            best_score = score
            best_pipeline = pipeline

    best_pipeline.fit(X, y_enc)
    test_pred_enc = best_pipeline.predict(X_test)
    test_pred_values = [classes[i] for i in test_pred_enc]

    save_predictions(test_pred_values, y_original, len(X_test))


if __name__ == "__main__":
    main()