import pandas as pd
import numpy as np
from pandas.api.types import is_object_dtype, is_categorical_dtype, is_bool_dtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/segment/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/segment/data/test.parquet"
TARGET = "class"
RANDOM_STATE = 42


class SafePreprocessor:
    """
    Fit on training data only, then transform train/validation/test.
    - Numeric columns: coerce to numeric, replace inf with NaN, impute with training median.
    - Categorical columns: convert to object, impute missing with 'missing',
      one-hot encode using categories seen in training plus 'missing' and 'unknown'.
    """

    def __init__(self):
        self.numeric_cols = []
        self.categorical_cols = []
        self.numeric_medians = None
        self.categorical_categories = {}

    def _is_categorical(self, series: pd.Series) -> bool:
        return is_object_dtype(series) or is_categorical_dtype(series) or is_bool_dtype(series)

    def fit(self, X: pd.DataFrame) -> "SafePreprocessor":
        X = X.copy()

        self.categorical_cols = [col for col in X.columns if self._is_categorical(X[col])]
        self.numeric_cols = [col for col in X.columns if col not in self.categorical_cols]

        for col in self.numeric_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)

        self.numeric_medians = X[self.numeric_cols].median().fillna(0.0) if self.numeric_cols else pd.Series(dtype=float)

        for col in self.categorical_cols:
            values = X[col].astype("object").where(X[col].notna(), "missing")
            categories = sorted(values.unique().tolist())
            if "missing" not in categories:
                categories.append("missing")
            if "unknown" not in categories:
                categories.append("unknown")
            self.categorical_categories[col] = categories

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        parts = []

        if self.numeric_cols:
            numeric_out = pd.DataFrame(index=X.index)
            for col in self.numeric_cols:
                if col in X.columns:
                    series = pd.to_numeric(X[col], errors="coerce")
                    series = series.replace([np.inf, -np.inf], np.nan)
                else:
                    series = pd.Series(np.nan, index=X.index)
                numeric_out[col] = series.fillna(self.numeric_medians.get(col, 0.0))
            parts.append(numeric_out)

        if self.categorical_cols:
            categorical_out = pd.DataFrame(index=X.index)
            for col in self.categorical_cols:
                if col in X.columns:
                    series = X[col].astype("object").where(X[col].notna(), "missing")
                else:
                    series = pd.Series("missing", index=X.index)

                categories = self.categorical_categories[col]
                series = series.where(series.isin(categories), "unknown")

                for category in categories:
                    categorical_out[f"{col}__{category}"] = (series == category).astype(int)
            parts.append(categorical_out)

        if not parts:
            return pd.DataFrame(index=X.index)

        return pd.concat(parts, axis=1)


def main() -> None:
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    train_df = train_df.dropna(subset=[TARGET])

    if "id" in train_df.columns:
        train_df = train_df.drop(columns=["id"])
    if "id" in test_df.columns:
        test_df = test_df.drop(columns=["id"])

    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET].astype(str)

    X_test = test_df.drop(columns=[TARGET], errors="ignore")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    preprocessor = SafePreprocessor().fit(X_train_raw)
    X_train = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)

    model = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    print(f"Validation accuracy: {val_accuracy:.4f}")

    final_preprocessor = SafePreprocessor().fit(X)
    X_full = final_preprocessor.transform(X)
    X_test_final = final_preprocessor.transform(X_test)

    final_model = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    final_model.fit(X_full, y_encoded)
    test_pred_encoded = final_model.predict(X_test_final)
    test_pred = label_encoder.inverse_transform(test_pred_encoded)

    predictions = pd.DataFrame({
        "id": np.arange(len(test_pred), dtype=int),
        "class": test_pred,
    })

    predictions.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()