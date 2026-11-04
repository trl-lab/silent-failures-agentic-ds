import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/ramen-ratings/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/ramen-ratings/data/test.parquet"
TARGET = "Stars"


def clean_text_series(s: pd.Series) -> pd.Series:
    s = s.fillna("missing").astype(str).str.strip()
    s = s.replace({"": "missing", "nan": "missing", "None": "missing"})
    return s


def prepare_target(df: pd.DataFrame) -> pd.Series:
    y = df[TARGET].copy()
    y = y[~y.isna()]

    # If the target is numeric-like, keep it numeric; otherwise treat it as a string label.
    numeric = pd.to_numeric(y, errors="coerce")
    if numeric.notna().all():
        y = numeric
        if (y == y.astype("int64")).all():
            y = y.astype("int64")
    else:
        y = y.astype(str).str.strip()

    return y


class FeatureBuilder:
    def __init__(self, low_cardinality_threshold: int = 80, max_features: int = 12000):
        self.low_cardinality_threshold = low_cardinality_threshold
        self.max_features = max_features
        self.fitted = False
        self.ohe_cols = []
        self.freq_cols = []
        self.text_cols = []
        self.ohe = None
        self.freq_maps = {}
        self.tfidf_word = None
        self.tfidf_char = None

    def _is_text_column(self, col: str, s: pd.Series) -> bool:
        if col.lower() in {"variety", "name", "description", "text", "flavor", "flavour"}:
            return True
        return s.str.len().mean() > 12

    def fit(self, X: pd.DataFrame) -> "FeatureBuilder":
        X = X.copy()
        self.columns = list(X.columns)

        for col in self.columns:
            s = clean_text_series(X[col])
            X[col] = s
            nunique = s.nunique()

            if nunique <= self.low_cardinality_threshold:
                self.ohe_cols.append(col)
            else:
                self.freq_cols.append(col)
                if self._is_text_column(col, s):
                    self.text_cols.append(col)

        if "Variety" in X.columns and "Variety" not in self.text_cols:
            self.text_cols.append("Variety")

        if self.ohe_cols:
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self.ohe.fit(X[self.ohe_cols])
        else:
            self.ohe = None

        self.freq_maps = {}
        for col in self.freq_cols:
            counts = X[col].value_counts()
            self.freq_maps[col] = counts.to_dict()

        if self.text_cols:
            texts = X[self.text_cols].apply(lambda row: " ".join(row.values), axis=1)
            self.tfidf_word = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
                token_pattern=r"(?u)\b\w+\b",
            )
            self.tfidf_char = TfidfVectorizer(
                max_features=self.max_features,
                analyzer="char_wb",
                ngram_range=(2, 4),
                min_df=1,
                sublinear_tf=True,
            )
            self.tfidf_word.fit(texts)
            self.tfidf_char.fit(texts)
        else:
            self.tfidf_word = None
            self.tfidf_char = None

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("FeatureBuilder must be fitted before transform.")

        X = X.copy()
        for col in self.columns:
            X[col] = clean_text_series(X[col])

        parts = []

        if self.ohe is not None and self.ohe_cols:
            parts.append(self.ohe.transform(X[self.ohe_cols]))

        for col in self.freq_cols:
            freq = np.array(
                [self.freq_maps[col].get(str(v), 0) for v in X[col]],
                dtype=np.float32,
            ).reshape(-1, 1)
            parts.append(freq)

        if self.text_cols:
            texts = X[self.text_cols].apply(lambda row: " ".join(row.values), axis=1)
            parts.append(self.tfidf_word.transform(texts).toarray().astype(np.float32))
            parts.append(self.tfidf_char.transform(texts).toarray().astype(np.float32))

        if not parts:
            return np.ones((len(X), 1), dtype=np.float32)

        X_out = np.hstack(parts)
        if X_out.shape[1] == 0:
            X_out = np.ones((len(X), 1), dtype=np.float32)

        return X_out

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)


def make_models() -> dict:
    return {
        "lgbm": LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=5,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
        ),
        "catboost": CatBoostClassifier(
            iterations=600,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=0,
            allow_writing_files=False,
        ),
    }


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    feature_cols = [c for c in train_df.columns if c != TARGET]

    if len(train_df) == 0 or len(feature_cols) == 0:
        fallback = "missing"
        if TARGET in train_df.columns and len(train_df) > 0:
            fallback = train_df[TARGET].dropna().iloc[0]
        out = pd.DataFrame({"id": range(len(test_df)), "Stars": [fallback] * len(test_df)})
        out.to_csv("predictions.csv", index=False)
        return

    y_raw = prepare_target(train_df)
    train_df = train_df.loc[y_raw.index].copy()
    y = y_raw.values

    for col in feature_cols:
        train_df[col] = clean_text_series(train_df[col])
        test_df[col] = clean_text_series(test_df[col])

    X_full = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    best_model_name = "lgbm"

    if len(X_full) >= 10 and len(np.unique(y_enc)) >= 2:
        try:
            X_fit, X_val, y_fit, y_val = train_test_split(
                X_full,
                y_enc,
                test_size=0.2,
                random_state=42,
                stratify=y_enc,
            )
        except ValueError:
            X_fit, X_val, y_fit, y_val = train_test_split(
                X_full,
                y_enc,
                test_size=0.2,
                random_state=42,
            )

        builder = FeatureBuilder()
        X_fit_m = builder.fit_transform(X_fit)
        X_val_m = builder.transform(X_val)

        best_acc = -1.0
        for name, model in make_models().items():
            try:
                model.fit(X_fit_m, y_fit)
                pred = model.predict(X_val_m)
                acc = accuracy_score(y_val, pred)
                if acc > best_acc:
                    best_acc = acc
                    best_model_name = name
            except Exception:
                continue

    builder = FeatureBuilder()
    X_full_m = builder.fit_transform(X_full)
    X_test_m = builder.transform(X_test)

    final_pred_enc = None
    for name in [best_model_name] + [n for n in make_models().keys() if n != best_model_name]:
        try:
            model = make_models()[name]
            model.fit(X_full_m, y_enc)
            final_pred_enc = model.predict(X_test_m)
            break
        except Exception:
            continue

    if final_pred_enc is None:
        mode_label = label_encoder.inverse_transform([np.bincount(y_enc).argmax()])[0]
        final_pred = np.full(len(X_test), mode_label)
    else:
        final_pred = label_encoder.inverse_transform(final_pred_enc)

    out = pd.DataFrame({"id": range(len(test_df)), "Stars": final_pred})
    out.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()