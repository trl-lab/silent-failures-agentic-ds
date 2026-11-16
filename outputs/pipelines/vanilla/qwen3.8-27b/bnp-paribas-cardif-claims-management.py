import numpy as np
import pandas as pd

from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/bnp-paribas-cardif-claims-management/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/bnp-paribas-cardif-claims-management/data/test.parquet"
TARGET = "target"
ID_COL = "ID"
RANDOM_STATE = 42
MAX_ONEHOT_CATEGORIES = 30
MISSING_TOKEN = "<missing>"


def clean_frame(df: pd.DataFrame, drop_target: bool = False) -> pd.DataFrame:
    df = df.copy()

    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    if drop_target and TARGET in df.columns:
        df = df.drop(columns=[TARGET])

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    categorical_cols = [
        col
        for col in df.columns
        if col not in numeric_cols and col != TARGET
    ]

    for col in categorical_cols:
        series = df[col].astype("string")
        series = series.str.strip()
        series = series.replace(
            {
                "": pd.NA,
                "None": pd.NA,
                "none": pd.NA,
                "NULL": pd.NA,
                "null": pd.NA,
                "nan": pd.NA,
                "NaN": pd.NA,
                "N/A": pd.NA,
                "NA": pd.NA,
            }
        )
        df[col] = series

    return df


class SafePreprocessor:
    def __init__(self, max_onehot_categories: int = MAX_ONEHOT_CATEGORIES):
        self.max_onehot_categories = max_onehot_categories
        self.feature_cols = []
        self.numeric_cols = []
        self.categorical_cols = []
        self.low_cardinality_cols = []
        self.high_cardinality_cols = []
        self.onehot_encoder = None
        self.frequency_maps = {}

    def _split_columns(self, df: pd.DataFrame):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [
            col for col in df.columns if col not in numeric_cols and col != TARGET
        ]
        return numeric_cols, categorical_cols

    def fit(self, X: pd.DataFrame):
        X = clean_frame(X, drop_target=True)
        self.feature_cols = X.columns.tolist()
        self.numeric_cols, self.categorical_cols = self._split_columns(X)

        self.low_cardinality_cols = []
        self.high_cardinality_cols = []

        for col in self.categorical_cols:
            nunique = X[col].dropna().nunique()
            if nunique <= self.max_onehot_categories:
                self.low_cardinality_cols.append(col)
            else:
                self.high_cardinality_cols.append(col)

        if self.low_cardinality_cols:
            self.onehot_encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                dtype=np.float32,
            )
            self.onehot_encoder.fit(X[self.low_cardinality_cols].fillna(MISSING_TOKEN))

        self.frequency_maps = {}
        for col in self.high_cardinality_cols:
            counts = X[col].fillna(MISSING_TOKEN).value_counts(dropna=False)
            self.frequency_maps[col] = counts.to_dict()

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = clean_frame(X, drop_target=True)
        X = X.reindex(columns=self.feature_cols)

        parts = []

        if self.numeric_cols:
            numeric_block = X[self.numeric_cols].astype(np.float32)
            parts.append(numeric_block.to_numpy())

        if self.low_cardinality_cols and self.onehot_encoder is not None:
            low_block = X[self.low_cardinality_cols].fillna(MISSING_TOKEN)
            parts.append(self.onehot_encoder.transform(low_block))

        if self.high_cardinality_cols:
            high_block = np.zeros((len(X), len(self.high_cardinality_cols)), dtype=np.float32)
            for i, col in enumerate(self.high_cardinality_cols):
                values = X[col].fillna(MISSING_TOKEN).astype("string")
                freq_map = self.frequency_maps[col]
                high_block[:, i] = values.map(freq_map).fillna(0.0).astype(np.float32)
            parts.append(high_block)

        if not parts:
            return np.empty((len(X), 0), dtype=np.float32)

        return np.hstack(parts)


def positive_probability(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes = np.asarray(model.classes_)
        if len(classes) == 1:
            return np.full(len(X), float(classes[0]), dtype=np.float64)
        if 1 in classes:
            idx = int(np.where(classes == 1)[0][0])
            return proba[:, idx].astype(np.float64)
        if 0 in classes:
            idx = int(np.where(classes == 0)[0][0])
            return 1.0 - proba[:, idx]
        return proba[:, -1].astype(np.float64)

    return model.predict(X).astype(np.float64)


def make_models(n_samples: int, n_features: int, pos_weight: float):
    lgb_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="logloss",
    )

    cat_model = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        subsample=0.8,
        rsm=0.8,
        random_seed=RANDOM_STATE,
        verbose=0,
        allow_writing_files=False,
    )

    return [lgb_model, xgb_model, cat_model]


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    train_df = train_df.dropna(subset=[TARGET])
    y = train_df[TARGET].astype(int)
    X_raw = train_df.drop(columns=[TARGET])

    X_fit_raw, X_val_raw, y_fit, y_val = train_test_split(
        X_raw,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    preprocessor = SafePreprocessor(max_onehot_categories=MAX_ONEHOT_CATEGORIES)
    preprocessor.fit(X_fit_raw)

    X_fit = preprocessor.transform(X_fit_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(test_df)

    pos_count = float(y_fit.sum())
    neg_count = float(len(y_fit) - pos_count)
    pos_weight = max(1.0, neg_count / max(pos_count, 1.0))

    models = make_models(len(X_fit), X_fit.shape[1], pos_weight)

    val_probs = []
    for model in models:
        model.fit(X_fit, y_fit)
        val_probs.append(positive_probability(model, X_val))

    val_prob = np.mean(val_probs, axis=0)
    val_prob = np.clip(val_prob, 1e-7, 1.0 - 1e-7)
    val_score = log_loss(y_val, val_prob, labels=[0, 1])
    print(f"Validation log_loss: {val_score:.6f}")

    preprocessor.fit(X_raw)
    X_train_full = preprocessor.transform(X_raw)
    X_test = preprocessor.transform(test_df)

    pos_count_full = float(y.sum())
    neg_count_full = float(len(y) - pos_count_full)
    pos_weight_full = max(1.0, neg_count_full / max(pos_count_full, 1.0))

    final_models = make_models(len(X_train_full), X_train_full.shape[1], pos_weight_full)
    test_probs = []
    for model in final_models:
        model.fit(X_train_full, y)
        test_probs.append(positive_probability(model, X_test))

    test_prob = np.mean(test_probs, axis=0)
    test_prob = np.clip(test_prob, 1e-7, 1.0 - 1e-7)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_prob), dtype=int),
            "target": test_prob.astype(np.float64),
        }
    )
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()