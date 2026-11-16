import gc
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/forest-cover-type-prediction/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/forest-cover-type-prediction/data/test.parquet"
TARGET = "Cover_Type"
ID_COL = "Id"
RANDOM_STATE = 42


class FeaturePreprocessor:
    def __init__(self):
        self.num_cols = []
        self.cat_cols = []
        self.medians = {}
        self.mappings = {}

    def _is_categorical(self, series):
        return series.dtype == object or str(series.dtype) == "category"

    def fit(self, X):
        self.num_cols = []
        self.cat_cols = []
        self.medians = {}
        self.mappings = {}

        for col in X.columns:
            if self._is_categorical(X[col]):
                self.cat_cols.append(col)
                values = X[col].dropna().unique()
                try:
                    values = sorted(values)
                except TypeError:
                    values = list(values)
                self.mappings[col] = {value: idx for idx, value in enumerate(values)}
            else:
                self.num_cols.append(col)
                numeric = pd.to_numeric(X[col], errors="coerce")
                median = numeric.median()
                self.medians[col] = 0.0 if pd.isna(median) else float(median)

        return self

    def transform(self, X):
        X = X.copy()

        for col in self.num_cols:
            X[col] = (
                pd.to_numeric(X[col], errors="coerce")
                .fillna(self.medians[col])
                .astype(np.float32)
            )

        for col in self.cat_cols:
            mapping = self.mappings[col]
            X[col] = X[col].map(mapping).fillna(-1).astype(np.float32)

        return X


def make_model(n_estimators=2000):
    return LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    y_full = train_df[TARGET].astype(int)
    X_full = train_df.drop(columns=[TARGET, ID_COL], errors="ignore")

    X_test = test_df.drop(columns=[ID_COL], errors="ignore")
    X_test = X_test.reindex(columns=X_full.columns)

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_full,
        y_full,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_full,
    )

    preprocessor = FeaturePreprocessor().fit(X_train_raw)
    X_train = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)

    del X_train_raw, X_val_raw
    gc.collect()

    model = make_model(n_estimators=2000)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)

    best_iteration = getattr(model, "best_iteration_", None)
    if best_iteration is None or best_iteration <= 0:
        best_iteration = model.n_estimators
    best_iteration = int(best_iteration)

    del model, X_train, X_val, y_train, y_val
    gc.collect()

    final_preprocessor = FeaturePreprocessor().fit(X_full)
    X_train_full = final_preprocessor.transform(X_full)
    X_test_final = final_preprocessor.transform(X_test)

    del X_full, X_test
    gc.collect()

    final_model = make_model(n_estimators=best_iteration)
    final_model.fit(X_train_full, y_full)

    predictions = final_model.predict(X_test_final).astype(int)

    output = pd.DataFrame(
        {
            "id": np.arange(len(predictions), dtype=int),
            "Cover_Type": predictions,
        }
    )
    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()