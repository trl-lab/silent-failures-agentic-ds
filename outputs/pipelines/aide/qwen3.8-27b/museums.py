import os
import re
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

TARGET = "REVENUE15"
SEED = 42
NUM_BOOST_ROUND = 5000
EARLY_STOPPING_ROUNDS = 50

os.makedirs("./working", exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

if TARGET in test.columns:
    test = test.drop(columns=[TARGET])

y = pd.to_numeric(train[TARGET], errors="coerce").replace([np.inf, -np.inf], np.nan)
X = train.drop(columns=[TARGET])
X = X.loc[y.index]
y = y.dropna()
X = X.loc[y.index]


class Preprocessor:
    def __init__(self):
        self.feature_cols = []
        self.categorical_cols = []
        self.numeric_cols = []
        self.categorical_mappings = {}

    def _is_id_like(self, col, nunique, n):
        name = str(col).lower()
        if name in {
            "id",
            "index",
            "row_id",
            "rowid",
            "unique_id",
            "sample_id",
            "museum_id",
            "museumid",
        }:
            return True
        if re.search(r"(^|_)id($|_)", name) and nunique > max(10, 0.5 * n):
            return True
        return False

    def fit(self, X_train):
        X = X_train.copy()
        n = len(X)

        # Convert datetime columns to year to keep them numeric and simple.
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                X[col] = X[col].dt.year

        drop_cols = set()

        for col in X.columns:
            nunique = X[col].nunique(dropna=True)
            if self._is_id_like(col, nunique, n):
                drop_cols.add(col)

        for col in X.columns:
            if col in drop_cols:
                continue
            if X[col].isna().all():
                drop_cols.add(col)

        for col in X.columns:
            if col in drop_cols:
                continue
            if X[col].nunique(dropna=True) <= 1:
                drop_cols.add(col)

        # Drop near-unique object/category columns, which are often identifiers/free text.
        for col in X.columns:
            if col in drop_cols:
                continue
            if X[col].dtype == object or str(X[col].dtype).startswith("category"):
                nunique = X[col].nunique(dropna=True)
                if nunique > max(50, 0.95 * n):
                    drop_cols.add(col)

        self.feature_cols = [col for col in X.columns if col not in drop_cols]
        self.categorical_cols = []
        self.numeric_cols = []

        for col in self.feature_cols:
            if X[col].dtype == object or str(X[col].dtype).startswith("category"):
                self.categorical_cols.append(col)
            else:
                self.numeric_cols.append(col)

        for col in self.categorical_cols:
            values = X[col].dropna().astype(str).unique()
            self.categorical_mappings[col] = {
                str(value): idx for idx, value in enumerate(sorted(values))
            }

        return self

    def transform(self, X):
        X = X.copy()

        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = np.nan

        X = X[self.feature_cols]

        for col in self.numeric_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)

        for col in self.categorical_cols:
            mapping = self.categorical_mappings.get(col, {})
            if not mapping:
                X[col] = -1
            else:
                X[col] = (
                    X[col]
                    .map(lambda v: mapping.get(str(v), -1) if pd.notna(v) else -1)
                    .astype("int32")
                )

        return X


def make_params():
    return {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "lambda_l2": 1.0,
        "seed": SEED,
        "verbosity": -1,
    }


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


params = make_params()
callbacks = [
    lgb.early_stopping(EARLY_STOPPING_ROUNDS),
    lgb.log_evaluation(-1),
]

cv_scores = []
best_iterations = []

if len(y) >= 2:
    n_splits = min(5, len(y))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for train_idx, valid_idx in kf.split(X):
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_va = X.iloc[valid_idx]
        y_va = y.iloc[valid_idx]

        prep = Preprocessor().fit(X_tr)
        X_tr_p = prep.transform(X_tr)
        X_va_p = prep.transform(X_va)

        if X_tr_p.shape[1] == 0:
            pred = np.full(len(y_va), y_tr.mean())
            cv_scores.append(rmse(y_va, pred))
            best_iterations.append(1)
            continue

        dtrain = lgb.Dataset(
            X_tr_p,
            label=y_tr,
            categorical_feature=prep.categorical_cols,
        )
        dvalid = lgb.Dataset(
            X_va_p,
            label=y_va,
            reference=dtrain,
            categorical_feature=prep.categorical_cols,
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=callbacks,
        )

        best_iter = max(1, int(model.best_iteration))
        best_iterations.append(best_iter)

        pred = model.predict(X_va_p, num_iteration=best_iter)
        pred = np.maximum(pred, y_tr.min())
        cv_scores.append(rmse(y_va, pred))

cv_rmse = float(np.mean(cv_scores)) if cv_scores else float("nan")
print(f"CV RMSE: {cv_rmse:.6f}")

prep_final = Preprocessor().fit(X)
X_train_final = prep_final.transform(X)
X_test_final = prep_final.transform(test)

if X_train_final.shape[1] == 0:
    test_pred = np.full(len(test), y.mean())
else:
    final_rounds = int(np.mean(best_iterations)) if best_iterations else 200
    final_rounds = max(1, min(final_rounds, NUM_BOOST_ROUND))

    dtrain_final = lgb.Dataset(
        X_train_final,
        label=y,
        categorical_feature=prep_final.categorical_cols,
    )

    final_model = lgb.train(
        params,
        dtrain_final,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_pred = final_model.predict(X_test_final, num_iteration=final_rounds)
    test_pred = np.maximum(test_pred, y.min())

submission = pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        TARGET: test_pred.astype(float),
    }
)

submission.to_csv("./working/submission.csv", index=False)
