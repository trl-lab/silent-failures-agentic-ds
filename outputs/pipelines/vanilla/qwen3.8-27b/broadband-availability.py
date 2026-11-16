import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TARGET = "Population for whom broadband available, 2019 (%)"
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/broadband-availability/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/broadband-availability/data/test.parquet"
DROP_COLS = {"GEO_ID", "NAME", "County", "FIPS_County"}


def sanitize_feature_name(name):
    s = re.sub(r"[^0-9a-zA-Z_]", "_", str(name))
    if not s or s[0].isdigit():
        s = "f_" + s
    return s


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


class Preprocessor:
    def __init__(self, drop_cols):
        self.drop_cols = set(drop_cols)
        self.feature_cols = []
        self.sanitized_cols = []
        self.name_map = {}
        self.numeric_cols = []
        self.categorical_cols = []
        self.medians = {}
        self.categories = {}

    def fit(self, X):
        self.feature_cols = [c for c in X.columns if c not in self.drop_cols]

        self.name_map = {}
        seen = {}
        for col in self.feature_cols:
            base = sanitize_feature_name(col)
            if base in seen:
                seen[base] += 1
                self.name_map[col] = f"{base}_{seen[base]}"
            else:
                seen[base] = 0
                self.name_map[col] = base

        self.sanitized_cols = [self.name_map[col] for col in self.feature_cols]

        self.numeric_cols = []
        self.categorical_cols = []
        for col in self.feature_cols:
            if is_numeric_dtype(X[col]):
                self.numeric_cols.append(col)
            else:
                self.categorical_cols.append(col)

        self.medians = {}
        for col in self.numeric_cols:
            vals = pd.to_numeric(X[col], errors="coerce")
            vals = vals.replace([np.inf, -np.inf], np.nan)
            med = vals.median()
            if pd.isna(med):
                med = 0.0
            self.medians[col] = float(med)

        self.categories = {}
        for col in self.categorical_cols:
            vals = X[col].astype(object).fillna("missing").astype(str)
            cats = sorted(vals.unique().tolist(), key=str)
            self.categories[col] = {v: i for i, v in enumerate(cats)}

        return self

    def transform(self, X):
        X = X[self.feature_cols].copy()

        for col in self.numeric_cols:
            vals = pd.to_numeric(X[col], errors="coerce")
            vals = vals.replace([np.inf, -np.inf], np.nan)
            X[col] = vals.fillna(self.medians[col])

        for col in self.categorical_cols:
            vals = X[col].astype(object).fillna("missing").astype(str)
            X[col] = vals.map(self.categories[col]).fillna(-1).astype(int)

        X = X.rename(columns=self.name_map)
        return X[self.sanitized_cols]

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def make_models():
    return {
        "lgbm": LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": XGBRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        ),
        "catboost": CatBoostRegressor(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=0,
        ),
    }


def save_predictions(test_df, predictions):
    predictions = np.asarray(predictions, dtype=float).ravel()
    predictions = np.nan_to_num(predictions, nan=0.0, posinf=100.0, neginf=0.0)
    predictions = np.clip(predictions, 0.0, 100.0)

    output = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            TARGET: predictions,
        }
    )
    output.to_csv("predictions.csv", index=False)


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    train_df[TARGET] = pd.to_numeric(train_df[TARGET], errors="coerce")
    train_df = train_df.dropna(subset=[TARGET])

    if train_df.empty:
        save_predictions(test_df, np.zeros(len(test_df), dtype=float))
        return

    feature_cols = [
        col
        for col in train_df.columns
        if col != TARGET and col not in DROP_COLS and col in test_df.columns
    ]

    if not feature_cols:
        fallback = float(train_df[TARGET].mean())
        save_predictions(test_df, np.full(len(test_df), fallback, dtype=float))
        return

    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET].copy()
    X_test = test_df[feature_cols].copy()

    if len(X_train) < 2:
        X_tr, X_val, y_tr, y_val = X_train, X_train, y_train, y_train
    else:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

    preprocessor = Preprocessor(DROP_COLS)
    X_tr_proc = preprocessor.fit_transform(X_tr)
    X_val_proc = preprocessor.transform(X_val)

    models = make_models()
    val_predictions = {}

    for name, model in models.items():
        model.fit(X_tr_proc, y_tr)
        val_predictions[name] = model.predict(X_val_proc)

    ensemble_val = np.mean(list(val_predictions.values()), axis=0)

    scores = {name: rmse(y_val, pred) for name, pred in val_predictions.items()}
    scores["ensemble"] = rmse(y_val, ensemble_val)
    scores = {name: (score if np.isfinite(score) else np.inf) for name, score in scores.items()}

    best_name = min(scores, key=scores.get)

    full_preprocessor = Preprocessor(DROP_COLS)
    X_train_proc = full_preprocessor.fit_transform(X_train)
    X_test_proc = full_preprocessor.transform(X_test)

    if best_name == "ensemble":
        final_models = make_models()
        for model in final_models.values():
            model.fit(X_train_proc, y_train)
        test_predictions = np.mean(
            [model.predict(X_test_proc) for model in final_models.values()],
            axis=0,
        )
    else:
        final_model = make_models()[best_name]
        final_model.fit(X_train_proc, y_train)
        test_predictions = final_model.predict(X_test_proc)

    save_predictions(test_df, test_predictions)


if __name__ == "__main__":
    main()