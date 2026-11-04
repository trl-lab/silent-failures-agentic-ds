import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/industry-payments-entity/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/industry-payments-entity/data/test.parquet"
TARGET = "entityPayment"
RANDOM_STATE = 42
SMOOTHING = 20.0


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def clean_frame(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) == "string":
            df[col] = df[col].astype("string").fillna("missing")
    return df


class FeatureBuilder:
    def __init__(self, numeric_cols, categorical_cols):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.interaction_specs = [
            ("reportingCompany", "paymentType"),
            ("entityCountry", "paymentType"),
            ("entityName", "paymentType"),
            ("reportingCompany", "entityCountry"),
            ("reportingCompany", "entityCountry", "paymentType"),
        ]
        self.interactions = []
        self.interaction_map = {}
        self.cat_features = []
        self.medians = {}
        self.freq_maps = {}
        self.stats = {}
        self.global_mean = 0.0

    def _init_interactions(self, df):
        self.interactions = []
        self.interaction_map = {}
        for spec in self.interaction_specs:
            if all(col in df.columns for col in spec):
                name = "_".join(spec)
                self.interactions.append((name, spec))
                self.interaction_map[name] = spec
        self.cat_features = list(self.categorical_cols) + [name for name, _ in self.interactions]

    def _feature_values(self, df, name, cache=None):
        if cache is not None and name in cache:
            return cache[name]

        if name in self.categorical_cols:
            values = df[name]
        else:
            spec = self.interaction_map[name]
            values = df[list(spec)].astype(str).agg("|".join, axis=1)

        if cache is not None:
            cache[name] = values
        return values

    def fit(self, df):
        self._init_interactions(df)
        self.global_mean = float(df[TARGET].mean())

        for col in self.numeric_cols:
            values = pd.to_numeric(df[col], errors="coerce")
            median = values.median()
            self.medians[col] = 0.0 if pd.isna(median) else float(median)

        cache = {}
        for name in self.cat_features:
            values = self._feature_values(df, name, cache)
            self.freq_maps[name] = values.value_counts().to_dict()

            grouped = (
                values.to_frame("category")
                .assign(y=df[TARGET].to_numpy(dtype=float))
                .groupby("category")["y"]
                .agg(["sum", "count"])
            )
            self.stats[name] = grouped

        return self

    def transform(self, df, oof=False):
        out = pd.DataFrame(index=df.index)

        for col in self.numeric_cols:
            out[col] = pd.to_numeric(df[col], errors="coerce").fillna(self.medians.get(col, 0.0))

        cache = {}
        for name in self.cat_features:
            values = self._feature_values(df, name, cache)

            out[f"{name}_freq"] = (
                values.map(self.freq_maps.get(name, {}))
                .fillna(0.0)
                .astype(float)
            )

            stats = self.stats[name]
            if oof:
                y = df[TARGET].to_numpy(dtype=float)
                sums = stats["sum"].reindex(values).to_numpy(dtype=float)
                counts = stats["count"].reindex(values).to_numpy(dtype=float)
                sums = np.nan_to_num(sums, nan=0.0)
                counts = np.nan_to_num(counts, nan=0.0)

                denom = counts - 1.0 + SMOOTHING
                denom = np.where(denom > 0.0, denom, SMOOTHING)
                numer = sums - y + SMOOTHING * self.global_mean
                out[f"{name}_target"] = numer / denom
            else:
                smoothed = (stats["sum"] + SMOOTHING * self.global_mean) / (stats["count"] + SMOOTHING)
                out[f"{name}_target"] = (
                    values.map(smoothed)
                    .fillna(self.global_mean)
                    .astype(float)
                )

        for col in self.categorical_cols:
            if col not in df.columns:
                continue
            text = df[col].astype(str)
            out[f"{col}_len"] = text.str.len().fillna(0).astype(float)
            out[f"{col}_words"] = text.str.split().str.len().fillna(0).astype(float)
            if col in {"nrgiNotes", "notes"}:
                out[f"{col}_has_changed"] = text.str.contains("changed", case=False, na=False).astype(float)

        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out.astype(float)


def make_models():
    return {
        "lgbm": LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.04,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": XGBRegressor(
            n_estimators=1200,
            learning_rate=0.04,
            max_depth=7,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        ),
        "cat": CatBoostRegressor(
            iterations=1200,
            learning_rate=0.04,
            depth=7,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            verbose=0,
        ),
    }


def main():
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    train = train.dropna(subset=[TARGET]).copy()
    train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
    train = train.dropna(subset=[TARGET]).copy()
    train[TARGET] = train[TARGET].astype(float)

    feature_cols = [col for col in train.columns if col != TARGET]
    for col in feature_cols:
        if col not in test.columns:
            test[col] = np.nan

    train = clean_frame(train)
    test = clean_frame(test)

    numeric_cols = [
        col
        for col in feature_cols
        if pd.api.types.is_numeric_dtype(train[col]) or col == "reportYear"
    ]
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]

    train_split, val_split = train_test_split(
        train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    val_builder = FeatureBuilder(numeric_cols, categorical_cols).fit(train_split)
    X_train_val = val_builder.transform(train_split, oof=True)
    X_val = val_builder.transform(val_split, oof=False)
    X_test_val = val_builder.transform(test, oof=False)

    y_train_val = train_split[TARGET].to_numpy(dtype=float)
    y_val = val_split[TARGET].to_numpy(dtype=float)

    val_models = make_models()
    val_preds = {}
    val_scores = {}

    for name, model in val_models.items():
        model.fit(X_train_val, y_train_val)
        pred = model.predict(X_val)
        val_preds[name] = pred
        val_scores[name] = rmse(y_val, pred)

    weights = {}
    finite_scores = {name: score for name, score in val_scores.items() if np.isfinite(score) and score > 0}
    if finite_scores:
        inv_scores = {name: 1.0 / score for name, score in finite_scores.items()}
        total = sum(inv_scores.values())
        weights = {name: value / total for name, value in inv_scores.items()}
    else:
        n = len(val_models)
        weights = {name: 1.0 / n for name in val_models}

    final_builder = FeatureBuilder(numeric_cols, categorical_cols).fit(train)
    X_train_final = final_builder.transform(train, oof=True)
    X_test_final = final_builder.transform(test, oof=False)

    y_train_final = train[TARGET].to_numpy(dtype=float)

    final_models = make_models()
    test_preds = np.zeros(len(test), dtype=float)

    for name, model in final_models.items():
        model.fit(X_train_final, y_train_final)
        pred = model.predict(X_test_final)
        test_preds += weights.get(name, 0.0) * pred

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test), dtype=int),
            "entityPayment": test_preds.astype(float),
        }
    )
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()