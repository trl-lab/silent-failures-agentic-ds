import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/diabetes/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/diabetes/data/test.parquet"
TARGET = "class"
RANDOM_STATE = 42
ZERO_AS_MISSING = {"plas", "pres", "skin", "insu", "mass"}


class SafeMedianImputer(SimpleImputer):
    def __init__(self):
        super().__init__(strategy="median")

    def fit(self, X, y=None):
        super().fit(X, y)
        if hasattr(self, "statistics_"):
            stats = np.asarray(self.statistics_, dtype=float)
            stats = np.where(np.isnan(stats), 0.0, stats)
            self.statistics_ = stats
        return self


def load_data():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    return train_df, test_df


def prepare_target(y):
    y = y.astype(str)
    labels = sorted(y.unique())

    positive_candidates = ["tested_positive", "positive", "1", "yes", "true"]
    positive_label = None
    for candidate in positive_candidates:
        if candidate in labels:
            positive_label = candidate
            break

    if positive_label is None:
        positive_label = labels[-1]

    y_binary = (y == positive_label).astype(int)
    return y_binary, positive_label


def clean_features(x_train, x_test):
    x_train = x_train.copy()
    x_test = x_test.copy()

    for df in (x_train, x_test):
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    for col in ZERO_AS_MISSING.intersection(x_train.columns):
        if col in x_test.columns and pd.api.types.is_numeric_dtype(x_train[col]):
            x_train.loc[x_train[col] == 0, col] = np.nan
            x_test.loc[x_test[col] == 0, col] = np.nan

    numeric_cols = []
    categorical_cols = []

    for col in x_train.columns:
        series = x_train[col]
        is_categorical = (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
            or pd.api.types.is_string_dtype(series)
        )

        if is_categorical:
            converted = pd.to_numeric(series, errors="coerce")
            non_missing = series.notna().sum()
            if non_missing > 0 and converted.notna().sum() >= 0.9 * non_missing:
                x_train[col] = converted
                if col in x_test.columns:
                    x_test[col] = pd.to_numeric(x_test[col], errors="coerce")
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        else:
            numeric_cols.append(col)

    return x_train, x_test, numeric_cols, categorical_cols


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(numeric_cols, categorical_cols):
    transformers = []

    if numeric_cols:
        transformers.append(("num", SafeMedianImputer(), numeric_cols))

    if categorical_cols:
        cat_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", make_one_hot_encoder()),
            ]
        )
        transformers.append(("cat", cat_pipeline, categorical_cols))

    if not transformers:
        return "passthrough"

    return ColumnTransformer(transformers, remainder="drop")


def make_models():
    return {
        "lightgbm": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="auc",
        ),
        "catboost": CatBoostClassifier(
            iterations=500,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            verbose=0,
        ),
    }


def fit_and_score(models, x_fit, y_fit, x_val, y_val, numeric_cols, categorical_cols):
    val_probabilities = {}

    for name, model in models.items():
        pipeline = Pipeline(
            [
                ("preprocessor", make_preprocessor(numeric_cols, categorical_cols)),
                ("model", model),
            ]
        )
        pipeline.fit(x_fit, y_fit)
        val_probabilities[name] = pipeline.predict_proba(x_val)[:, 1]

    ensemble_proba = np.mean(list(val_probabilities.values()), axis=0)

    scores = {name: roc_auc_score(y_val, proba) for name, proba in val_probabilities.items()}
    scores["ensemble"] = roc_auc_score(y_val, ensemble_proba)

    best_name = max(scores, key=scores.get)
    return best_name, scores


def main():
    train_df, test_df = load_data()

    x_train = train_df.drop(columns=[TARGET], errors="ignore")
    y_raw = train_df[TARGET]
    y_train, _ = prepare_target(y_raw)

    x_test = test_df.drop(columns=[TARGET], errors="ignore")
    x_test = x_test.reindex(columns=x_train.columns)

    x_train, x_test, numeric_cols, categorical_cols = clean_features(x_train, x_test)

    if len(x_test) == 0:
        pd.DataFrame({"id": np.arange(0, dtype=int), "class": []}).to_csv("predictions.csv", index=False)
        return

    if len(x_train) == 0 or y_train.nunique() < 2:
        default_proba = float(y_train.iloc[0]) if len(y_train) > 0 else 0.5
        predictions = pd.DataFrame(
            {
                "id": np.arange(len(x_test), dtype=int),
                "class": default_proba,
            }
        )
        predictions.to_csv("predictions.csv", index=False)
        return

    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    models = make_models()
    best_name, _ = fit_and_score(models, x_fit, y_fit, x_val, y_val, numeric_cols, categorical_cols)

    if best_name == "ensemble":
        final_pipelines = []
        for model in make_models().values():
            pipeline = Pipeline(
                [
                    ("preprocessor", make_preprocessor(numeric_cols, categorical_cols)),
                    ("model", model),
                ]
            )
            pipeline.fit(x_train, y_train)
            final_pipelines.append(pipeline)

        test_proba = np.mean(
            [pipeline.predict_proba(x_test)[:, 1] for pipeline in final_pipelines],
            axis=0,
        )
    else:
        model = make_models()[best_name]
        pipeline = Pipeline(
            [
                ("preprocessor", make_preprocessor(numeric_cols, categorical_cols)),
                ("model", model),
            ]
        )
        pipeline.fit(x_train, y_train)
        test_proba = pipeline.predict_proba(x_test)[:, 1]

    test_proba = np.clip(test_proba, 0.0, 1.0)

    predictions = pd.DataFrame(
        {
            "id": np.arange(len(x_test), dtype=int),
            "class": test_proba,
        }
    )
    predictions.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()