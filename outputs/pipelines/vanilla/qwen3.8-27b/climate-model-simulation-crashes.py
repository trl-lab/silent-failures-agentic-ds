import pandas as pd
import numpy as np
import inspect

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/climate-model-simulation-crashes/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/climate-model-simulation-crashes/data/test.parquet"
TARGET = "outcome"
RANDOM_STATE = 42


def encode_binary_target(series: pd.Series) -> pd.Series:
    """Convert target to 0/1 integers, where 1 means success."""
    s = series.copy()

    if isinstance(s.dtype, pd.CategoricalDtype):
        s = s.astype(str)

    s = s.astype(str).str.strip().str.lower()

    mapping = {
        "0": 0,
        "1": 1,
        "0.0": 0,
        "1.0": 1,
        "failure": 0,
        "fail": 0,
        "crash": 0,
        "succeed": 1,
        "success": 1,
        "ok": 1,
        "pass": 1,
    }

    mapped = s.map(mapping)
    if mapped.notna().any():
        s = mapped

    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().any():
        s = numeric

    if s.isna().any():
        s = pd.Categorical(s).codes

    s = pd.to_numeric(s, errors="coerce")

    if s.isna().any():
        raise ValueError("Could not convert target column to binary 0/1 values.")

    return s.astype(int)


def make_one_hot_encoder() -> OneHotEncoder:
    """Create a dense OneHotEncoder compatible with older/newer sklearn."""
    params = inspect.signature(OneHotEncoder.__init__).parameters
    if "sparse_output" in params:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return OneHotEncoder(handle_unknown="ignore", sparse=False)


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Return feature columns and split them into numeric/categorical groups."""
    feature_cols = [
        col for col in train_df.columns
        if col in test_df.columns and col != TARGET
    ]

    numeric_cols = []
    categorical_cols = []

    for col in feature_cols:
        train_series = train_df[col]
        test_series = test_df[col]

        if pd.api.types.is_numeric_dtype(train_series):
            numeric_cols.append(col)
            continue

        train_numeric = pd.to_numeric(train_series, errors="coerce")
        test_numeric = pd.to_numeric(test_series, errors="coerce")

        non_null = int(train_series.notna().sum())
        if non_null > 0 and train_numeric.notna().sum() / non_null >= 0.9:
            train_df[col] = train_numeric
            test_df[col] = test_numeric
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return feature_cols, numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols):
    """Build a preprocessing pipeline for numeric and categorical features."""
    transformers = []

    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                numeric_cols,
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", make_one_hot_encoder()),
                ]),
                categorical_cols,
            )
        )

    if not transformers:
        return None

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_models(y_fail: pd.Series):
    """Create candidate classifiers trained to predict failure."""
    pos = int((y_fail == 1).sum())
    neg = int((y_fail == 0).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    return {
        "lgbm": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
            is_unbalance=True,
        ),
        "xgb": XGBClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="auc",
            scale_pos_weight=scale_pos_weight,
        ),
        "catboost": CatBoostClassifier(
            iterations=500,
            learning_rate=0.03,
            depth=4,
            random_seed=RANDOM_STATE,
            verbose=0,
            auto_class_weights="Balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        ),
    }


def predict_success_probability(pipeline, X) -> np.ndarray:
    """Return P(success) = 1 - P(failure)."""
    prob_fail = pipeline.predict_proba(X)[:, 1]
    prob_success = 1.0 - prob_fail
    return np.clip(prob_success, 0.0, 1.0)


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    y_success = encode_binary_target(train_df[TARGET])
    y_fail = 1 - y_success

    feature_cols, numeric_cols, categorical_cols = prepare_features(train_df, test_df)

    if not feature_cols:
        predictions = pd.DataFrame({
            "id": np.arange(len(test_df), dtype=int),
            "outcome": 1.0,
        })
        predictions.to_csv("predictions.csv", index=False)
        return

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    if len(np.unique(y_fail)) < 2:
        constant = 1.0 if int(y_success.iloc[0]) == 1 else 0.0
        predictions = pd.DataFrame({
            "id": np.arange(len(test_df), dtype=int),
            "outcome": constant,
        })
        predictions.to_csv("predictions.csv", index=False)
        return

    X_fit, X_val, y_fail_fit, y_success_val = train_test_split(
        X_train,
        y_success,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_fail,
    )

    y_fail_fit = 1 - y_fail_fit

    models = make_models(y_fail)
    val_scores = {}
    fitted_val_models = {}

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        pipeline.fit(X_fit, y_fail_fit)
        prob_success_val = predict_success_probability(pipeline, X_val)

        try:
            auc = roc_auc_score(y_success_val, prob_success_val)
        except ValueError:
            auc = 0.5

        val_scores[name] = auc
        fitted_val_models[name] = pipeline

        print(f"{name}: validation ROC AUC = {auc:.6f}")

    weights = np.array([max(val_scores[name] - 0.5, 0.0) for name in models], dtype=float)
    if weights.sum() <= 0:
        weights = np.ones(len(models), dtype=float)
    weights = weights / weights.sum()

    final_predictions = np.zeros(len(X_test), dtype=float)

    for (name, _), weight in zip(models.items(), weights):
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", models[name]),
        ])

        pipeline.fit(X_train, y_fail)
        prob_success_test = predict_success_probability(pipeline, X_test)
        final_predictions += weight * prob_success_test

    final_predictions = np.clip(final_predictions, 0.0, 1.0)

    predictions = pd.DataFrame({
        "id": np.arange(len(test_df), dtype=int),
        "outcome": final_predictions,
    })

    predictions.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()