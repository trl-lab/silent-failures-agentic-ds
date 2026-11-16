import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/spambase/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/spambase/data/test.parquet"
TARGET = "class"
RANDOM_STATE = 42


def to_binary_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)

    s = series.astype(str).str.strip().str.lower()
    return s.map(lambda v: 1 if v in {"1", "1.0", "spam", "yes", "true"} else 0)


def get_feature_columns(df: pd.DataFrame, target: str) -> list:
    return [c for c in df.columns if c != target]


def get_categorical_columns(df: pd.DataFrame, feature_cols: list) -> list:
    categorical_cols = []
    for col in feature_cols:
        dtype = df[col].dtype
        if dtype == object or isinstance(dtype, pd.CategoricalDtype):
            categorical_cols.append(col)
    return categorical_cols


def clean_numeric(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    for col in feature_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    return df


def build_preprocessor(feature_cols: list, categorical_cols: list) -> ColumnTransformer:
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    transformers = []

    if numeric_cols:
        transformers.append(("num", "passthrough", numeric_cols))

    if categorical_cols:
        transformers.append(
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical_cols,
            )
        )

    if not transformers:
        return ColumnTransformer(transformers=[], remainder="passthrough")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_model(name: str):
    if name == "lgbm":
        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )

    if name == "xgb":
        return XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist",
        )

    if name == "cat":
        return CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_seed=RANDOM_STATE,
            verbose=0,
        )

    raise ValueError(f"Unknown model: {name}")


def main() -> None:
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    feature_cols = get_feature_columns(train_df, TARGET)
    categorical_cols = get_categorical_columns(train_df, feature_cols)

    train_df = clean_numeric(train_df, feature_cols)
    test_df = clean_numeric(test_df, feature_cols)

    X_full = train_df[feature_cols].copy()
    y_full = to_binary_target(train_df[TARGET])

    X_test = test_df.drop(columns=[TARGET], errors="ignore").reindex(columns=feature_cols)

    X_train, X_val, y_train, y_val = train_test_split(
        X_full,
        y_full,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_full,
    )

    val_preprocessor = build_preprocessor(feature_cols, categorical_cols).fit(X_train)
    X_train_p = val_preprocessor.transform(X_train)
    X_val_p = val_preprocessor.transform(X_val)

    model_names = ["lgbm", "xgb", "cat"]
    val_scores = {}

    for name in model_names:
        try:
            model = make_model(name)
            model.fit(X_train_p, y_train)
            proba = model.predict_proba(X_val_p)[:, 1]
            score = roc_auc_score(y_val, proba)
            if np.isfinite(score):
                val_scores[name] = float(score)
        except Exception:
            pass

    final_preprocessor = build_preprocessor(feature_cols, categorical_cols).fit(X_full)
    X_full_p = final_preprocessor.transform(X_full)
    X_test_p = final_preprocessor.transform(X_test)

    if val_scores:
        names = list(val_scores.keys())
        weights = np.array([val_scores[name] for name in names], dtype=float)
        weights = np.clip(weights, 1e-6, None)
        weights /= weights.sum()

        test_proba = np.zeros(len(X_test_p), dtype=float)

        for name, weight in zip(names, weights):
            try:
                model = make_model(name)
                model.fit(X_full_p, y_full)
                test_proba += weight * model.predict_proba(X_test_p)[:, 1]
            except Exception:
                continue

        if not np.any(test_proba):
            model = make_model("lgbm")
            model.fit(X_full_p, y_full)
            test_proba = model.predict_proba(X_test_p)[:, 1]
    else:
        model = make_model("lgbm")
        model.fit(X_full_p, y_full)
        test_proba = model.predict_proba(X_test_p)[:, 1]

    test_proba = np.clip(test_proba, 0.0, 1.0)

    predictions = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            "class": test_proba,
        }
    )

    predictions.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()