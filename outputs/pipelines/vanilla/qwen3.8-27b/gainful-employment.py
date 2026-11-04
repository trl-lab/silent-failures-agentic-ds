import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TARGET = "Debt-to-Earnings Annual Rate"
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/gainful-employment/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/gainful-employment/data/test.parquet"


def finite_predictions(pred, fallback):
    pred = np.asarray(pred, dtype=float)
    pred = np.where(np.isfinite(pred), pred, fallback)
    return pred


def build_preprocessor(numeric_cols, categorical_cols):
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            )
        )
    if not transformers:
        return None
    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_models():
    return {
        "lgbm": LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
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
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="rmse",
        ),
        "cat": CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=0,
        ),
    }


def main():
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    if TARGET not in train.columns:
        fallback = 0.0
    else:
        train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
        train = train.replace([np.inf, -np.inf], np.nan)
        train = train.dropna(subset=[TARGET])
        fallback = float(train[TARGET].median()) if len(train) else 0.0

    if len(train) == 0:
        pd.DataFrame(
            {
                "id": np.arange(len(test), dtype=int),
                TARGET: np.full(len(test), fallback, dtype=float),
            }
        ).to_csv("predictions.csv", index=False)
        return

    feature_cols = [c for c in train.columns if c != TARGET and c in test.columns]

    explicit_drop = {
        "Institution Code (six-digit OPEID)",
        "Institution Name",
        "City",
        "Zip",
        "CIP Name",
    }
    feature_cols = [c for c in feature_cols if c not in explicit_drop]
    feature_cols = [
        c
        for c in feature_cols
        if "Name" not in c and train[c].notna().any() and train[c].nunique(dropna=True) > 1
    ]

    n_train = len(train)
    cardinality_limit = max(1000, int(0.5 * n_train))
    feature_cols = [
        c for c in feature_cols if train[c].nunique(dropna=True) <= cardinality_limit
    ]

    if not feature_cols:
        pd.DataFrame(
            {
                "id": np.arange(len(test), dtype=int),
                TARGET: np.full(len(test), fallback, dtype=float),
            }
        ).to_csv("predictions.csv", index=False)
        return

    numeric_cols = []
    categorical_cols = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(train[col]):
            if "Code" in col and train[col].nunique(dropna=True) > 10:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    for col in numeric_cols:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")
        train[col] = train[col].replace([np.inf, -np.inf], np.nan)
        test[col] = test[col].replace([np.inf, -np.inf], np.nan)

    for col in categorical_cols:
        train[col] = train[col].astype("string")
        test[col] = test[col].astype("string")

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    if preprocessor is None:
        pd.DataFrame(
            {
                "id": np.arange(len(test), dtype=int),
                TARGET: np.full(len(test), fallback, dtype=float),
            }
        ).to_csv("predictions.csv", index=False)
        return

    X = train[feature_cols]
    y = train[TARGET]
    X_test = test[feature_cols]

    if len(train) >= 100:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        X_train, X_val, y_train, y_val = X, X, y, y

    val_scores = {}
    val_predictions = {}
    for name, model in make_models().items():
        try:
            pipe = Pipeline([("prep", clone(preprocessor)), ("model", model)])
            pipe.fit(X_train, y_train)
            pred = finite_predictions(pipe.predict(X_val), fallback)
            rmse = root_mean_squared_error(y_val, pred)
            if np.isfinite(rmse):
                val_scores[name] = float(rmse)
                val_predictions[name] = pred
        except Exception:
            continue

    if val_scores:
        weights = {name: 1.0 / max(score, 1e-9) for name, score in val_scores.items()}
        total_weight = sum(weights.values())
        weights = {name: weight / total_weight for name, weight in weights.items()}
    else:
        weights = {"lgbm": 1.0}

    final_predictions = np.zeros(len(X_test), dtype=float)
    for name, weight in weights.items():
        try:
            model = make_models()[name]
            pipe = Pipeline([("prep", clone(preprocessor)), ("model", model)])
            pipe.fit(X, y)
            pred = finite_predictions(pipe.predict(X_test), fallback)
            final_predictions += weight * pred
        except Exception:
            continue

    if not np.any(np.isfinite(final_predictions)):
        final_predictions = np.full(len(X_test), fallback, dtype=float)

    pd.DataFrame(
        {
            "id": np.arange(len(test), dtype=int),
            TARGET: final_predictions,
        }
    ).to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()