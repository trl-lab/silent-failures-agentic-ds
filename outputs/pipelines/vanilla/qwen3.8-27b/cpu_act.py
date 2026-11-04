import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/cpu_act/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/cpu_act/data/test.parquet"
TARGET = "usr"


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def make_preprocessor(num_cols, cat_cols):
    transformers = []

    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                num_cols,
            )
        )

    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            )
        )

    if not transformers:
        return ColumnTransformer([], remainder="passthrough", sparse_threshold=0.0)

    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def make_models():
    return {
        "lgbm": LGBMRegressor(
            n_estimators=1500,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": XGBRegressor(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            objective="reg:squarederror",
        ),
        "cat": CatBoostRegressor(
            iterations=1500,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=0,
        ),
    }


def main():
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    train = train.copy()
    test = test.copy()

    train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
    train = train.dropna(subset=[TARGET])

    features = [c for c in train.columns if c != TARGET and c in test.columns]

    num_cols = []
    cat_cols = []

    for col in features:
        if (
            train[col].dtype == object
            or train[col].dtype.name == "category"
            or test[col].dtype == object
            or test[col].dtype.name == "category"
        ):
            cat_cols.append(col)
        else:
            num_cols.append(col)

    for col in num_cols:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")
        train[col] = train[col].replace([np.inf, -np.inf], np.nan)
        test[col] = test[col].replace([np.inf, -np.inf], np.nan)

    for col in cat_cols:
        train[col] = train[col].astype("object").fillna("missing")
        test[col] = test[col].astype("object").fillna("missing")

    X = train[features]
    y = train[TARGET]
    X_test = test[features]

    if len(X) == 0 or len(features) == 0:
        pred_test = np.full(len(X_test), float(y.mean()) if len(y) > 0 else 0.0)
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        models = make_models()
        val_scores = {}

        for name, model in models.items():
            try:
                pipeline = Pipeline(
                    [
                        ("prep", make_preprocessor(num_cols, cat_cols)),
                        ("model", model),
                    ]
                )
                pipeline.fit(X_train, y_train)
                val_pred = pipeline.predict(X_val)
                val_scores[name] = rmse(y_val, val_pred)
            except Exception:
                continue

        if not val_scores:
            pred_test = np.full(len(X_test), float(y.mean()))
        else:
            eps = 1e-8
            weights = {name: 1.0 / (score + eps) for name, score in val_scores.items()}
            total_weight = sum(weights.values())
            weights = {name: weight / total_weight for name, weight in weights.items()}

            test_preds = {}
            for name in weights:
                try:
                    pipeline = Pipeline(
                        [
                            ("prep", make_preprocessor(num_cols, cat_cols)),
                            ("model", make_models()[name]),
                        ]
                    )
                    pipeline.fit(X, y)
                    test_preds[name] = pipeline.predict(X_test)
                except Exception:
                    continue

            if not test_preds:
                pred_test = np.full(len(X_test), float(y.mean()))
            else:
                pred_test = np.zeros(len(X_test), dtype=float)
                for name, weight in weights.items():
                    if name in test_preds:
                        pred_test += weight * np.asarray(test_preds[name], dtype=float)

    if len(y) > 0:
        if y.min() >= 0 and y.max() <= 100:
            pred_test = np.clip(pred_test, 0.0, 100.0)
        else:
            pred_test = np.clip(pred_test, float(y.min()), float(y.max()))

    output = pd.DataFrame(
        {
            "id": np.arange(len(pred_test), dtype=int),
            "usr": pred_test,
        }
    )
    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()