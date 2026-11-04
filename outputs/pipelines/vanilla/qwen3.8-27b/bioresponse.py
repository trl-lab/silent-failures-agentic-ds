import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/bioresponse/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/bioresponse/data/test.parquet"
TARGET = "Activity"
RANDOM_STATE = 42
EPS = 1e-6


def load_data():
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    if TARGET in test.columns:
        test = test.drop(columns=[TARGET])

    y = train[TARGET].astype(int)
    X = train.drop(columns=[TARGET])

    return X, y, test


def prepare_features(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    non_numeric_cols = [
        col for col in X_train.columns
        if not pd.api.types.is_numeric_dtype(X_train[col])
    ]

    for col in X_test.columns:
        if col in X_train.columns and not pd.api.types.is_numeric_dtype(X_test[col]):
            if col not in non_numeric_cols:
                non_numeric_cols.append(col)

    non_numeric_cols = list(dict.fromkeys(non_numeric_cols))

    if non_numeric_cols:
        for col in non_numeric_cols:
            X_train[col] = X_train[col].fillna("missing").astype(str)
            X_test[col] = X_test[col].fillna("missing").astype(str)

        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train[non_numeric_cols] = encoder.fit_transform(X_train[non_numeric_cols])
        X_test[non_numeric_cols] = encoder.transform(X_test[non_numeric_cols])

    for col in X_train.columns:
        if not pd.api.types.is_numeric_dtype(X_train[col]):
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    return X_train, X_test


def make_models():
    return [
        (
            "lgbm",
            LGBMClassifier(
                n_estimators=1500,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=-1,
                min_child_samples=20,
                subsample=0.9,
                subsample_freq=1,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
        ),
        (
            "xgb",
            XGBClassifier(
                n_estimators=1500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric="logloss",
                tree_method="hist",
            ),
        ),
        (
            "cat",
            CatBoostClassifier(
                iterations=1500,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3.0,
                random_seed=RANDOM_STATE,
                verbose=0,
            ),
        ),
    ]


def fit_with_validation(name, model, X_train, y_train, X_val, y_val):
    if name == "lgbm":
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="logloss",
            early_stopping_rounds=100,
            verbose=False,
        )
        best_iter = getattr(model, "best_iteration_", None)
        return model, best_iter

    if name == "xgb":
        try:
            model.set_params(early_stopping_rounds=100)
        except Exception:
            pass

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        best_iter = getattr(model, "best_iteration", None)
        return model, best_iter

    if name == "cat":
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,
            verbose=0,
        )
        best_iter = model.get_best_iteration()
        return model, best_iter

    model.fit(X_train, y_train)
    return model, None


def refit_for_final(name, model, X_full, y_full, best_iter):
    params = model.get_params()

    for key in ("early_stopping_rounds", "callbacks"):
        if key in params:
            params.pop(key)

    if best_iter is None or best_iter <= 0:
        if "n_estimators" in params:
            n_trees = params["n_estimators"]
        elif "iterations" in params:
            n_trees = params["iterations"]
        else:
            n_trees = 1000
    else:
        n_trees = int(max(best_iter, 100))

    if "n_estimators" in params:
        params["n_estimators"] = n_trees
    if "iterations" in params:
        params["iterations"] = n_trees

    final_model = type(model)(**params)
    final_model.fit(X_full, y_full)
    return final_model


def main():
    X, y, X_test = load_data()
    X, X_test = prepare_features(X, X_test)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = make_models()
    fitted_models = {}
    best_iterations = {}

    for name, model in models:
        try:
            fitted_model, best_iter = fit_with_validation(
                name,
                model,
                X_train,
                y_train,
                X_val,
                y_val,
            )
            val_proba = fitted_model.predict_proba(X_val)[:, 1]
            val_score = log_loss(y_val, np.clip(val_proba, EPS, 1.0 - EPS))
            fitted_models[name] = fitted_model
            best_iterations[name] = best_iter
            print(f"{name} validation logloss: {val_score:.6f}")
        except Exception as exc:
            print(f"{name} failed: {exc}")

    if not fitted_models:
        fallback = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        fallback.fit(X, y)
        test_proba = fallback.predict_proba(X_test)[:, 1]
    else:
        final_models = []

        for name, model in models:
            if name not in fitted_models:
                continue

            try:
                final_model = refit_for_final(
                    name,
                    model,
                    X,
                    y,
                    best_iterations[name],
                )
                final_models.append(final_model)
            except Exception as exc:
                print(f"{name} refit failed: {exc}")
                final_models.append(fitted_models[name])

        if not final_models:
            test_proba = np.full(len(X_test), float(y.mean()), dtype=float)
        else:
            probabilities = []
            for model in final_models:
                proba = model.predict_proba(X_test)[:, 1]
                probabilities.append(np.clip(proba, EPS, 1.0 - EPS))
            test_proba = np.mean(probabilities, axis=0)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(X_test), dtype=int),
            "Activity": test_proba.astype(float),
        }
    )
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()