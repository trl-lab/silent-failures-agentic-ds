import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/mercedes-benz-greener-manufacturing/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/mercedes-benz-greener-manufacturing/data/test.parquet"
TARGET = "y"
ID_COL = "ID"
RANDOM_STATE = 42


def prepare_features(X: pd.DataFrame, X_test: pd.DataFrame):
    cat_cols = []

    for col in X.columns:
        if X[col].dtype == object or X[col].dtype.name == "category":
            train_series = X[col].astype(object)
            test_series = X_test[col].astype(object)

            non_null_train = train_series.dropna()
            if len(non_null_train) > 0:
                converted = pd.to_numeric(non_null_train, errors="coerce")
                if converted.notna().all():
                    X[col] = pd.to_numeric(train_series, errors="coerce")
                    X_test[col] = pd.to_numeric(test_series, errors="coerce")
                    continue

            cat_cols.append(col)

            train_vals = train_series.fillna("missing")
            test_vals = test_series.fillna("missing")

            train_categories = set(train_vals.unique())
            train_categories.add("missing")
            categories = sorted(train_categories, key=str)

            X[col] = pd.Categorical(train_vals, categories=categories)
            test_vals_mapped = test_vals.map(lambda v: v if v in train_categories else "missing")
            X_test[col] = pd.Categorical(test_vals_mapped, categories=categories)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    return X, X_test, cat_cols


def main():
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    train = train.dropna(subset=[TARGET]).reset_index(drop=True)
    test = test.reset_index(drop=True)

    feature_cols = [
        col for col in train.columns
        if col not in (TARGET, ID_COL) and col in test.columns
    ]

    X = train[feature_cols].copy()
    y = train[TARGET].copy()
    X_test = test[feature_cols].copy()

    X, X_test, cat_cols = prepare_features(X, X_test)

    default_pred = float(y.mean()) if len(y) else 0.0

    if X.shape[1] == 0 or len(X) == 0:
        predictions = np.full(len(X_test), default_pred, dtype=float)
    else:
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE
            )

            params = {
                "objective": "regression",
                "metric": "rmse",
                "learning_rate": 0.05,
                "num_leaves": 63,
                "max_depth": -1,
                "min_child_samples": 50,
                "subsample": 0.8,
                "subsample_freq": 1,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": RANDOM_STATE,
                "n_jobs": -1,
                "verbose": -1,
            }

            model = LGBMRegressor(n_estimators=3000, **params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                early_stopping_rounds=100,
                categorical_feature=cat_cols,
            )

            best_iter = int(getattr(model, "best_iteration_", 0) or 0)
            if best_iter <= 0:
                best_iter = int(model.n_estimators)

            final_model = LGBMRegressor(n_estimators=max(1, best_iter), **params)
            final_model.fit(X, y, categorical_feature=cat_cols)

            predictions = final_model.predict(X_test)
        except Exception:
            predictions = np.full(len(X_test), default_pred, dtype=float)

    predictions = np.asarray(predictions, dtype=float)
    predictions = np.where(np.isfinite(predictions), predictions, default_pred)

    pd.DataFrame({
        "id": np.arange(len(predictions), dtype=int),
        "y": predictions
    }).to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()