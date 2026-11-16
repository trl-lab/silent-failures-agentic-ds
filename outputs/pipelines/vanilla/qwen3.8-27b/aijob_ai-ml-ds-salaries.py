import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

try:
    from lightgbm import early_stopping
except Exception:
    early_stopping = None

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/aijob_ai-ml-ds-salaries/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/aijob_ai-ml-ds-salaries/data/test.parquet"
TARGET = "salary_in_usd"


def _is_categorical_series(s: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    )


def _prepare(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str):
    train_drop_cols = [c for c in [target, "id"] if c in train_df.columns]
    X_train = train_df.drop(columns=train_drop_cols, errors="ignore").copy()
    y_train = train_df[target].astype(float)

    test_drop_cols = [c for c in [target, "id"] if c in test_df.columns]
    X_test = test_df.drop(columns=test_drop_cols, errors="ignore").copy()

    valid = y_train.notna() & np.isfinite(y_train)
    X_train = X_train.loc[valid].copy()
    y_train = y_train.loc[valid].copy()

    cat_cols = [c for c in X_train.columns if _is_categorical_series(X_train[c])]

    for c in X_train.columns:
        if c not in X_test.columns:
            X_test[c] = np.nan

    for c in cat_cols:
        train_cats = X_train[c].dropna().unique()
        if len(train_cats) == 0:
            continue

        X_train[c] = X_train[c].astype("category")
        X_test[c] = X_test[c].astype("category")

        X_train[c] = X_train[c].cat.set_categories(train_cats)
        X_test[c] = X_test[c].cat.set_categories(train_cats)

    num_cols = [c for c in X_train.columns if c not in cat_cols]
    for c in num_cols:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce")

    X_train = X_train.dropna(axis=1, how="all")
    X_test = X_test.reindex(columns=X_train.columns)
    cat_cols = [c for c in cat_cols if c in X_train.columns]

    return X_train, y_train, X_test, cat_cols


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    X_train, y_train, X_test, cat_cols = _prepare(train_df, test_df, TARGET)

    if len(X_train) == 0 or X_train.shape[1] == 0:
        if len(y_train) > 0:
            mean_val = float(np.nanmean(y_train.to_numpy(dtype=float)))
            default = mean_val if np.isfinite(mean_val) else 0.0
        else:
            default = 0.0
        pred = np.full(len(test_df), default, dtype=float)
    else:
        params = {
            "n_estimators": 3000,
            "learning_rate": 0.02,
            "num_leaves": 63,
            "min_child_samples": 20,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
            "categorical_feature": cat_cols if cat_cols else "auto",
        }

        if len(X_train) > 10:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )

            val_model = LGBMRegressor(**params)
            fit_kwargs = {"eval_set": [(X_val, y_val)]}

            if early_stopping is not None:
                try:
                    fit_kwargs["callbacks"] = [
                        early_stopping(stopping_rounds=100, verbose=False)
                    ]
                except TypeError:
                    fit_kwargs["callbacks"] = [early_stopping(stopping_rounds=100)]

            try:
                val_model.fit(X_tr, y_tr, **fit_kwargs)
            except TypeError:
                val_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])

            val_pred = val_model.predict(X_val)
            val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
            print(f"Validation RMSE: {val_rmse:.6f}")

            best_iter = getattr(val_model, "best_iteration_", None)
            if best_iter is None or best_iter <= 0:
                best_iter = params["n_estimators"]

            best_iter = int(min(max(best_iter, 100), params["n_estimators"]))
        else:
            best_iter = params["n_estimators"]

        final_params = dict(params)
        final_params["n_estimators"] = best_iter

        final_model = LGBMRegressor(**final_params)
        final_model.fit(X_train, y_train)
        pred = final_model.predict(X_test)

    pred = np.asarray(pred, dtype=float)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    pred = np.clip(pred, 0.0, None)

    pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            "salary_in_usd": pred,
        }
    ).to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()