import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/managed-care-enrollment/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/managed-care-enrollment/data/test.parquet"
TARGET = "Total Enrollment"


def is_categorical_series(s: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    )


def clean_target(y: pd.Series) -> pd.Series:
    y = pd.to_numeric(y, errors="coerce")
    y = y.replace([np.inf, -np.inf], np.nan)
    return y


def prepare_data(train: pd.DataFrame, test: pd.DataFrame, target: str):
    X = train.drop(columns=[target], errors="ignore").copy()
    y = train[target].copy() if target in train.columns else pd.Series(np.nan, index=train.index)
    X_test = test.copy()
    if target in X_test.columns:
        X_test = X_test.drop(columns=[target])

    y = clean_target(y)
    valid = y.notna()
    X = X.loc[valid].copy()
    y = y.loc[valid].reset_index(drop=True)
    X = X.reset_index(drop=True)

    cat_cols = [c for c in X.columns if is_categorical_series(X[c])]

    for col in cat_cols:
        len_col = f"{col}_len"
        X[len_col] = X[col].astype("string").str.len().fillna(0).astype(float)
        if col in X_test.columns:
            X_test[len_col] = X_test[col].astype("string").str.len().fillna(0).astype(float)
        else:
            X_test[len_col] = 0.0

    final_cat_cols = []
    for col in cat_cols:
        X[col] = X[col].astype("string").fillna("missing")
        if col in X_test.columns:
            X_test[col] = X_test[col].astype("string").fillna("missing")
        else:
            X_test[col] = "missing"

        cats = list(X[col].unique())
        if "missing" not in cats:
            cats.append("missing")

        X_test[col] = X_test[col].where(X_test[col].isin(cats), "missing")

        dtype = pd.CategoricalDtype(categories=cats, ordered=False)
        X[col] = X[col].astype(dtype)
        X_test[col] = X_test[col].astype(dtype)
        final_cat_cols.append(col)

    for col in X.columns:
        if col not in final_cat_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            if col in X_test.columns:
                X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
            else:
                X_test[col] = np.nan

    for col in X.columns:
        if col not in X_test.columns:
            X_test[col] = np.nan
    X_test = X_test[X.columns].copy()

    X = X.loc[:, X.notna().any()]
    X_test = X_test.loc[:, X.columns]
    final_cat_cols = [c for c in final_cat_cols if c in X.columns]

    num_cols = [c for c in X.columns if c not in final_cat_cols]
    for c in num_cols:
        X[c] = X[c].replace([np.inf, -np.inf], np.nan)
        X_test[c] = X_test[c].replace([np.inf, -np.inf], np.nan)

    return X, y, X_test, final_cat_cols


def make_model(n_estimators: int):
    return lgb.LGBMRegressor(
        objective="regression",
        learning_rate=0.03,
        n_estimators=n_estimators,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=10,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        max_bin=255,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        categorical_feature="auto",
    )


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    X, y, X_test, _ = prepare_data(train_df, test_df, TARGET)

    fallback_value = float(y.mean()) if len(y) else 0.0

    if len(X) == 0 or X.shape[1] == 0 or y.nunique() <= 1:
        pred = np.full(len(X_test), fallback_value, dtype=float)
    else:
        if len(X) >= 100:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            model = make_model(3000)
            callbacks = []
            if hasattr(lgb, "early_stopping"):
                callbacks.append(lgb.early_stopping(stopping_rounds=100, verbose=False))
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="l2",
                callbacks=callbacks,
            )
            best_iter = getattr(model, "best_iteration_", 0)
            if not best_iter or best_iter <= 0:
                best_iter = 3000
            best_iter = int(max(1, min(best_iter, 3000)))
        else:
            best_iter = 500

        final_model = make_model(best_iter)
        final_model.fit(X, y)
        pred = final_model.predict(X_test)

    pred = np.asarray(pred, dtype=float)
    pred = np.nan_to_num(pred, nan=fallback_value, posinf=0.0, neginf=0.0)
    if len(y) and float(y.min()) >= 0:
        pred = np.clip(pred, 0.0, None)

    out = pd.DataFrame({
        "id": np.arange(len(pred), dtype=int),
        "Total Enrollment": pred
    })
    out.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()