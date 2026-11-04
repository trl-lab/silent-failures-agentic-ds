import pandas as pd
import numpy as np
import inspect
from sklearn.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
import lightgbm as lgb

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/local-law-enforcements/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/local-law-enforcements/data/test.parquet"
TARGET = "number_of_officers"

DROP_COLS = {
    "OBJECTID",
    "ID",
    "NAME",
    "ADDRESS",
    "CITY",
    "ZIP",
    "ZIP4",
    "TELEPHONE",
    "SOURCE",
    "SOURCEDATE",
    "VAL_DATE",
    "VAL_METHOD",
    "WEBSITE",
    "CI_ID",
    "CSLLEA08ID",
}


def prepare_features(train: pd.DataFrame, test: pd.DataFrame, target: str):
    train = train.copy()
    test = test.copy()

    if target in test.columns:
        test = test.drop(columns=[target])

    if target in train.columns:
        y = pd.to_numeric(train[target], errors="coerce")
    else:
        y = pd.Series(np.zeros(len(train), dtype=float), index=train.index)

    valid = y.notna() & np.isfinite(y)
    train = train.loc[valid].copy()
    y = y.loc[valid].copy()

    train = train.drop(columns=[target], errors="ignore")

    common_cols = [c for c in train.columns if c in test.columns]
    train = train[common_cols]
    test = test[common_cols]

    drop_present = [c for c in DROP_COLS if c in train.columns]
    train = train.drop(columns=drop_present, errors="ignore")
    test = test.drop(columns=drop_present, errors="ignore")

    for col in train.columns:
        dtype = train[col].dtype
        if is_datetime64_any_dtype(dtype) or isinstance(dtype, pd.DatetimeTZDtype):
            train[col] = pd.to_numeric(train[col], errors="coerce")
            test[col] = pd.to_numeric(test[col], errors="coerce")

    cat_cols = [col for col in train.columns if not is_numeric_dtype(train[col].dtype)]

    high_cardinality_drop = [
        col for col in cat_cols if train[col].nunique(dropna=True) > 1000
    ]
    train = train.drop(columns=high_cardinality_drop, errors="ignore")
    test = test.drop(columns=high_cardinality_drop, errors="ignore")
    cat_cols = [col for col in cat_cols if col in train.columns]

    for col in cat_cols:
        train[col] = train[col].fillna("missing").astype(str)
        test[col] = test[col].fillna("missing").astype(str)

        categories = set(train[col].unique())
        categories.add("missing")

        test[col] = test[col].where(test[col].isin(categories), "missing")

        train[col] = pd.Categorical(train[col], categories=sorted(categories))
        test[col] = pd.Categorical(test[col], categories=sorted(categories))

    num_cols = [col for col in train.columns if col not in cat_cols]

    for col in num_cols:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")

    if num_cols:
        train[num_cols] = train[num_cols].replace([np.inf, -np.inf], np.nan)
        test[num_cols] = test[num_cols].replace([np.inf, -np.inf], np.nan)

    drop_cols = []
    for col in train.columns:
        if train[col].isna().all():
            drop_cols.append(col)
        elif train[col].nunique(dropna=True) <= 1:
            drop_cols.append(col)

    train = train.drop(columns=drop_cols, errors="ignore")
    test = test.drop(columns=drop_cols, errors="ignore")

    cat_cols = [col for col in cat_cols if col in train.columns]

    return train, test, y, cat_cols


def train_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, cat_cols: list):
    if len(X_train) == 0 or X_train.shape[1] == 0:
        return None

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    cat_feature = cat_cols if cat_cols else "auto"
    train_signature = inspect.signature(lgb.train)

    if len(X_train) >= 20:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        train_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_feature)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=cat_feature)

        train_kwargs = {
            "params": params,
            "train_set": train_data,
            "num_boost_round": 5000,
            "valid_sets": [valid_data],
            "valid_names": ["valid"],
        }

        if "early_stopping_rounds" in train_signature.parameters:
            train_kwargs["early_stopping_rounds"] = 100
        if "verbose_eval" in train_signature.parameters:
            train_kwargs["verbose_eval"] = False

        booster = lgb.train(**train_kwargs)

        best_iter = getattr(booster, "best_iteration", 0)
        if not best_iter:
            best_iter = 1000
        best_iter = max(1, int(best_iter))
    else:
        best_iter = 100

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_feature)

    final_kwargs = {
        "params": params,
        "train_set": train_data,
        "num_boost_round": best_iter,
    }
    if "verbose_eval" in train_signature.parameters:
        final_kwargs["verbose_eval"] = False

    final_booster = lgb.train(**final_kwargs)
    return final_booster


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    X_train, X_test, y_train, cat_cols = prepare_features(train_df, test_df, TARGET)

    model = train_model(X_train, y_train, X_test, cat_cols)

    if model is None:
        fallback = float(y_train.mean()) if len(y_train) > 0 else 0.0
        predictions = np.full(len(X_test), fallback, dtype=float)
    else:
        predictions = model.predict(X_test)
        predictions = np.asarray(predictions, dtype=float)
        predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
        predictions = np.clip(predictions, 0.0, None)

    output = pd.DataFrame(
        {
            "id": np.arange(len(predictions), dtype=int),
            "number_of_officers": predictions,
        }
    )
    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()