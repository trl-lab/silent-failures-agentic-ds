import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from pandas.api.types import is_string_dtype, is_categorical_dtype

TARGET = "BEDS"
SEED = 42
MAX_CATEGORIES = 1000


def is_drop_column(col: str) -> bool:
    c = str(col).lower().strip()
    if c in {
        "id",
        "index",
        "row_id",
        "identifier",
        "uid",
        "hospital_id",
        "phone",
        "fax",
        "address",
    }:
        return True
    if c.startswith("id") or c.endswith("_id") or c.endswith(" id"):
        return True
    if "phone" in c or "fax" in c or "address" in c:
        return True
    return False


def prepare_data():
    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    if TARGET not in train.columns:
        raise ValueError("Target column BEDS not found in train.parquet.")

    y = train[TARGET].astype(float)
    y = y.replace([np.inf, -np.inf], np.nan)
    X_train = train.drop(columns=[TARGET]).copy()

    valid = y.notna()
    y = y[valid]
    X_train = X_train[valid]

    X_test = test.copy()

    common = [
        c for c in X_train.columns if c in X_test.columns and not is_drop_column(c)
    ]
    if not common:
        common = [c for c in X_train.columns if not is_drop_column(c)]

    X_train = X_train[common]
    X_test = X_test.reindex(columns=common)

    keep = []
    for c in common:
        s = X_train[c]
        if s.notna().sum() == 0:
            continue
        if s.nunique(dropna=True) <= 1:
            continue
        keep.append(c)

    X_train = X_train[keep]
    X_test = X_test[keep]
    features = keep

    cat_cols = [
        c
        for c in features
        if is_string_dtype(X_train[c])
        or is_categorical_dtype(X_train[c])
        or is_string_dtype(X_test[c])
        or is_categorical_dtype(X_test[c])
    ]
    num_cols = [c for c in features if c not in cat_cols]

    for c in num_cols:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce")

    if num_cols:
        X_train[num_cols] = X_train[num_cols].replace([np.inf, -np.inf], np.nan)
        X_test[num_cols] = X_test[num_cols].replace([np.inf, -np.inf], np.nan)

    for c in cat_cols:
        if is_categorical_dtype(X_train[c]):
            X_train[c] = X_train[c].cat.remove_unused_categories()
        else:
            X_train[c] = X_train[c].astype(object)

        if is_categorical_dtype(X_test[c]):
            X_test[c] = X_test[c].cat.remove_unused_categories()
        else:
            X_test[c] = X_test[c].astype(object)

    return X_train, X_test, y, features, cat_cols


def encode_categoricals(
    X_tr, X_other, cat_cols, mapping=None, max_categories=MAX_CATEGORIES
):
    X_tr = X_tr.copy()
    X_other = X_other.copy() if X_other is not None else None

    if mapping is None:
        mapping = {}
        for c in cat_cols:
            counts = X_tr[c].value_counts(dropna=True)
            top = counts.head(max_categories).index.tolist()
            mapping[c] = {v: i for i, v in enumerate(top)}

    for c in cat_cols:
        mp = mapping[c]
        X_tr[c] = X_tr[c].map(mp).fillna(-1).astype(np.int32)
        if X_other is not None:
            X_other[c] = X_other[c].map(mp).fillna(-1).astype(np.int32)

    return X_tr, X_other, mapping


def main():
    X_train, X_test, y, features, cat_cols = prepare_data()

    if len(features) == 0 or len(y) == 0:
        fallback = float(y.mean()) if len(y) else 0.0
        metric = (
            0.0
            if len(y) == 0
            else root_mean_squared_error(y, np.full(len(y), fallback))
        )
        os.makedirs("./working", exist_ok=True)
        pd.DataFrame({"id": np.arange(len(X_test)), "BEDS": fallback}).to_csv(
            "./working/submission.csv", index=False
        )
        print(f"CV RMSE: {metric:.4f}")
        return

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "seed": SEED,
        "verbosity": -1,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rmses = []
    best_iters = []

    for tr_idx, val_idx in kf.split(X_train):
        X_tr_raw = X_train.iloc[tr_idx]
        X_val_raw = X_train.iloc[val_idx]
        y_tr = y.iloc[tr_idx]
        y_val = y.iloc[val_idx]

        X_tr, X_val, _ = encode_categoricals(X_tr_raw, X_val_raw, cat_cols, None)

        dtrain = lgb.Dataset(
            X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False
        )
        dval = lgb.Dataset(
            X_val, label=y_val, categorical_feature=cat_cols, free_raw_data=False
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
        )

        best_it = getattr(model, "best_iteration", 0) or 1
        best_iters.append(best_it)

        pred = model.predict(X_val, num_iteration=best_it)
        pred = np.maximum(pred, 0.0)
        rmses.append(root_mean_squared_error(y_val, pred))

    cv_rmse = float(np.mean(rmses))

    X_tr_final, X_te_final, _ = encode_categoricals(X_train, X_test, cat_cols, None)
    dtrain_final = lgb.Dataset(
        X_tr_final, label=y, categorical_feature=cat_cols, free_raw_data=False
    )
    num_round = int(np.median(best_iters)) if best_iters else 1000
    final_model = lgb.train(params, dtrain_final, num_boost_round=max(1, num_round))

    test_pred = final_model.predict(X_te_final)
    test_pred = np.maximum(test_pred, 0.0)

    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame({"id": np.arange(len(X_test)), "BEDS": test_pred})
    submission.to_csv("./working/submission.csv", index=False)

    print(f"CV RMSE: {cv_rmse:.4f}")


if __name__ == "__main__":
    main()
