import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

TARGET = "Population for whom broadband available, 2019 (%)"
TRAIN_PATH = "./input/train.parquet"
TEST_PATH = "./input/test.parquet"
OUT_DIR = "./working"


def prepare(train: pd.DataFrame, test: pd.DataFrame):
    # Remove identifier columns if present; submission id will be the row index.
    id_cols = [c for c in ("id", "ID", "Id") if c in train.columns]
    train = train.drop(columns=id_cols)
    test = test.drop(columns=[c for c in id_cols if c in test.columns])

    if TARGET in test.columns:
        test = test.drop(columns=[TARGET])

    if TARGET not in train.columns:
        raise ValueError("Target column not found in train.")

    y = train[TARGET].copy()
    train = train.drop(columns=[TARGET])

    feature_cols = list(train.columns)
    X_train = train[feature_cols].copy()
    X_test = test.reindex(columns=feature_cols).copy()

    # Identify categorical-like columns.
    cat_cols = []
    for c in feature_cols:
        dtype = X_train[c].dtype
        if dtype.name in ("object", "category", "string") or str(dtype).startswith(
            "string"
        ):
            cat_cols.append(c)

    num_cols = [c for c in feature_cols if c not in cat_cols]

    # Numeric cleanup: coerce to numeric and replace infinities with NaN.
    for c in num_cols:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
        X_train[c] = X_train[c].replace([np.inf, -np.inf], np.nan)
        X_test[c] = X_test[c].replace([np.inf, -np.inf], np.nan)

    # Categorical cleanup: use only categories observed in train, plus "missing".
    for c in cat_cols:
        X_train[c] = (
            X_train[c].astype("object").where(X_train[c].notna(), "missing").astype(str)
        )
        X_test[c] = (
            X_test[c].astype("object").where(X_test[c].notna(), "missing").astype(str)
        )

        cats = sorted(set(X_train[c].unique()) | {"missing"})
        X_train[c] = pd.Categorical(X_train[c], categories=cats)
        X_test[c] = pd.Categorical(
            X_test[c].where(X_test[c].isin(cats), "missing"),
            categories=cats,
        )

    # Drop features with no observed values in training.
    feature_cols = [c for c in feature_cols if X_train[c].notna().any()]
    X_train = X_train[feature_cols].copy()
    X_test = X_test[feature_cols].copy()
    cat_cols = [c for c in cat_cols if c in feature_cols]

    # Drop rows with missing target.
    valid = y.notna()
    y = y[valid].reset_index(drop=True)
    X_train = X_train.loc[valid].reset_index(drop=True)

    return X_train, X_test, y, cat_cols


def main():
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    X_train, X_test, y, cat_cols = prepare(train, test)

    params = {
        "objective": "regression_l2",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": 42,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X_train))
    best_iters = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train), start=1):
        dtrain = lgb.Dataset(
            X_train.iloc[tr_idx],
            label=y.iloc[tr_idx],
            categorical_feature=cat_cols,
            free_raw_data=False,
        )
        dvalid = lgb.Dataset(
            X_train.iloc[va_idx],
            label=y.iloc[va_idx],
            categorical_feature=cat_cols,
            reference=dtrain,
            free_raw_data=False,
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
        )

        best_iters.append(model.best_iteration)
        pred = model.predict(X_train.iloc[va_idx], num_iteration=model.best_iteration)
        pred = np.clip(pred, 0.0, 100.0)
        oof[va_idx] = pred

        fold_rmse = root_mean_squared_error(y.iloc[va_idx], pred)
        print(f"Fold {fold} RMSE: {fold_rmse:.4f}")

    cv_rmse = root_mean_squared_error(y, oof)
    print(f"CV RMSE: {cv_rmse:.4f}")

    positive_iters = [b for b in best_iters if b is not None and b > 0]
    final_rounds = int(np.median(positive_iters)) if positive_iters else 1000
    final_rounds = max(1, final_rounds)

    dtrain_all = lgb.Dataset(
        X_train,
        label=y,
        categorical_feature=cat_cols,
        free_raw_data=False,
    )

    final_model = lgb.train(
        params,
        dtrain_all,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_pred = final_model.predict(X_test, num_iteration=final_rounds)
    test_pred = np.clip(test_pred, 0.0, 100.0)

    os.makedirs(OUT_DIR, exist_ok=True)
    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_pred)),
            TARGET: test_pred,
        }
    )
    submission.to_csv(os.path.join(OUT_DIR, "submission.csv"), index=False)
    print(f"Saved submission to {os.path.join(OUT_DIR, 'submission.csv')}")


if __name__ == "__main__":
    main()
