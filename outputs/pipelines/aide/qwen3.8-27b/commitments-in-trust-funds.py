import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

TARGET = "New Commitments (US$)"
SEED = 42


def _to_str_object(series: pd.Series) -> pd.Series:
    obj = series.astype(object).where(series.notna(), None)
    return obj.map(lambda x: None if x is None else str(x))


def main():
    os.makedirs("./working", exist_ok=True)

    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    train = train.copy()
    train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
    train[TARGET] = train[TARGET].replace([np.inf, -np.inf], np.nan)
    train = train.dropna(subset=[TARGET])
    y = train[TARGET].astype(float)

    if y.empty:
        print("Validation RMSE: nan")
        pd.DataFrame(
            {
                "id": np.arange(len(test), dtype=int),
                TARGET: np.zeros(len(test), dtype=float),
            }
        ).to_csv("./working/submission.csv", index=False)
        return

    nonnegative_target = bool(y.min() >= 0)

    drop_names = {TARGET}
    drop_names.update(
        [
            c
            for c in train.columns
            if str(c).strip().lower() in {"id", "index", "row_id", "rowid", "idx"}
        ]
    )

    common = [c for c in train.columns if c in test.columns and c not in drop_names]

    X_train = train[common].copy()
    X_test = test[common].copy()

    for c in common:
        if pd.api.types.is_numeric_dtype(X_train[c]) or pd.api.types.is_numeric_dtype(
            X_test[c]
        ):
            X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
            X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
        else:
            X_train[c] = _to_str_object(X_train[c])
            X_test[c] = _to_str_object(X_test[c])

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    for c in numeric_cols:
        X_train[c] = X_train[c].replace([np.inf, -np.inf], np.nan)
        X_test[c] = X_test[c].replace([np.inf, -np.inf], np.nan)

    n_rows = len(X_train)
    keep_cols = []
    for c in X_train.columns:
        nuniq = X_train[c].nunique(dropna=True)
        if nuniq <= 1:
            continue
        if X_train[c].dtype == object and nuniq > 10 and nuniq > 0.95 * n_rows:
            continue
        keep_cols.append(c)

    X_train = X_train[keep_cols]
    X_test = X_test[keep_cols]

    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_cols:
        X_train[c] = _to_str_object(X_train[c])
        X_test[c] = _to_str_object(X_test[c])

    if X_train.shape[1] == 0 or y.nunique() == 0:
        val_rmse = (
            0.0 if y.nunique() == 0 else float(np.sqrt(np.mean((y - y.mean()) ** 2)))
        )
        print(f"Validation RMSE: {val_rmse:.6f}")

        pred = np.full(len(test), y.mean(), dtype=float)
        if nonnegative_target:
            pred = np.clip(pred, 0.0, None)

        pd.DataFrame({"id": np.arange(len(test), dtype=int), TARGET: pred}).to_csv(
            "./working/submission.csv", index=False
        )
        return

    if len(X_train) < 2:
        val_rmse = float(np.sqrt(np.mean((y - y.mean()) ** 2)))
        print(f"Validation RMSE: {val_rmse:.6f}")

        pred = np.full(len(test), y.mean(), dtype=float)
        if nonnegative_target:
            pred = np.clip(pred, 0.0, None)

        pd.DataFrame({"id": np.arange(len(test), dtype=int), TARGET: pred}).to_csv(
            "./working/submission.csv", index=False
        )
        return

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "seed": SEED,
        "verbosity": -1,
    }

    n_splits = 5 if len(X_train) >= 5 else len(X_train)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    val_scores = []
    best_iterations = []

    for tr_idx, va_idx in kf.split(X_train):
        dtrain = lgb.Dataset(
            X_train.iloc[tr_idx], y.iloc[tr_idx], categorical_feature=cat_cols
        )
        dvalid = lgb.Dataset(
            X_train.iloc[va_idx],
            y.iloc[va_idx],
            categorical_feature=cat_cols,
            reference=dtrain,
        )

        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
        )

        best_iter = int(getattr(booster, "best_iteration", 0) or 0)
        if best_iter <= 0:
            best_iter = 1
        best_iterations.append(best_iter)

        val_pred = booster.predict(X_train.iloc[va_idx], num_iteration=best_iter)
        val_pred = np.asarray(val_pred, dtype=float).reshape(-1)
        val_pred = np.where(np.isfinite(val_pred), val_pred, y.iloc[va_idx].mean())

        if nonnegative_target:
            val_pred = np.clip(val_pred, 0.0, None)

        val_scores.append(root_mean_squared_error(y.iloc[va_idx], val_pred))

    val_rmse = float(np.mean(val_scores))
    print(f"Validation RMSE: {val_rmse:.6f}")

    final_rounds = int(np.clip(np.mean(best_iterations), 1, 5000))

    dtrain_full = lgb.Dataset(X_train, y, categorical_feature=cat_cols)
    final_booster = lgb.train(
        params,
        dtrain_full,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    if len(X_test) == 0:
        test_pred = np.array([], dtype=float)
    else:
        test_pred = final_booster.predict(X_test, num_iteration=final_rounds)

    test_pred = np.asarray(test_pred, dtype=float).reshape(-1)
    if nonnegative_target:
        test_pred = np.clip(test_pred, 0.0, None)
    test_pred = np.where(np.isfinite(test_pred), test_pred, y.mean())

    pd.DataFrame({"id": np.arange(len(test), dtype=int), TARGET: test_pred}).to_csv(
        "./working/submission.csv", index=False
    )


if __name__ == "__main__":
    main()
