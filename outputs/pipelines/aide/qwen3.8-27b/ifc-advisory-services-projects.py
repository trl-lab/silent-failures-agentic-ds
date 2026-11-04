import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

TARGET = "Estimated Total Budget ($)"
SEED = 42


def is_id_like(col):
    s = str(col).strip().lower()
    return s in {
        "id",
        "index",
        "row_id",
        "rowid",
        "identifier",
        "project_id",
        "projectid",
        "project id",
        "case_id",
        "caseid",
    }


def clean_pair(train, test, target_col):
    y = train[target_col].copy()
    train = train.drop(columns=[target_col])
    if target_col in test.columns:
        test = test.drop(columns=[target_col])

    train = train.loc[:, ~train.columns.duplicated()]
    test = test.loc[:, ~test.columns.duplicated()]

    train = train.drop(
        columns=[c for c in train.columns if is_id_like(c)], errors="ignore"
    )
    test = test.drop(
        columns=[c for c in test.columns if is_id_like(c)], errors="ignore"
    )

    features = list(dict.fromkeys([c for c in train.columns if c in test.columns]))
    if not features:
        features = list(train.columns)
        for c in features:
            if c not in test.columns:
                test[c] = np.nan

    features = [c for c in features if train[c].notna().any()]
    train = train[features].copy()
    test = test[features].copy()

    y = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mask = y.notna()
    y = y[mask].astype(float)
    train = train.loc[mask].copy().reset_index(drop=True)
    y = y.reset_index(drop=True)
    test = test.reset_index(drop=True)

    categorical_cols = []
    for c in features:
        if c not in train.columns or c not in test.columns:
            continue
        try:
            if pd.api.types.is_datetime64_any_dtype(train[c]):
                train[c] = (
                    pd.to_datetime(train[c], errors="coerce")
                    - pd.Timestamp("1970-01-01")
                ).dt.total_seconds()
                test[c] = (
                    pd.to_datetime(test[c], errors="coerce")
                    - pd.Timestamp("1970-01-01")
                ).dt.total_seconds()
                train[c] = pd.to_numeric(train[c], errors="coerce")
                test[c] = pd.to_numeric(test[c], errors="coerce")
            elif pd.api.types.is_bool_dtype(train[c]):
                train[c] = train[c].astype(int)
                test[c] = test[c].astype(int)
            elif pd.api.types.is_numeric_dtype(train[c]):
                train[c] = pd.to_numeric(train[c], errors="coerce")
                test[c] = pd.to_numeric(test[c], errors="coerce")
                train[c] = train[c].replace([np.inf, -np.inf], np.nan)
                test[c] = test[c].replace([np.inf, -np.inf], np.nan)
            else:
                vals = train[c].dropna().unique()
                mapping = {v: i for i, v in enumerate(vals)}
                train[c] = (
                    train[c]
                    .map(lambda x: mapping.get(x, -1) if pd.notna(x) else -1)
                    .astype(int)
                )
                test[c] = (
                    test[c]
                    .map(lambda x: mapping.get(x, -1) if pd.notna(x) else -1)
                    .astype(int)
                )
                categorical_cols.append(c)
        except Exception:
            train[c] = np.nan
            test[c] = np.nan

    keep = []
    for c in features:
        try:
            if c in train.columns and train[c].nunique(dropna=True) > 1:
                keep.append(c)
        except Exception:
            pass

    features = keep
    categorical_cols = [c for c in categorical_cols if c in features]
    train = train[features].copy()
    test = test[features].copy()

    return train, test, y, categorical_cols


def save_submission(test_pred, target_col, n_rows):
    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame(
        {
            "id": np.arange(n_rows, dtype=int),
            target_col: np.asarray(test_pred, dtype=float),
        }
    )
    submission.to_csv("./working/submission.csv", index=False)


def main():
    os.makedirs("./working", exist_ok=True)

    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    target_col = (
        TARGET
        if TARGET in train.columns
        else next(
            (c for c in train.columns if "Estimated Total Budget" in str(c)),
            None,
        )
    )
    if target_col is None:
        raise ValueError("Target column not found in train.parquet")

    X, X_test, y, categorical_cols = clean_pair(train, test, target_col)

    if len(y) == 0:
        save_submission(np.zeros(len(X_test)), target_col, len(X_test))
        print("CV RMSE: nan")
        return

    if len(y) < 2 or len(X.columns) == 0:
        pred = float(y.mean())
        save_submission(np.full(len(X_test), pred), target_col, len(X_test))
        print("CV RMSE: nan")
        return

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 10,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "seed": SEED,
    }

    n_splits = 5 if len(y) >= 5 else 2
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=float)
    best_iters = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
        dtrain = lgb.Dataset(
            X.iloc[tr_idx],
            label=y.iloc[tr_idx],
            categorical_feature=categorical_cols if categorical_cols else None,
        )
        dval = lgb.Dataset(
            X.iloc[va_idx],
            label=y.iloc[va_idx],
            reference=dtrain,
            categorical_feature=categorical_cols if categorical_cols else None,
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            valid_sets=[dval],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        best_iters.append(max(1, model.best_iteration))
        pred = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
        oof[va_idx] = pred

        fold_rmse = np.sqrt(mean_squared_error(y.iloc[va_idx], pred))
        print(f"Fold {fold} RMSE: {fold_rmse:.6f}")

    cv_rmse = np.sqrt(mean_squared_error(y, oof))
    print(f"CV RMSE: {cv_rmse:.6f}")

    if len(X_test) == 0:
        save_submission(np.array([], dtype=float), target_col, 0)
        return

    dtrain_all = lgb.Dataset(
        X,
        label=y,
        categorical_feature=categorical_cols if categorical_cols else None,
    )

    final_rounds = max(50, int(np.mean(best_iters))) if best_iters else 500
    final_model = lgb.train(
        params,
        dtrain_all,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_pred = final_model.predict(X_test, num_iteration=final_rounds)
    test_pred = np.asarray(test_pred, dtype=float)
    test_pred = np.nan_to_num(
        test_pred,
        nan=float(y.mean()),
        posinf=float(y.max()),
        neginf=float(y.min()),
    )

    if y.min() >= 0:
        test_pred = np.maximum(test_pred, 0.0)

    save_submission(test_pred, target_col, len(X_test))
    print(f"Saved ./working/submission.csv with {len(X_test)} rows.")


if __name__ == "__main__":
    main()
