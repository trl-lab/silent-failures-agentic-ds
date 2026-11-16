import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

TARGET = "Female (%)"
ID_NAMES = {"id", "idx", "index", "row_id", "rowid"}


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def prepare_data(train: pd.DataFrame, test: pd.DataFrame, target: str):
    train = train.copy()
    test = test.copy()

    train[target] = pd.to_numeric(train[target], errors="coerce")
    train = train.dropna(subset=[target]).reset_index(drop=True)
    y = train[target].astype(float)

    id_cols = {c for c in train.columns if str(c).strip().lower() in ID_NAMES}
    feature_cols = [
        c
        for c in train.columns
        if c in test.columns and c != target and c not in id_cols
    ]

    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    cat_cols = []
    drop_cols = []
    n_train = len(X_train)

    for col in feature_cols:
        s = X_train[col]

        if s.isna().all():
            drop_cols.append(col)
            continue

        if not pd.api.types.is_numeric_dtype(s):
            numeric = pd.to_numeric(s, errors="coerce")
            non_null = s.notna().sum()

            if non_null > 0 and numeric.notna().sum() >= 0.9 * non_null:
                X_train[col] = numeric
                X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
            else:
                if s.nunique(dropna=True) <= 1:
                    drop_cols.append(col)
                    continue

                if s.nunique(dropna=True) > max(500, int(0.8 * n_train)):
                    drop_cols.append(col)
                    continue

                cats = pd.unique(s.dropna())
                try:
                    mapping = {v: i for i, v in enumerate(cats)}
                    X_train[col] = X_train[col].map(mapping).fillna(-1).astype("int32")
                    X_test[col] = X_test[col].map(mapping).fillna(-1).astype("int32")
                except TypeError:
                    str_mapping = {str(v): i for i, v in enumerate(cats)}
                    X_train[col] = (
                        X_train[col]
                        .astype(str)
                        .map(str_mapping)
                        .fillna(-1)
                        .astype("int32")
                    )
                    X_test[col] = (
                        X_test[col]
                        .astype(str)
                        .map(str_mapping)
                        .fillna(-1)
                        .astype("int32")
                    )

                cat_cols.append(col)
        else:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

        if X_train[col].nunique(dropna=True) <= 1:
            drop_cols.append(col)

    drop_cols = list(set(drop_cols))
    X_train = X_train.drop(columns=drop_cols, errors="ignore")
    X_test = X_test.drop(columns=drop_cols, errors="ignore")
    cat_cols = [c for c in cat_cols if c in X_train.columns]

    for col in X_train.columns:
        if col in cat_cols:
            X_train[col] = X_train[col].fillna(-1).astype("int32")
            X_test[col] = X_test[col].fillna(-1).astype("int32")
        else:
            median = X_train[col].median()
            if pd.isna(median):
                median = 0.0
            X_train[col] = X_train[col].fillna(median)
            X_test[col] = X_test[col].fillna(median)

    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    for col in X_train.columns:
        if col not in cat_cols:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce").fillna(0)
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce").fillna(0)

    return X_train, X_test, y, cat_cols


def main():
    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    X_train, X_test, y, cat_cols = prepare_data(train, test, TARGET)

    if X_train.shape[1] == 0 or len(y) == 0:
        val_rmse = rmse(y, np.full(len(y), y.mean())) if len(y) > 0 else np.nan
        print(f"5-fold CV RMSE: {val_rmse:.6f}")
        test_pred = np.full(len(test), y.mean() if len(y) > 0 else 0.0)
    else:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "min_child_samples": 5,
            "lambda_l2": 1.0,
            "seed": 42,
            "verbose": -1,
        }

        n = len(X_train)
        n_splits = 5 if n >= 5 else (2 if n >= 2 else 1)

        rmse_scores = []
        best_iters = []

        if n_splits > 1:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

            for tr_idx, va_idx in kf.split(X_train):
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
                    free_raw_data=False,
                    reference=dtrain,
                )

                model = lgb.train(
                    params,
                    dtrain,
                    num_boost_round=5000,
                    valid_sets=[dvalid],
                    valid_names=["valid"],
                    callbacks=[
                        lgb.early_stopping(50, verbose=False),
                        lgb.log_evaluation(-1),
                    ],
                )

                best_it = model.best_iteration
                if best_it > 0:
                    pred = model.predict(X_train.iloc[va_idx], num_iteration=best_it)
                else:
                    pred = model.predict(X_train.iloc[va_idx])

                rmse_scores.append(rmse(y.iloc[va_idx], pred))
                if best_it > 0:
                    best_iters.append(best_it)
        else:
            pred = np.full(len(y), y.mean())
            rmse_scores.append(rmse(y, pred))

        val_rmse = float(np.mean(rmse_scores))
        print(f"5-fold CV RMSE: {val_rmse:.6f}")

        final_rounds = int(np.mean(best_iters)) + 50 if best_iters else 500

        dtrain_full = lgb.Dataset(
            X_train,
            label=y,
            categorical_feature=cat_cols,
        )

        final_model = lgb.train(
            params,
            dtrain_full,
            num_boost_round=final_rounds,
            callbacks=[lgb.log_evaluation(-1)],
        )

        test_pred = final_model.predict(X_test)

    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame(
        {
            "id": np.arange(len(test)),
            TARGET: test_pred,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
