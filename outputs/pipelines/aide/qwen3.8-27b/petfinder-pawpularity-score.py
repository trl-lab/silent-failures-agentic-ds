import os
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")


def main() -> None:
    os.makedirs("./working", exist_ok=True)

    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    target = "Pawpularity"
    drop_names = {
        "id",
        "pawpularity",
        "image",
        "image_path",
        "img",
        "photo",
        "url",
        "link",
        "file",
        "path",
        "filename",
    }

    train[target] = pd.to_numeric(train[target], errors="coerce")
    train = train[train[target].notna()].reset_index(drop=True)
    y = train[target].astype(float)

    common_cols = [
        c
        for c in train.columns
        if c in test.columns and c.lower() not in drop_names and c != target
    ]

    X = train[common_cols].copy()
    X_test = test[common_cols].copy()

    constant_cols = [c for c in common_cols if X[c].nunique(dropna=True) <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)
        X_test = X_test.drop(columns=constant_cols)
        common_cols = [c for c in common_cols if c not in constant_cols]

    cat_cols = []
    num_cols = []

    for c in common_cols:
        s = X[c]

        if s.dtype == bool:
            X[c] = s.astype(int)
            X_test[c] = X_test[c].astype(int)
            num_cols.append(c)
            continue

        if s.dtype.kind in "iuf":
            num_cols.append(c)
            continue

        if s.dtype.kind == "M":
            X[c] = pd.to_numeric(s.astype(str), errors="coerce")
            X_test[c] = pd.to_numeric(X_test[c].astype(str), errors="coerce")
            num_cols.append(c)
            continue

        if s.nunique(dropna=True) > 1000:
            X = X.drop(columns=[c])
            X_test = X_test.drop(columns=[c])
            common_cols.remove(c)
            continue

        converted = pd.to_numeric(s, errors="coerce")
        non_null = int(s.notna().sum())

        if non_null > 0 and int(converted.notna().sum()) >= 0.9 * non_null:
            X[c] = converted
            X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
            num_cols.append(c)
        else:
            cat_cols.append(c)

    for c in cat_cols:
        X[c] = X[c].fillna("missing").astype(str)
        X_test[c] = X_test[c].fillna("missing").astype(str)

        categories = sorted(X[c].unique())
        mapping = {value: idx for idx, value in enumerate(categories)}

        X[c] = X[c].map(mapping).astype(int)
        X_test[c] = X_test[c].map(mapping).fillna(-1).astype(int)

    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce")

    all_nan_cols = [c for c in num_cols if X[c].isna().all()]
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)
        X_test = X_test.drop(columns=all_nan_cols)
        num_cols = [c for c in num_cols if c not in all_nan_cols]

    if X.shape[1] == 0 or len(X) == 0:
        mean_target = float(y.mean()) if len(y) > 0 else 50.0
        predictions = np.full(len(test), mean_target)
        submission = pd.DataFrame(
            {
                "id": np.arange(len(test)),
                "Pawpularity": predictions,
            }
        )
        submission.to_csv("./working/submission.csv", index=False)
        print("CV RMSE: nan")
        return

    X = X.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    params = {
        "objective": "regression_l2",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": 42,
        "n_jobs": -1,
    }

    num_boost_round = 3000
    early_stopping_rounds = 50

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []
    best_iterations = []

    for train_idx, val_idx in kf.split(X):
        dtrain = lgb.Dataset(
            X.iloc[train_idx],
            label=y.iloc[train_idx],
            categorical_feature=cat_cols,
            free_raw_data=False,
        )
        dval = lgb.Dataset(
            X.iloc[val_idx],
            label=y.iloc[val_idx],
            categorical_feature=cat_cols,
            reference=dtrain,
            free_raw_data=False,
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds),
                lgb.log_evaluation(-1),
            ],
        )

        val_pred = model.predict(X.iloc[val_idx], num_iteration=model.best_iteration)
        rmse_scores.append(np.sqrt(mean_squared_error(y.iloc[val_idx], val_pred)))
        best_iterations.append(model.best_iteration)

    cv_rmse = float(np.mean(rmse_scores))
    print(f"CV RMSE: {cv_rmse:.6f}")

    final_rounds = max(100, int(np.mean(best_iterations)))

    dtrain_full = lgb.Dataset(
        X,
        label=y,
        categorical_feature=cat_cols,
        free_raw_data=False,
    )

    final_model = lgb.train(
        params,
        dtrain_full,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_pred = final_model.predict(X_test, num_iteration=final_rounds)
    test_pred = np.clip(test_pred, 0.0, 100.0)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test)),
            "Pawpularity": test_pred,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
