import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def _monotone_constraints_for_features(feature_cols):
    constraints = [0] * len(feature_cols)
    for i, col in enumerate(feature_cols):
        name = str(col).strip().lower()
        if name == "v1" or "recency" in name:
            constraints[i] = -1
        elif name == "v2" or "frequency" in name:
            constraints[i] = 1
        elif name == "v3" or "monetary" in name:
            constraints[i] = 1
    return constraints


def _write_submission(pred, n_rows):
    pred = np.clip(np.asarray(pred, dtype=float), 0.0, 1.0)
    submission = pd.DataFrame({"id": np.arange(n_rows, dtype=int), "Class": pred})
    submission.to_csv("./working/submission.csv", index=False)


def main():
    os.makedirs("./working", exist_ok=True)

    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    train["Class"] = pd.to_numeric(train["Class"], errors="coerce")
    train = train.dropna(subset=["Class"]).reset_index(drop=True)

    if train.empty:
        _write_submission(np.full(len(test), 0.5), len(test))
        print("0.500000")
        return

    classes = set(train["Class"].unique())
    if 2 in classes and (1 in classes or 0 in classes):
        y = (train["Class"] == 2).astype(int)
    elif 1 in classes and 0 in classes:
        y = (train["Class"] == 1).astype(int)
    else:
        y = train["Class"].astype(int)

    feature_cols = [c for c in train.columns if c not in ("Class", "id")]
    feature_cols = [c for c in feature_cols if c in test.columns]

    X = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    feature_cols = [
        c for c in feature_cols if X[c].notna().any() or X_test[c].notna().any()
    ]
    X = X[feature_cols]
    X_test = X_test[feature_cols]

    if len(feature_cols) == 0:
        pred = np.full(len(X_test), float(y.mean()) if len(y) else 0.5)
        _write_submission(pred, len(X_test))
        print("0.500000")
        return

    X = X.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    medians = X.median(numeric_only=True)
    X = X.fillna(medians).fillna(0)
    X_test = X_test.fillna(medians).fillna(0)

    if y.nunique() < 2:
        pred = np.full(len(X_test), float(y.mean()) if len(y) else 0.5)
        _write_submission(pred, len(X_test))
        print("0.500000")
        return

    min_class_count = int(y.value_counts().min())
    n_splits = min(5, min_class_count)

    if n_splits < 2:
        pred = np.full(len(X_test), float(y.mean()) if len(y) else 0.5)
        _write_submission(pred, len(X_test))
        print("0.500000")
        return

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": -1,
        "min_child_samples": 10,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "is_unbalance": True,
        "seed": 42,
        "verbose": -1,
    }

    constraints = _monotone_constraints_for_features(feature_cols)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(X), dtype=float)
    best_iters = []

    for tr_idx, va_idx in skf.split(X, y):
        dtrain = lgb.Dataset(
            X.iloc[tr_idx].values,
            label=y.iloc[tr_idx].values,
            monotone_constraints=constraints,
        )
        dvalid = lgb.Dataset(
            X.iloc[va_idx].values,
            label=y.iloc[va_idx].values,
            reference=dtrain,
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        best_iter = model.best_iteration
        if best_iter is None or best_iter <= 0:
            best_iter = model.num_trees()
        if best_iter is None or best_iter <= 0:
            best_iter = 1

        oof[va_idx] = model.predict(X.iloc[va_idx].values, num_iteration=best_iter)
        best_iters.append(int(best_iter))

    cv_auc = roc_auc_score(y, oof)
    print(f"{cv_auc:.6f}")

    final_rounds = max(1, max(best_iters)) if best_iters else 300
    dtrain_all = lgb.Dataset(X.values, label=y.values, monotone_constraints=constraints)
    final_model = lgb.train(params, dtrain_all, num_boost_round=final_rounds)

    if len(X_test) == 0:
        test_pred = np.array([], dtype=float)
    else:
        test_pred = final_model.predict(X_test.values)

    _write_submission(test_pred, len(X_test))


if __name__ == "__main__":
    main()
