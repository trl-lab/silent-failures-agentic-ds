import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

NUM_CLASSES = 4
LABELS = np.array([1, 2, 3, 4], dtype=float)


def qwk(y_true, y_pred, num_classes=NUM_CLASSES):
    y_true = np.clip(np.asarray(y_true, dtype=int) - 1, 0, num_classes - 1)
    y_pred = np.clip(np.asarray(y_pred, dtype=int) - 1, 0, num_classes - 1)

    idx = np.arange(num_classes)
    W = ((idx[:, None] - idx[None, :]) ** 2) / float((num_classes - 1) ** 2)

    C = np.zeros((num_classes, num_classes), dtype=float)
    np.add.at(C, (y_true, y_pred), 1.0)

    total = C.sum()
    if total == 0:
        return 1.0

    O = np.outer(C.sum(axis=1), C.sum(axis=0)) / total
    denom = np.sum(W * O)

    if denom < 1e-12:
        return 1.0

    return 1.0 - np.sum(W * C) / denom


def probs_to_response(probs):
    score = np.asarray(probs, dtype=float) @ LABELS
    score = np.nan_to_num(score, nan=2.0, posinf=4.0, neginf=1.0)
    return np.clip(np.floor(score + 0.5).astype(int), 1, 4)


def main():
    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    train["Response"] = pd.to_numeric(train["Response"], errors="coerce")
    train = train.dropna(subset=["Response"]).reset_index(drop=True)
    test = test.reset_index(drop=True)

    exclude = {"id", "response"}
    feature_cols = [
        c for c in train.columns if c.lower() not in exclude and c in test.columns
    ]

    X = train[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_test = test[feature_cols].apply(pd.to_numeric, errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    X = X.loc[:, X.dropna(axis=1, how="all").columns]
    X = X.loc[:, X.nunique() > 1]
    X_test = X_test[X.columns]

    y = train["Response"].astype(int).to_numpy()
    y_cls = np.clip(y - 1, 0, NUM_CLASSES - 1).astype(int)

    params = {
        "objective": "multiclass",
        "num_class": NUM_CLASSES,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 30,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "seed": 42,
        "num_threads": -1,
        "verbosity": -1,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    best_iters = []
    num_boost_round = 1500

    for tr_idx, va_idx in skf.split(X, y_cls):
        dtrain = lgb.Dataset(X.iloc[tr_idx], label=y_cls[tr_idx])
        dvalid = lgb.Dataset(X.iloc[va_idx], label=y_cls[va_idx], reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        best_iter = model.best_iteration
        if best_iter is None or best_iter <= 0:
            best_iter = num_boost_round
        best_iters.append(best_iter)

        val_probs = model.predict(X.iloc[va_idx], num_iteration=best_iter)
        val_pred = probs_to_response(val_probs)
        fold_scores.append(qwk(y[va_idx], val_pred, NUM_CLASSES))

    cv_qwk = float(np.mean(fold_scores))
    print(f"CV QWK: {cv_qwk:.5f}")

    final_rounds = min(num_boost_round, max(1, int(np.median(best_iters) * 1.05) + 1))

    dtrain_full = lgb.Dataset(X, label=y_cls)
    final_model = lgb.train(params, dtrain_full, num_boost_round=final_rounds)

    test_probs = final_model.predict(X_test, num_iteration=final_rounds)
    test_pred = probs_to_response(test_probs)

    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame(
        {
            "id": np.arange(len(X_test), dtype=int),
            "Response": test_pred,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
