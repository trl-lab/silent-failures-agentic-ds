import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def stable_sigmoid(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def make_pipeline():
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(C=0.5, max_iter=100000, random_state=42)),
        ]
    )


def main():
    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    y = train["target"].astype(float)
    if not np.all(np.isin(y, [0.0, 1.0])):
        y = (y > y.median()).astype(float)
    y = y.astype(int)

    feature_cols = [c for c in train.columns if c not in ("target", "id")]
    X = train[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    all_nan_cols = X.columns[X.isna().all()]
    if len(all_nan_cols) > 0:
        X = X.drop(columns=all_nan_cols)

    feature_cols = X.columns.tolist()

    X_test = test.reindex(columns=feature_cols, fill_value=np.nan)
    X_test = X_test.apply(pd.to_numeric, errors="coerce")
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    if X.shape[1] == 0 or len(np.unique(y)) < 2:
        cv_auc = 0.5
        test_scores = np.zeros(len(X_test), dtype=float)
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_aucs = []

        for train_idx, val_idx in skf.split(X, y):
            pipe = make_pipeline()
            pipe.fit(X.iloc[train_idx], y.iloc[train_idx])

            val_score = np.asarray(
                pipe.decision_function(X.iloc[val_idx]), dtype=float
            ).ravel()
            val_true = y.iloc[val_idx]

            if len(np.unique(val_true)) < 2:
                fold_aucs.append(0.5)
            else:
                fold_aucs.append(roc_auc_score(val_true, val_score))

        cv_auc = float(np.mean(fold_aucs))

        pipe = make_pipeline()
        pipe.fit(X, y)
        test_scores = np.asarray(pipe.decision_function(X_test), dtype=float).ravel()

    print(f"{cv_auc:.6f}")

    os.makedirs("./working", exist_ok=True)

    if X.shape[1] == 0 or len(np.unique(y)) < 2:
        pred = np.full(len(X_test), 0.5, dtype=float)
    else:
        pred = stable_sigmoid(test_scores)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(pred), dtype=int),
            "target": pred,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
