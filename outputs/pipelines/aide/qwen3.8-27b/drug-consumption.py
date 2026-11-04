import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score

TARGET = "cannabis"
TARGET_COLS = [
    "alcohol",
    "amphet",
    "amyl",
    "benzos",
    "caff",
    "cannabis",
    "choc",
    "coke",
    "crack",
    "ecstasy",
    "heroin",
    "ketamine",
    "legalh",
    "lsd",
    "meth",
    "mushrooms",
    "nicotine",
    "semer",
    "vsa",
]


def main():
    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    feature_cols = [c for c in train.columns if c != "id" and c not in TARGET_COLS]

    train = train.dropna(subset=[TARGET])
    y = train[TARGET].astype(int)
    X = train[feature_cols].copy()
    X_test = test.reindex(columns=feature_cols).copy()

    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    X = X.dropna(axis=1, how="all")
    feature_cols = X.columns.tolist()
    X_test = X_test.reindex(columns=feature_cols)

    def save_predictions(pred):
        os.makedirs("./working", exist_ok=True)
        submission = pd.DataFrame(
            {
                "id": np.arange(len(test)),
                "cannabis": pred,
            }
        )
        submission.to_csv("./working/submission.csv", index=False)

    if len(y) == 0:
        print("CV accuracy: 0.0000 (+/- 0.0000)")
        save_predictions(np.zeros(len(test), dtype=int))
        return

    if y.nunique() <= 1:
        print("CV accuracy: 1.0000 (+/- 0.0000)")
        save_predictions(np.full(len(test), int(y.iloc[0]), dtype=int))
        return

    if not feature_cols:
        n_folds = min(5, len(y))
        skf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        accs = []
        for tr_idx, va_idx in skf.split(np.arange(len(y)), y):
            majority = int(y.iloc[tr_idx].mode().iloc[0])
            accs.append(accuracy_score(y.iloc[va_idx], majority))
        print(f"CV accuracy: {np.mean(accs):.4f} (+/- {np.std(accs):.4f})")
        save_predictions(np.full(len(test), int(y.mode().iloc[0]), dtype=int))
        return

    class_counts = y.value_counts()
    min_count = int(class_counts.min())

    try:
        if min_count >= 5:
            n_splits = 5
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = list(splitter.split(X, y))
        elif min_count >= 2:
            n_splits = min(5, min_count)
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = list(splitter.split(X, y))
        else:
            n_splits = min(5, len(y))
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = list(splitter.split(X, y))
    except Exception:
        n_splits = min(5, len(y))
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(splitter.split(X, y))

    num_classes = int(y.nunique())
    base_params = dict(
        objective="multiclass" if num_classes > 2 else "binary",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=7,
        max_depth=4,
        min_child_samples=10,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        eval_metric="multi_logloss" if num_classes > 2 else "binary_logloss",
    )

    accs = []
    best_iters = []

    for tr_idx, va_idx in splits:
        model = lgb.LGBMClassifier(**base_params)
        model.fit(
            X.iloc[tr_idx],
            y.iloc[tr_idx],
            eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        pred = model.predict(X.iloc[va_idx])
        accs.append(accuracy_score(y.iloc[va_idx], pred))
        best_iters.append(getattr(model, "best_iteration_", 0) or 0)

    print(f"CV accuracy: {np.mean(accs):.4f} (+/- {np.std(accs):.4f})")

    valid_iters = [b for b in best_iters if b and b > 0]
    best_n = int(np.median(valid_iters)) if valid_iters else 300
    best_n = int(max(50, min(best_n, 1000)))

    final_model = lgb.LGBMClassifier(**{**base_params, "n_estimators": best_n})
    final_model.fit(X, y)

    pred_test = final_model.predict(X_test)
    try:
        pred_test = pred_test.astype(int)
    except Exception:
        pass

    save_predictions(pred_test)


if __name__ == "__main__":
    main()
