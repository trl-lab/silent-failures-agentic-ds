import os
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    os.makedirs("./working", exist_ok=True)

    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    target = "Activity"
    y_raw = train[target].astype(float)
    X_train = train.drop(columns=[target]).copy()
    X_test = test.copy()

    # Keep only numeric feature columns present in both train and test.
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    # Coerce to numeric and replace infinities with NaN.
    for col in X_train.columns:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    # Remove uninformative columns using training-set statistics only.
    n_train = len(X_train)
    keep_cols = []
    for col in X_train.columns:
        series = X_train[col]
        finite_mask = np.isfinite(series)
        finite_count = int(finite_mask.sum())

        # Require at least two finite values and not too much missingness.
        if finite_count < 2:
            continue
        if finite_count / n_train < 0.05:
            continue

        finite_values = series[finite_mask]
        std = finite_values.std(ddof=0)
        if std is None or not np.isfinite(std) or std <= 1e-12:
            continue

        keep_cols.append(col)

    if len(keep_cols) == 0:
        # Fallback: if every column is degenerate, use the base rate.
        base_rate = float(y_raw.mean())
        cv_log_loss = log_loss(
            (y_raw == base_rate).astype(int), np.full(len(y_raw), base_rate)
        )
        print(f"CV log_loss: {cv_log_loss:.6f}")
        submission = pd.DataFrame(
            {"id": np.arange(len(test)), "Activity": np.full(len(test), base_rate)}
        )
        submission.to_csv("./working/submission.csv", index=False)
        return

    X_train = X_train[keep_cols]
    X_test = X_test[keep_cols]

    # The competition target is binary in the provided memory; map to 0/1 if needed.
    classes = np.unique(y_raw)
    if len(classes) != 2:
        raise ValueError("Expected a binary Activity target.")

    positive_class = max(classes)
    y = (y_raw == positive_class).astype(int)

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=10000,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # 5-fold stratified cross-validation with log-loss on held-out folds.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for train_idx, val_idx in skf.split(X_train, y):
        X_tr = X_train.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y.iloc[val_idx]

        model = clone(pipeline)
        model.fit(X_tr, y_tr)

        proba = model.predict_proba(X_val)
        positive_idx = list(model.named_steps["classifier"].classes_).index(1)
        val_positive_proba = proba[:, positive_idx]

        fold_scores.append(log_loss(y_val, val_positive_proba))

    cv_log_loss = float(np.mean(fold_scores))
    print(f"CV log_loss: {cv_log_loss:.6f}")

    # Refit on the full training set and produce test predictions.
    final_model = clone(pipeline)
    final_model.fit(X_train, y)

    test_proba = final_model.predict_proba(X_test)
    positive_idx = list(final_model.named_steps["classifier"].classes_).index(1)
    test_positive_proba = test_proba[:, positive_idx]
    test_positive_proba = np.clip(test_positive_proba, 1e-15, 1.0 - 1e-15)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test)),
            "Activity": test_positive_proba,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
