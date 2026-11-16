import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def encode_target(raw):
    """Convert target values to 0/1 integers robustly."""
    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.notna().any():
        vals = numeric.dropna().unique()
        if set(vals).issubset({0, 1}):
            return numeric.fillna(0).astype(int)

    s = raw.astype(str).str.strip().str.lower()
    mapping = {
        "0": 0,
        "1": 1,
        "fail": 0,
        "failure": 0,
        "succeed": 1,
        "success": 1,
    }

    if s.isin(mapping).all():
        return s.map(mapping).astype(int)

    uniques = sorted(s.unique())
    if len(uniques) == 2:
        return s.map({uniques[0]: 0, uniques[1]: 1}).astype(int)

    return s.map({"fail": 0, "succeed": 1}).fillna(0).astype(int)


def make_svm():
    """Create the RBF SVM pipeline used in the ensemble."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=42,
                    cache_size=1000,
                ),
            ),
        ]
    )


def make_logreg():
    """Create the balanced logistic regression pipeline used in the ensemble."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=10000,
                    random_state=42,
                ),
            ),
        ]
    )


def main():
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"
    out_dir = "./working"

    os.makedirs(out_dir, exist_ok=True)

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    target_col = "outcome"
    if target_col not in train.columns:
        target_col = train.columns[-1]

    y = encode_target(train[target_col])

    train = train.drop(columns=[target_col], errors="ignore")
    test = test.drop(columns=[target_col], errors="ignore")

    exclude = {
        "id",
        "index",
        "outcome",
        "target",
        "label",
        "batch",
        "ensemble",
        "run",
        "simulation",
        "case",
        "member",
        "fold",
        "split",
    }

    common_cols = [c for c in train.columns if c in test.columns]

    features = []
    for col in common_cols:
        if str(col).lower() in exclude:
            continue

        train_numeric = pd.to_numeric(train[col], errors="coerce")
        test_numeric = pd.to_numeric(test[col], errors="coerce")

        if train_numeric.notna().any() and test_numeric.notna().any():
            features.append(col)

    # Respect the dataset note that the first two original variables are batch-like
    # and should not be used if extra numeric columns are present.
    if len(features) > 18:
        features = features[-18:]

    if not features:
        features = [c for c in common_cols if pd.api.types.is_numeric_dtype(train[c])]
        if len(features) > 18:
            features = features[-18:]

    X = train[features].apply(pd.to_numeric, errors="coerce")
    X_test = test[features].apply(pd.to_numeric, errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    val_scores = np.zeros(len(X), dtype=float)

    for train_idx, val_idx in skf.split(X, y):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]

        svm = make_svm()
        svm.fit(X_train, y_train)
        p_svm = svm.predict_proba(X_val)[:, 1]

        logreg = make_logreg()
        logreg.fit(X_train, y_train)
        p_logreg = logreg.predict_proba(X_val)[:, 1]

        val_scores[val_idx] = 0.5 * p_svm + 0.5 * p_logreg

    cv_auc = float(roc_auc_score(y, val_scores))
    print(f"5-fold CV ROC AUC: {cv_auc:.6f}")

    svm_final = make_svm()
    svm_final.fit(X, y)
    p_svm_test = svm_final.predict_proba(X_test)[:, 1]

    logreg_final = make_logreg()
    logreg_final.fit(X, y)
    p_logreg_test = logreg_final.predict_proba(X_test)[:, 1]

    test_probs = 0.5 * p_svm_test + 0.5 * p_logreg_test
    test_probs = np.nan_to_num(test_probs, nan=0.5, posinf=1.0, neginf=0.0)
    test_probs = np.clip(test_probs, 0.0, 1.0)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(X_test), dtype=int),
            "outcome": test_probs,
        }
    )

    submission.to_csv(os.path.join(out_dir, "submission.csv"), index=False)


if __name__ == "__main__":
    main()
