import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score


def clean_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    X = df.reindex(columns=feature_cols)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)
    return X.astype(np.float64)


def decision_to_bounded_probability(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    return 0.5 * (1.0 + scores / (1.0 + np.abs(scores)))


def main() -> None:
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"
    output_dir = "./working"
    target = "class"

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    train = train.dropna(subset=[target])
    y = train[target].astype(int).to_numpy()

    id_like_columns = {"id", "index", "row_id", "sample_id"}
    feature_cols = [
        col
        for col in train.columns
        if col != target and str(col).lower() not in id_like_columns
    ]

    X_train = clean_features(train, feature_cols)

    test = test.drop(columns=[target], errors="ignore")
    X_test = clean_features(test, feature_cols)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(C=1.0, max_iter=10000, random_state=42)),
        ]
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_scores = np.zeros(len(X_train), dtype=np.float64)

    for train_idx, val_idx in skf.split(X_train, y):
        X_train_fold = X_train.iloc[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X_train.iloc[val_idx]

        model.fit(X_train_fold, y_train_fold)
        oof_scores[val_idx] = model.decision_function(X_val_fold)

    if len(np.unique(y)) < 2:
        cv_auc = 1.0
    else:
        cv_auc = roc_auc_score(y, oof_scores)

    print(f"Validation Metric: {cv_auc:.6f}")

    model.fit(X_train, y)
    test_scores = model.decision_function(X_test)
    test_predictions = decision_to_bounded_probability(test_scores)

    os.makedirs(output_dir, exist_ok=True)
    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_predictions), dtype=int),
            "class": test_predictions,
        }
    )
    submission.to_csv(os.path.join(output_dir, "submission.csv"), index=False)


if __name__ == "__main__":
    main()
