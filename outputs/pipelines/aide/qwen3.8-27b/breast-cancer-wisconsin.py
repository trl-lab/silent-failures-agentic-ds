import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def encode_diagnosis(value):
    s = str(value).strip().lower()
    if s in {"m", "malignant", "1", "true", "yes"} or s.startswith("m"):
        return 1
    if s in {"b", "benign", "0", "false", "no"} or s.startswith("b"):
        return 0
    return np.nan


def clean_features(df, feature_cols, medians=None):
    out = df.reindex(columns=feature_cols).copy()
    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)

    if medians is None:
        medians = out.median().fillna(0.0)

    out = out.fillna(medians)
    return out, medians


def positive_proba(model, X):
    proba = model.predict_proba(X)
    classes = model.classes_
    if 1 in classes:
        idx = np.where(classes == 1)[0][0]
        return proba[:, idx]
    return np.zeros(len(X), dtype=float)


def main():
    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    train = train.drop(columns=["ID"], errors="ignore")
    test = test.drop(columns=["ID"], errors="ignore")

    y = train["Diagnosis"].map(encode_diagnosis)
    valid = y.notna()

    train = train[valid].reset_index(drop=True)
    y = y[valid].reset_index(drop=True).astype(int)

    feature_cols = [c for c in train.columns if c != "Diagnosis"]
    X_raw = train[feature_cols].copy()
    X_test_raw = test.reindex(columns=feature_cols).copy()

    if len(X_raw) == 0 or y.nunique() < 2:
        val_auc = 0.5
        test_proba = np.full(len(X_test_raw), 0.5)
        print(f"Validation ROC AUC: {val_auc:.6f}")

        os.makedirs("./working", exist_ok=True)
        submission = pd.DataFrame(
            {
                "id": np.arange(len(test_proba), dtype=int),
                "Diagnosis": test_proba,
            }
        )
        submission.to_csv("./working/submission.csv", index=False)
        return

    min_class_count = int(y.value_counts().min())
    n_splits = min(5, min_class_count)

    if n_splits < 2:
        val_auc = 0.5
        test_proba = np.full(len(X_test_raw), 0.5)
        print(f"Validation ROC AUC: {val_auc:.6f}")

        os.makedirs("./working", exist_ok=True)
        submission = pd.DataFrame(
            {
                "id": np.arange(len(test_proba), dtype=int),
                "Diagnosis": test_proba,
            }
        )
        submission.to_csv("./working/submission.csv", index=False)
        return

    _, full_medians = clean_features(X_raw, feature_cols)
    X_test_clean, _ = clean_features(X_test_raw, feature_cols, medians=full_medians)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(X_raw), dtype=float)
    test_proba = np.zeros(len(X_test_raw), dtype=float)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y), start=1):
        X_tr, fold_medians = clean_features(X_raw.iloc[train_idx], feature_cols)
        X_val, _ = clean_features(
            X_raw.iloc[val_idx], feature_cols, medians=fold_medians
        )

        model = RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_tr, y.iloc[train_idx])

        oof[val_idx] = positive_proba(model, X_val)
        test_proba += positive_proba(model, X_test_clean) / n_splits

        try:
            fold_auc = roc_auc_score(y.iloc[val_idx], oof[val_idx])
            print(f"Fold {fold} AUC: {fold_auc:.6f}")
        except ValueError:
            print(f"Fold {fold} AUC: nan")

    val_auc = roc_auc_score(y, oof)
    print(f"Validation ROC AUC: {val_auc:.6f}")

    test_proba = np.clip(test_proba, 0.0, 1.0)

    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_proba), dtype=int),
            "Diagnosis": test_proba,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
