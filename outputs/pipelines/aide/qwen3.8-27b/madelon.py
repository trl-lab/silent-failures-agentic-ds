import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import lightgbm as lgb

TARGET = "Class"
EXCLUDE = {"Class", "id", "ID", "index", "Index", "row_id", "RowID"}


def make_lgbm():
    return lgb.LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def make_svm_pipeline(n_features: int) -> Pipeline:
    k = min(100, max(1, n_features))
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=f_classif, k=k)),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                    probability=True,
                    random_state=42,
                    cache_size=500,
                ),
            ),
        ]
    )


def clean_features(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def impute_fit_transform(X_tr: pd.DataFrame, X_other: pd.DataFrame):
    imputer = SimpleImputer(strategy="median")
    try:
        return imputer.fit_transform(X_tr), imputer.transform(X_other)
    except ValueError:
        X_tr_i = X_tr.fillna(0).astype(float)
        X_other_i = X_other.fillna(0).astype(float)
        return X_tr_i.to_numpy(), X_other_i.to_numpy()


def positive_probability(probs: np.ndarray, n_samples: int) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    if probs.ndim == 1:
        return probs
    if probs.ndim == 2:
        if probs.shape[1] == 1:
            return probs.ravel()
        return probs[:, 1]
    return np.full(n_samples, 0.5)


def main():
    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    valid = train[TARGET].notna()
    train = train.loc[valid].reset_index(drop=True)

    feature_cols = [c for c in train.columns if c not in EXCLUDE and c in test.columns]

    X = clean_features(train, feature_cols)
    X_test = clean_features(test, feature_cols)
    y = train[TARGET]

    label_encoder = LabelEncoder().fit(y)
    y_enc = label_encoder.transform(y).astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for tr_idx, va_idx in skf.split(X, y_enc):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y_enc[tr_idx], y_enc[va_idx]

        X_tr_i, X_va_i = impute_fit_transform(X_tr, X_va)

        lgbm = make_lgbm()
        lgbm.fit(X_tr_i, y_tr)

        svm = make_svm_pipeline(X_tr_i.shape[1])
        svm.fit(X_tr_i, y_tr)

        p_lgb = positive_probability(lgbm.predict_proba(X_va_i), len(X_va_i))
        p_svm = positive_probability(svm.predict_proba(X_va_i), len(X_va_i))

        p_lgb = np.nan_to_num(p_lgb, nan=0.5, posinf=1.0, neginf=0.0)
        p_svm = np.nan_to_num(p_svm, nan=0.5, posinf=1.0, neginf=0.0)

        p_ens = 0.5 * (p_lgb + p_svm)
        pred_va = (p_ens >= 0.5).astype(int)

        fold_scores.append(accuracy_score(y_va, pred_va))

    cv_accuracy = float(np.mean(fold_scores))
    print(f"5-fold CV accuracy: {cv_accuracy:.5f}")

    X_i, X_test_i = impute_fit_transform(X, X_test)

    lgbm_final = make_lgbm()
    lgbm_final.fit(X_i, y_enc)

    svm_final = make_svm_pipeline(X_i.shape[1])
    svm_final.fit(X_i, y_enc)

    p_lgb_test = positive_probability(lgbm_final.predict_proba(X_test_i), len(X_test_i))
    p_svm_test = positive_probability(svm_final.predict_proba(X_test_i), len(X_test_i))

    p_lgb_test = np.nan_to_num(p_lgb_test, nan=0.5, posinf=1.0, neginf=0.0)
    p_svm_test = np.nan_to_num(p_svm_test, nan=0.5, posinf=1.0, neginf=0.0)

    p_test = 0.5 * (p_lgb_test + p_svm_test)
    pred_enc = (p_test >= 0.5).astype(int)
    pred_labels = label_encoder.inverse_transform(pred_enc)

    if np.issubdtype(pred_labels.dtype, np.number):
        if np.all(np.isfinite(pred_labels)) and np.all(
            pred_labels == np.round(pred_labels)
        ):
            pred_labels = pred_labels.astype(int)

    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame(
        {
            "id": np.arange(len(test), dtype=int),
            "Class": pred_labels,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)

    print(f"Saved submission with {len(submission)} rows.")


if __name__ == "__main__":
    main()
