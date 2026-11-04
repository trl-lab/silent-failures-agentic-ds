import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

TARGET = "DEATH_EVENT"

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

feature_cols = [c for c in train.columns if c != TARGET and c.lower() != "id"]

for c in feature_cols:
    if c not in test.columns:
        test[c] = np.nan


def to_numeric_frame(df):
    out = df[feature_cols].apply(lambda col: pd.to_numeric(col, errors="coerce"))
    return out.replace([np.inf, -np.inf], np.nan)


X = to_numeric_frame(train)
y = train[TARGET].astype(int)
X_test = to_numeric_frame(test)

C_GRID = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
GAMMA_GRID = [0.01, 0.03, 0.08, 0.2, 0.5]


def make_pipeline(C, gamma, probability=False):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=C,
                    gamma=gamma,
                    probability=probability,
                    random_state=42,
                ),
            ),
        ]
    )


def mean_inner_auc(X_train, y_train, C, gamma):
    scores = []

    # Atomic improvement: use 5 repeated inner CV splits instead of 3.
    for repeat in range(5):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7 + repeat)

        for tr_idx, va_idx in cv.split(X_train, y_train):
            model = make_pipeline(C, gamma, probability=False)
            model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            decision = model.decision_function(X_train.iloc[va_idx])
            scores.append(roc_auc_score(y_train.iloc[va_idx], decision))

    return float(np.mean(scores))


def select_best_params(X_data, y_data):
    best_score = -np.inf
    best_params = (1.0, 0.08)

    for C in C_GRID:
        for gamma in GAMMA_GRID:
            score = mean_inner_auc(X_data, y_data, C, gamma)
            if score > best_score:
                best_score = score
                best_params = (C, gamma)

    return best_params, best_score


outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X), dtype=float)

for outer_train_idx, outer_val_idx in outer_cv.split(X, y):
    X_outer_train = X.iloc[outer_train_idx]
    y_outer_train = y.iloc[outer_train_idx]

    best_params, _ = select_best_params(X_outer_train, y_outer_train)

    outer_model = make_pipeline(best_params[0], best_params[1], probability=True)
    outer_model.fit(X_outer_train, y_outer_train)
    oof[outer_val_idx] = outer_model.predict_proba(X.iloc[outer_val_idx])[:, 1]

cv_auc = roc_auc_score(y, oof)
print(f"CV ROC AUC: {cv_auc:.4f}")

final_params, _ = select_best_params(X, y)
final_model = make_pipeline(final_params[0], final_params[1], probability=True)
final_model.fit(X, y)

test_probs = np.clip(final_model.predict_proba(X_test)[:, 1], 0.0, 1.0)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {
        "id": np.arange(len(test_probs), dtype=int),
        "DEATH_EVENT": test_probs,
    }
)
submission.to_csv("./working/submission.csv", index=False)
