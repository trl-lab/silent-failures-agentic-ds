import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import (
    StratifiedGroupKFold,
    GroupKFold,
    StratifiedKFold,
    GridSearchCV,
)

SEED = 42
np.random.seed(SEED)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Identify target column.
target = None
for col in train.columns:
    if isinstance(col, str) and col.strip().lower() == "class":
        target = col
        break
if target is None:
    target = train.columns[-1]

# Identify grouped sample identifier, if present.
group_col = None
for col in train.columns:
    if isinstance(col, str):
        normalized = col.strip().lower().replace(" ", "_")
        if "mouse" in normalized:
            group_col = col
            break


def is_excluded(col):
    if not isinstance(col, str):
        return False
    normalized = col.strip().lower().replace(" ", "_")
    if normalized == str(target).strip().lower():
        return True
    if "mouse" in normalized:
        return True
    if normalized in {"genotype", "treatment", "behavior", "class"}:
        return True
    return False


# Prefer the 77 protein expression features, which end with "_n".
protein_features = [
    col
    for col in train.columns
    if isinstance(col, str)
    and col.strip().endswith("_n")
    and col in test.columns
    and not is_excluded(col)
]

# Fallback if the naming convention is not available.
if len(protein_features) == 0:
    fallback = [
        col
        for col in train.columns
        if col in test.columns and col != target and not is_excluded(col)
    ]
    numeric_fallback = [
        col for col in fallback if pd.api.types.is_numeric_dtype(train[col])
    ]
    protein_features = numeric_fallback if numeric_fallback else fallback

features = protein_features

X_train = train[features].copy()
X_test = test[features].copy()

for col in features:
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# Remove rows with missing target labels.
y_raw = train[target]
valid = y_raw.notna()
X_train = X_train[valid].copy()
y = y_raw[valid].astype(str)

groups = None
if group_col is not None and group_col in train.columns:
    g = train[group_col][valid]
    if g.notna().all():
        groups = g.astype(str)
        if len(groups.unique()) < 5:
            groups = None

# Remove feature columns that are entirely missing.
X_train = X_train.dropna(axis=1, how="all")
X_test = X_test[X_train.columns]
features = X_train.columns.tolist()

# Last-resort fallback if no usable features were found.
if len(features) == 0:
    fallback = [
        col
        for col in train.columns
        if col in test.columns and col != target and not is_excluded(col)
    ]
    features = [col for col in fallback if pd.api.types.is_numeric_dtype(train[col])]
    if features:
        X_train = train[features][valid].copy()
        X_test = test[features].copy()

        for col in features:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        X_train = X_train.dropna(axis=1, how="all")
        X_test = X_test[X_train.columns]
        features = X_train.columns.tolist()

os.makedirs("./working", exist_ok=True)

if len(features) == 0 or len(X_train) == 0:
    pred_class = y.mode().iloc[0] if len(y) > 0 else "unknown"
    submission = pd.DataFrame(
        {
            "id": np.arange(len(test)),
            "class": pred_class,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)
    print("Validation accuracy: 0.0000")
else:

    def select_cv():
        if groups is not None and len(groups.unique()) >= 5:
            candidates = [
                (
                    StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED),
                    True,
                ),
                (
                    GroupKFold(n_splits=5, shuffle=True, random_state=SEED),
                    True,
                ),
            ]
            for cv_candidate, use_groups in candidates:
                try:
                    list(cv_candidate.split(X_train, y, groups=groups))
                    return cv_candidate, use_groups
                except Exception:
                    pass
        return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED), False

    cv, use_groups = select_cv()

    base_model = Pipeline(
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
                    random_state=SEED,
                    cache_size=500,
                ),
            ),
        ]
    )

    param_grid = {
        "clf__C": [1.0, 5.0, 10.0, 50.0, 100.0],
        "clf__gamma": ["scale", 0.005, 0.01, 0.02, 0.05, 0.1],
    }

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
        refit=True,
    )

    if use_groups:
        grid.fit(X_train, y, groups=groups)
    else:
        grid.fit(X_train, y)

    print(f"Validation accuracy: {grid.best_score_:.4f}")
    print(f"Best parameters: {grid.best_params_}")

    predictions = grid.predict(X_test)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test)),
            "class": predictions,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)
