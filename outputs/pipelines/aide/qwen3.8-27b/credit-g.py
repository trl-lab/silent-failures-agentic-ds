import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

os.makedirs("./working", exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

target_col = "class"

target_str = train[target_col].astype(str).str.lower().str.strip()
unique_target = set(target_str.unique())

if "bad" in unique_target:
    y = (target_str == "bad").astype(int)
elif "good" in unique_target:
    y = (target_str != "good").astype(int)
else:
    y = train[target_col].astype(int)

drop_cols = [target_col]
drop_cols += [c for c in train.columns if c.lower() in {"id", "index", "row_id"}]

X = train.drop(columns=drop_cols, errors="ignore").copy()
X_test = test.copy()

common_cols = [c for c in X.columns if c in X_test.columns]
X = X[common_cols].copy()
X_test = X_test[common_cols].copy()


def numeric_like(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True

    if series.dtype == object:
        non_null = series.notna().sum()
        if non_null == 0:
            return False

        converted = pd.to_numeric(series, errors="coerce")
        return converted.notna().sum() / non_null >= 0.9

    return False


numeric_cols = [c for c in X.columns if numeric_like(X[c])]
categorical_cols = [c for c in X.columns if c not in numeric_cols]

for c in numeric_cols:
    if not pd.api.types.is_numeric_dtype(X[c]):
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce")

    X[c] = X[c].replace([np.inf, -np.inf], np.nan)
    X_test[c] = X_test[c].replace([np.inf, -np.inf], np.nan)


def build_preprocessor():
    transformers = []

    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_cols,
            )
        )

    if not transformers:
        return FunctionTransformer(lambda x: x, validate=False)

    return ColumnTransformer(transformers, remainder="drop")


def build_model():
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def predict_bad_proba(pipeline, X_data):
    proba = pipeline.predict_proba(X_data)
    classes = list(pipeline.classes_)

    if 1 in classes:
        return proba[:, classes.index(1)]

    if 0 in classes:
        return 1.0 - proba[:, classes.index(0)]

    return proba[:, 0]


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

for train_idx, val_idx in skf.split(X, y):
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]

    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", build_model()),
        ]
    )

    pipeline.fit(X_train, y_train)
    val_proba = predict_bad_proba(pipeline, X_val)

    if len(np.unique(y_val)) > 1:
        auc_scores.append(roc_auc_score(y_val, val_proba))
    else:
        auc_scores.append(0.5)

cv_auc = float(np.mean(auc_scores))
print(f"CV ROC AUC: {cv_auc:.4f}")
print(f"Validation Metric: {cv_auc:.4f}")

final_pipeline = Pipeline(
    [
        ("preprocessor", build_preprocessor()),
        ("classifier", build_model()),
    ]
)

final_pipeline.fit(X, y)
test_proba = predict_bad_proba(final_pipeline, X_test)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        "class": test_proba,
    }
)

submission.to_csv("./working/submission.csv", index=False)
