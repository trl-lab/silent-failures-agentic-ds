import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC


def _to_numeric(X):
    X = pd.DataFrame(X)
    return X.apply(pd.to_numeric, errors="coerce")


def _clean_infinite(X):
    X = pd.DataFrame(X)
    return X.replace([np.inf, -np.inf], np.nan)


def _identity(X):
    return X


def build_preprocessor(numeric_cols, categorical_cols):
    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline(
            [
                ("to_numeric", FunctionTransformer(_to_numeric, validate=False)),
                (
                    "clean_infinite",
                    FunctionTransformer(_clean_infinite, validate=False),
                ),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    if not transformers:
        return FunctionTransformer(_identity, validate=False)

    return ColumnTransformer(transformers, remainder="drop")


def main():
    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    target = "Class"
    drop_cols = [target]
    for col in train.columns:
        if col.lower() in {"id", "index", "row_id", "rowid"}:
            drop_cols.append(col)

    y_raw = train[target]
    y_numeric = pd.to_numeric(y_raw, errors="coerce")
    if y_numeric.notna().all():
        y = y_numeric.astype(int)
    else:
        y = y_raw.astype(str)

    X_train = train.drop(columns=drop_cols, errors="ignore")
    X_test = test.drop(columns=drop_cols, errors="ignore")
    X_test = X_test.reindex(columns=X_train.columns, fill_value=np.nan)

    X_train = X_train.loc[:, X_train.notna().any()]
    X_test = X_test.reindex(columns=X_train.columns, fill_value=np.nan)

    numeric_cols = X_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = []

    for col in X_train.select_dtypes(include=["object", "category"]).columns:
        non_missing = X_train[col].notna().sum()
        if non_missing == 0:
            categorical_cols.append(col)
        else:
            converted = pd.to_numeric(X_train[col], errors="coerce")
            if converted.notna().sum() == non_missing:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    classifier = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=42,
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    # Explicitly use Class == 1 as the positive class for ROC AUC.
    positive_label = 1 if 1 in set(y.unique()) else max(y.unique())

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for train_idx, val_idx in cv.split(X_train, y):
        X_tr = X_train.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)
        proba = pipeline.predict_proba(X_val)
        classes = list(pipeline.named_steps["classifier"].classes_)

        if positive_label in classes:
            positive_idx = classes.index(positive_label)
        else:
            positive_idx = len(classes) - 1

        fold_scores.append(
            roc_auc_score(
                y_val,
                proba[:, positive_idx],
                pos_label=positive_label,
            )
        )

    cv_auc = float(np.mean(fold_scores))
    print(f"{cv_auc:.6f}")

    pipeline.fit(X_train, y)
    proba = pipeline.predict_proba(X_test)
    classes = list(pipeline.named_steps["classifier"].classes_)

    if positive_label in classes:
        positive_idx = classes.index(positive_label)
    else:
        positive_idx = len(classes) - 1

    test_predictions = proba[:, positive_idx]

    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_predictions), dtype=int),
            "Class": test_predictions,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
