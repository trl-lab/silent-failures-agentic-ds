import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/breast-cancer-wisconsin/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/breast-cancer-wisconsin/data/test.parquet"
TARGET = "Diagnosis"


def is_id_column(col):
    c = str(col).strip().lower()
    if c in {
        "id",
        "index",
        "row_id",
        "rowid",
        "patient_id",
        "patientid",
        "case_id",
        "caseid",
        "sample_id",
        "sampleid",
        "breast_id",
        "breastid",
    }:
        return True
    if c.endswith("_id") or c.startswith("id_"):
        return True
    return False


def get_positive_label(labels):
    labels = [str(x).strip() for x in labels]
    for candidate in ("M", "m", "Malignant", "malignant", "1", "1.0"):
        if candidate in labels:
            return candidate

    negative = {"B", "b", "Benign", "benign", "0", "0.0"}
    for label in labels:
        if label not in negative:
            return label

    return labels[-1] if labels else "1"


def make_preprocessor(numeric_features, categorical_features):
    transformers = []

    if numeric_features:
        numeric_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipe, numeric_features))

    if categorical_features:
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

        categorical_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", encoder),
            ]
        )
        transformers.append(("cat", categorical_pipe, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_model():
    if HAS_LGBM:
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            random_state=42,
            verbose=-1,
        )
    return LogisticRegression(max_iter=1000, random_state=42)


def save_predictions(y_pred, n_rows):
    if n_rows == 0:
        pd.DataFrame({"id": np.arange(0), "Diagnosis": np.array([])}).to_csv(
            "predictions.csv", index=False
        )
        return

    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, 1.0)
    pd.DataFrame({"id": np.arange(n_rows), "Diagnosis": y_pred}).to_csv(
        "predictions.csv", index=False
    )


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET not in train_df.columns:
        raise ValueError("Target column not found in train data")

    train_df = train_df.dropna(subset=[TARGET]).reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    drop_cols = {TARGET}
    for col in train_df.columns:
        if is_id_column(col):
            drop_cols.add(col)

    features = [c for c in train_df.columns if c not in drop_cols]
    X_train = train_df[features].copy()
    X_test = test_df.reindex(columns=features).copy()

    for col in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            X_train[col] = X_train[col].replace([np.inf, -np.inf], np.nan)
            if col in X_test.columns:
                X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan)

    features = [c for c in features if X_train[c].notna().any()]
    X_train = X_train[features].copy()
    X_test = X_test[features].copy()

    constant_cols = [c for c in features if X_train[c].nunique(dropna=True) <= 1]
    if constant_cols:
        X_train = X_train.drop(columns=constant_cols)
        X_test = X_test.drop(columns=constant_cols)
        features = [c for c in features if c not in constant_cols]

    if X_train.shape[1] == 0:
        save_predictions(0.5, len(X_test))
        return

    numeric_features = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    categorical_features = [c for c in X_train.columns if c not in numeric_features]

    high_cardinality = [
        c
        for c in categorical_features
        if X_train[c].nunique(dropna=True) > max(20, 0.5 * len(X_train))
    ]
    if high_cardinality:
        X_train = X_train.drop(columns=high_cardinality)
        X_test = X_test.drop(columns=high_cardinality)
        categorical_features = [c for c in categorical_features if c not in high_cardinality]

    if X_train.shape[1] == 0:
        save_predictions(0.5, len(X_test))
        return

    raw_y = train_df[TARGET].astype(str).str.strip()
    positive_label = get_positive_label(raw_y.unique())
    y_train = (raw_y == positive_label).astype(int)

    if y_train.nunique() < 2:
        save_predictions(0.5, len(X_test))
        return

    if len(X_train) >= 10:
        try:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            validation_pipe = Pipeline(
                [
                    ("preprocessor", make_preprocessor(numeric_features, categorical_features)),
                    ("classifier", make_model()),
                ]
            )
            validation_pipe.fit(X_tr, y_tr)
            val_proba = validation_pipe.predict_proba(X_val)
            val_classes = list(validation_pipe.named_steps["classifier"].classes_)
            val_pos_idx = val_classes.index(1) if 1 in val_classes else len(val_classes) - 1
            roc_auc_score(y_val, val_proba[:, val_pos_idx])
        except Exception:
            pass

    final_pipe = Pipeline(
        [
            ("preprocessor", make_preprocessor(numeric_features, categorical_features)),
            ("classifier", make_model()),
        ]
    )
    final_pipe.fit(X_train, y_train)

    if len(X_test) == 0:
        save_predictions(np.array([]), 0)
        return

    test_proba = final_pipe.predict_proba(X_test)
    classes = list(final_pipe.named_steps["classifier"].classes_)
    pos_idx = classes.index(1) if 1 in classes else len(classes) - 1
    y_pred = test_proba[:, pos_idx]

    save_predictions(y_pred, len(X_test))


if __name__ == "__main__":
    main()