import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/bank-marketing/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/bank-marketing/data/test.parquet"
TARGET = "Class"


def make_binary_target(train_df: pd.DataFrame) -> pd.Series:
    target = train_df[TARGET]
    unique_labels = list(target.dropna().unique())

    def label_sort_key(label):
        try:
            return (0, float(label), str(label))
        except (TypeError, ValueError):
            return (1, 0.0, str(label))

    positive_label = max(unique_labels, key=label_sort_key)
    return (target == positive_label).astype(int)


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in X_train.columns if col not in numeric_cols]

    transformers = []

    if numeric_cols:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        transformers.append(("num", numeric_transformer, numeric_cols))

    if categorical_cols:
        categorical_transformer = Pipeline(
            steps=[
                (
                    "clean",
                    FunctionTransformer(
                        lambda x: x.astype(object).fillna("missing").astype(str),
                        validate=False,
                    ),
                ),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    y_train_full = make_binary_target(train_df)

    X_train = train_df.drop(columns=[TARGET])
    X_test = test_df.drop(columns=[TARGET], errors="ignore")

    for col in X_train.columns:
        if col not in X_test.columns:
            X_test[col] = np.nan
    X_test = X_test[X_train.columns].copy()

    preprocessor = build_preprocessor(X_train)

    model = LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        is_unbalance=True,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    class_counts = y_train_full.value_counts()
    if len(class_counts) > 1 and class_counts.min() >= 2 and len(X_train) > 1:
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train,
            y_train_full,
            test_size=0.2,
            random_state=42,
            stratify=y_train_full,
        )

        pipeline.fit(X_train_split, y_train_split)
        val_pred = pipeline.predict_proba(X_val_split)[:, 1]
        try:
            roc_auc_score(y_val_split, val_pred)
        except Exception:
            pass

    pipeline.fit(X_train, y_train_full)

    if len(X_test) == 0:
        test_pred = np.array([], dtype=float)
    else:
        test_pred = pipeline.predict_proba(X_test)[:, 1]

    test_pred = np.nan_to_num(test_pred, nan=0.5, posinf=1.0, neginf=0.0)
    test_pred = np.clip(test_pred, 0.0, 1.0)

    predictions = pd.DataFrame(
        {
            "id": np.arange(len(X_test), dtype=int),
            "Class": test_pred,
        }
    )
    predictions.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()