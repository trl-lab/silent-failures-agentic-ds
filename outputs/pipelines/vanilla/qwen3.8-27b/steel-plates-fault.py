import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/steel-plates-fault/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/steel-plates-fault/data/test.parquet"
TARGET = "target"
RANDOM_STATE = 42


def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def prepare_data(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    if TARGET not in train_df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in train data.")

    train_df = train_df.dropna(subset=[TARGET])
    y_raw = train_df[TARGET].astype(str)

    X_train = train_df.drop(columns=[TARGET])
    X_test = test_df.reindex(columns=X_train.columns)

    categorical_cols = []
    for col in X_train.columns:
        dtype = X_train[col].dtype
        if dtype == object or dtype.name == "category":
            categorical_cols.append(col)

    numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

    for col in categorical_cols:
        X_train[col] = X_train[col].astype(object)
        X_test[col] = X_test[col].astype(object)

    for col in numeric_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    return X_train, X_test, y_raw, numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols):
    transformers = []

    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", make_onehot()),
                    ]
                ),
                categorical_cols,
            )
        )

    if not transformers:
        transformers.append(("passthrough", "passthrough", numeric_cols + categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_model(name):
    if name == "lgbm":
        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    if name == "xgb":
        return XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="mlogloss",
        )
    if name == "catboost":
        return CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_seed=RANDOM_STATE,
            verbose=0,
        )
    raise ValueError(f"Unknown model: {name}")


def balanced_sample_weight(y):
    y = np.asarray(y)
    n = len(y)
    classes = np.unique(y)
    counts = np.bincount(y, minlength=len(classes))
    counts = np.maximum(counts, 1)
    class_weights = n / (len(classes) * counts)
    return class_weights[y]


def fit_pipeline(pipeline, X, y, sample_weight=None):
    if sample_weight is None:
        pipeline.fit(X, y)
    else:
        pipeline.fit(X, y, model__sample_weight=sample_weight)


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    X_train, X_test, y_raw, numeric_cols, categorical_cols = prepare_data(train_df, test_df)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_raw)

    try:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y_train,
        )
    except ValueError:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )

    candidates = []
    for model_name in ["lgbm", "xgb", "catboost"]:
        candidates.append((f"{model_name}_unweighted", model_name, False))
        candidates.append((f"{model_name}_balanced", model_name, True))

    best_score = -1.0
    best_candidate = None

    for candidate_name, model_name, use_balanced in candidates:
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        model = make_model(model_name)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        sample_weight = balanced_sample_weight(y_fit) if use_balanced else None
        fit_pipeline(pipeline, X_fit, y_fit, sample_weight)

        val_pred = pipeline.predict(X_val)
        score = accuracy_score(y_val, val_pred)

        if score > best_score:
            best_score = score
            best_candidate = (candidate_name, model_name, use_balanced)

    if best_candidate is None:
        raise RuntimeError("No model candidate was trained successfully.")

    _, best_model_name, best_use_balanced = best_candidate

    final_preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    final_model = make_model(best_model_name)
    final_pipeline = Pipeline([("preprocessor", final_preprocessor), ("model", final_model)])

    final_sample_weight = balanced_sample_weight(y_train) if best_use_balanced else None
    fit_pipeline(final_pipeline, X_train, y_train, final_sample_weight)

    test_pred_idx = final_pipeline.predict(X_test)
    test_pred_labels = label_encoder.inverse_transform(test_pred_idx)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            "target": test_pred_labels,
        }
    )
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()