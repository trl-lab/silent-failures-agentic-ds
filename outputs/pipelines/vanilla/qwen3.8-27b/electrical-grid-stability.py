import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/electrical-grid-stability/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/electrical-grid-stability/data/test.parquet"
TARGET = "stabf"
RANDOM_STATE = 42


def load_data():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])

    feature_cols = [c for c in train_df.columns if c != TARGET]

    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = np.nan

    test_df = test_df[feature_cols]
    return train_df, test_df, feature_cols


def encode_target(train_df):
    y_raw = train_df[TARGET].astype(str)
    classes = sorted(y_raw.unique())
    class_to_int = {c: i for i, c in enumerate(classes)}
    y = y_raw.map(class_to_int).astype(int)

    if "unstable" in class_to_int:
        positive_idx = class_to_int["unstable"]
    elif len(classes) > 1:
        positive_idx = 1
    else:
        positive_idx = 0

    return y, classes, positive_idx


def split_feature_types(train_df, feature_cols):
    numeric_features = []
    categorical_features = []

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            numeric_features.append(col)
        else:
            categorical_features.append(col)

    return numeric_features, categorical_features


def fit_preprocessor(X_train, numeric_features, categorical_features):
    numeric_medians = {}
    for col in numeric_features:
        values = pd.to_numeric(X_train[col], errors="coerce").dropna()
        numeric_medians[col] = float(values.median()) if len(values) else 0.0

    categorical_categories = {}
    for col in categorical_features:
        values = X_train[col].astype(object).where(X_train[col].notna(), "missing")
        categorical_categories[col] = sorted(values.unique().tolist())

    return numeric_medians, categorical_categories


def transform_features(X, numeric_features, categorical_features, numeric_medians, categorical_categories):
    X = X.copy()

    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        if X[col].isna().any():
            X[col] = X[col].fillna(numeric_medians[col])

    dummy_frames = []
    for col in categorical_features:
        values = X[col].astype(object).where(X[col].notna(), "missing")
        categories = categorical_categories[col]

        dummy = pd.DataFrame(index=X.index)
        for cat in categories:
            dummy[f"{col}_{cat}"] = (values == cat).astype(int)

        dummy_frames.append(dummy)

    if dummy_frames:
        X = pd.concat([X.drop(columns=categorical_features, errors="ignore")] + dummy_frames, axis=1)
    else:
        X = X.drop(columns=categorical_features, errors="ignore")

    return X


def make_models():
    return {
        "lightgbm": LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=-1,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            min_child_samples=20,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=3,
            gamma=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
        ),
        "catboost": CatBoostClassifier(
            iterations=600,
            learning_rate=0.05,
            depth=4,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            verbose=0,
        ),
    }


def main():
    train_df, test_df, feature_cols = load_data()
    y, classes, positive_idx = encode_target(train_df)
    numeric_features, categorical_features = split_feature_types(train_df, feature_cols)

    X = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    numeric_medians, categorical_categories = fit_preprocessor(
        X,
        numeric_features,
        categorical_features,
    )

    X_processed = transform_features(
        X,
        numeric_features,
        categorical_features,
        numeric_medians,
        categorical_categories,
    )
    X_test_processed = transform_features(
        X_test,
        numeric_features,
        categorical_features,
        numeric_medians,
        categorical_categories,
    )

    use_validation = len(X_processed) >= 50 and len(np.unique(y)) > 1
    if use_validation:
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    else:
        X_train, X_val, y_train, y_val = X_processed, X_processed, y, y

    models = make_models()
    best_name = None
    best_score = -np.inf

    for name, model in models.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, positive_idx]

        if len(np.unique(y_val)) > 1:
            score = roc_auc_score(y_val, proba)
        else:
            score = 0.5

        if score > best_score:
            best_score = score
            best_name = name

    final_model = models[best_name]
    final_model.fit(X_processed, y)
    test_proba = final_model.predict_proba(X_test_processed)[:, positive_idx]

    predictions = pd.DataFrame({
        "id": np.arange(len(test_proba), dtype=int),
        "stabf": test_proba.astype(float),
    })
    predictions.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()