import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/ilpd/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/ilpd/data/test.parquet"
TARGET = "Class"
RANDOM_STATE = 42


def load_data():
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)
    return train, test


def prepare_target(train):
    y_raw = train[TARGET].copy()
    y_num = pd.to_numeric(y_raw, errors="coerce")

    if y_num.isna().any():
        y_num = y_raw.astype(str).str.strip().str.lower()
        mapping = {
            "1": 1,
            "2": 0,
            "liver": 1,
            "non liver": 0,
            "non-liver": 0,
            "yes": 1,
            "no": 0,
            "true": 1,
            "false": 0,
        }
        y_num = y_num.map(mapping)

    y_num = y_num.dropna()
    classes = np.sort(y_num.unique())

    if len(classes) != 2:
        raise ValueError(f"Expected a binary target, found classes: {classes}")

    positive_class = 1 if 1 in classes else classes[0]
    y = (y_num == positive_class).astype(int)

    return y, positive_class


def prepare_features(train, test):
    feature_cols = [c for c in train.columns if c != TARGET]

    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    numeric_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(X_train[c])
    ]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    for col in categorical_cols:
        X_train[col] = X_train[col].astype("object")
        X_test[col] = X_test[col].astype("object")

        mode_value = X_train[col].mode(dropna=True)
        fill_value = mode_value[0] if len(mode_value) > 0 else "missing"

        X_train[col] = X_train[col].fillna(fill_value)
        X_test[col] = X_test[col].fillna(fill_value)

        train_dummies = pd.get_dummies(X_train[col], prefix=col, dtype=float)
        X_train = pd.concat([X_train.drop(columns=[col]), train_dummies], axis=1)

        test_dummies = pd.get_dummies(X_test[col], prefix=col, dtype=float)
        test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0.0)
        X_test = pd.concat([X_test.drop(columns=[col]), test_dummies], axis=1)

    if numeric_cols:
        numeric_imputer = SimpleImputer(strategy="median")
        X_train[numeric_cols] = numeric_imputer.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = numeric_imputer.transform(X_test[numeric_cols])

    X_train = X_train.fillna(0.0)
    X_test = X_test.fillna(0.0)

    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    return X_train, X_test


def make_models():
    return {
        "lightgbm": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
        ),
        "catboost": CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            verbose=0,
        ),
    }


def main():
    train_df, test_df = load_data()

    y_train, _ = prepare_target(train_df)
    X_train, X_test = prepare_features(train_df, test_df)

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    models = make_models()
    val_scores = {}

    for name, model in models.items():
        model.fit(X_fit, y_fit)
        val_proba = model.predict_proba(X_val)[:, 1]
        val_scores[name] = roc_auc_score(y_val, val_proba)

    scores = np.array(list(val_scores.values()), dtype=float)
    weights = scores - scores.min() + 1e-6
    weights /= weights.sum()

    test_proba = np.zeros(len(X_test), dtype=float)
    for (name, model), weight in zip(models.items(), weights):
        model.fit(X_train, y_train)
        test_proba += weight * model.predict_proba(X_test)[:, 1]

    predictions = pd.DataFrame({
        "id": np.arange(len(test_df), dtype=int),
        "Class": test_proba,
    })

    predictions.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()