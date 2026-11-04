import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/cmc/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/cmc/data/test.parquet"
TARGET = "Contraceptive_method_used"


def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    transformers = []

    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
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
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            )
        )

    if not transformers:
        transformers = [("passthrough", "passthrough", X.columns.tolist())]

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_models():
    return {
        "lgbm": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=1,
        ),
        "catboost": CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=0,
            thread_count=1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            n_jobs=1,
        ),
    }


def to_1d_array(x):
    arr = np.asarray(x)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.reshape(-1)


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    train_df = train_df.dropna(subset=[TARGET]).reset_index(drop=True)

    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])

    X = train_df.drop(columns=[TARGET]).copy()
    y_raw = train_df[TARGET].astype(object).copy()

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y_raw)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_val_processed = preprocessor.transform(X_val)

    best_name = None
    best_score = -np.inf

    for name, model in build_models().items():
        model.fit(X_train_processed, y_train)
        val_pred = to_1d_array(model.predict(X_val_processed)).astype(int)
        score = accuracy_score(y_val, val_pred)

        if score > best_score:
            best_score = score
            best_name = name

    final_preprocessor = build_preprocessor(X)
    X_full_processed = final_preprocessor.fit_transform(X, y)
    X_test_processed = final_preprocessor.transform(test_df)

    final_model = build_models()[best_name]
    final_model.fit(X_full_processed, y)

    test_pred_encoded = to_1d_array(final_model.predict(X_test_processed)).astype(int)

    most_frequent_encoded = int(np.argmax(np.bincount(y)))
    valid = (test_pred_encoded >= 0) & (test_pred_encoded < len(target_encoder.classes_))
    test_pred_encoded = np.where(valid, test_pred_encoded, most_frequent_encoded)

    test_pred = to_1d_array(target_encoder.inverse_transform(test_pred_encoded)).astype(object)

    if len(test_pred) != len(test_df):
        n = len(test_df)
        if len(test_pred) < n:
            fill_value = target_encoder.classes_[most_frequent_encoded]
            test_pred = np.concatenate(
                [test_pred, np.full(n - len(test_pred), fill_value, dtype=object)]
            )
        else:
            test_pred = test_pred[:n]

    output = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            TARGET: test_pred,
        }
    )

    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()