import inspect
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from lightgbm import LGBMClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/dry-bean-dataset/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/dry-bean-dataset/data/test.parquet"
TARGET = "Class"
RANDOM_STATE = 42


def make_one_hot_encoder():
    params = inspect.signature(OneHotEncoder.__init__).parameters
    if "sparse_output" in params:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(X_train):
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in X_train.columns if col not in numeric_cols]

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
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical_cols,
            )
        )

    if not transformers:
        return "passthrough"

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_model():
    return LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET not in train_df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in train data.")

    train_df = train_df.dropna(subset=[TARGET])

    X_train = train_df.drop(columns=[TARGET], errors="ignore")
    X_test = test_df.drop(columns=[TARGET], errors="ignore")

    feature_cols = X_train.columns.tolist()
    X_train = X_train.reindex(columns=feature_cols)
    X_test = X_test.reindex(columns=feature_cols)

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    feature_cols = [col for col in feature_cols if X_train[col].notna().any()]
    X_train = X_train[feature_cols]
    X_test = X_test.reindex(columns=feature_cols)

    y_raw = train_df[TARGET]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    if len(np.unique(y)) == 1:
        pred_enc = np.full(len(X_test), int(y.iloc[0]), dtype=int)
    elif X_train.shape[1] == 0:
        most_frequent_class = int(np.bincount(y).argmax())
        pred_enc = np.full(len(X_test), most_frequent_class, dtype=int)
    else:
        preprocessor = make_preprocessor(X_train)

        try:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train,
                y,
                test_size=0.2,
                random_state=RANDOM_STATE,
                stratify=y,
            )
        except ValueError:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train,
                y,
                test_size=0.2,
                random_state=RANDOM_STATE,
            )

        validation_pipeline = Pipeline(
            [("preprocessor", preprocessor), ("classifier", make_model())]
        )
        validation_pipeline.fit(X_tr, y_tr)
        val_pred = validation_pipeline.predict(X_val)
        _ = accuracy_score(y_val, val_pred)

        final_pipeline = Pipeline(
            [("preprocessor", preprocessor), ("classifier", make_model())]
        )
        final_pipeline.fit(X_train, y)
        pred_enc = final_pipeline.predict(X_test)

    pred_labels = label_encoder.inverse_transform(pred_enc)

    output = pd.DataFrame(
        {
            "id": np.arange(len(pred_labels), dtype=int),
            "Class": pred_labels,
        }
    )
    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()