import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/petfinder-adoption-prediction/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/petfinder-adoption-prediction/data/test.parquet"


def quadratic_weighted_kappa(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    min_rating = min(int(y_true.min()), int(y_pred.min()))
    max_rating = max(int(y_true.max()), int(y_pred.max()))
    num_ratings = max_rating - min_rating + 1

    observed = np.zeros((num_ratings, num_ratings), dtype=np.float64)
    for i in range(num_ratings):
        for j in range(num_ratings):
            observed[i, j] = np.sum((y_true == i + min_rating) & (y_pred == j + min_rating))

    if observed.sum() == 0:
        return 0.0

    r = np.arange(num_ratings)
    r1 = r[:, None]
    r2 = r[None, :]
    weights = (r1 - r2) ** 2 / ((num_ratings - 1) ** 2)

    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    return 1.0 - np.sum(weights * observed) / np.sum(weights * expected)


class SimpleOrdinalEncoder:
    def __init__(self):
        self.categories_ = {}

    def fit(self, df):
        self.categories_ = {}
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            values = df[col].dropna().astype(str).unique()
            self.categories_[col] = {str(v): i for i, v in enumerate(sorted(values))}
        return self

    def transform(self, df):
        df = df.copy()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            if col in self.categories_:
                df[col] = (
                    df[col]
                    .astype(str)
                    .map(self.categories_[col])
                    .fillna(-1)
                    .astype(int)
                )
            else:
                df[col] = -1
        return df


def add_text_features(df):
    df = df.copy()

    for col in ["Name", "Description"]:
        if col not in df.columns:
            continue

        text = df[col].fillna("").astype(str)
        base = col.lower()

        df[f"{base}_len"] = text.str.len()
        df[f"{base}_word_count"] = text.str.count(r"\S+")
        df[f"{base}_digit_count"] = text.str.count(r"\d")
        df[f"{base}_exclamation_count"] = text.str.count("!")
        df[f"{base}_question_count"] = text.str.count(r"\?")
        df[f"{base}_uppercase_ratio"] = text.str.count(r"[A-Z]") / (text.str.len() + 1)

    return df


def prepare_dataframe(df, encoder=None, fit_encoder=False):
    df = df.copy()

    if "AdoptionSpeed" in df.columns:
        df = df.drop(columns=["AdoptionSpeed"])

    for col in ["PetID", "RescuerID"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = add_text_features(df)

    for col in ["Name", "Description"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    if "Breed1" in df.columns and "Breed2" in df.columns:
        df["Breed1_Breed2_same"] = (df["Breed1"] == df["Breed2"]).astype(int)
        df["Breed1_Breed2_sum"] = df["Breed1"] + df["Breed2"]

    if "Color1" in df.columns and "Color2" in df.columns and "Color3" in df.columns:
        df["Color_sum"] = df["Color1"] + df["Color2"] + df["Color3"]

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna("missing").astype(str)

    if fit_encoder:
        encoder = SimpleOrdinalEncoder()
        encoder.fit(df)

    if encoder is not None:
        df = encoder.transform(df)

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(-1)

    return df, encoder


def make_models():
    return [
        LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="mlogloss",
        ),
        CatBoostClassifier(
            iterations=600,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=0,
            thread_count=-1,
        ),
    ]


def average_probabilities(models, X, all_classes):
    class_to_idx = {int(c): i for i, c in enumerate(all_classes)}
    avg_proba = np.zeros((X.shape[0], len(all_classes)), dtype=np.float64)

    for model in models:
        proba = model.predict_proba(X)
        classes = model.classes_
        for i, cls in enumerate(classes):
            idx = class_to_idx.get(int(cls))
            if idx is not None:
                avg_proba[:, idx] += proba[:, i]

    avg_proba /= len(models)
    return avg_proba


def probabilities_to_labels(avg_proba, all_classes):
    expected = avg_proba @ np.asarray(all_classes, dtype=np.float64)
    labels = np.round(expected).astype(int)
    return np.clip(labels, int(all_classes.min()), int(all_classes.max()))


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    y = train_df["AdoptionSpeed"].astype(int)
    all_classes = np.unique(y.values)

    X_raw_train = train_df.drop(columns=["AdoptionSpeed"])
    X_raw_test = test_df

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_raw_train,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train, encoder_val = prepare_dataframe(X_train_raw, fit_encoder=True)
    X_val, _ = prepare_dataframe(X_val_raw, encoder=encoder_val)
    X_test_val, _ = prepare_dataframe(X_raw_test, encoder=encoder_val)

    X_val = X_val.reindex(columns=X_train.columns, fill_value=-1)
    X_test_val = X_test_val.reindex(columns=X_train.columns, fill_value=-1)

    val_models = make_models()
    for model in val_models:
        model.fit(X_train, y_train)

    val_proba = average_probabilities(val_models, X_val, all_classes)
    val_pred = probabilities_to_labels(val_proba, all_classes)
    val_qwk = quadratic_weighted_kappa(y_val.values, val_pred)
    print(f"Validation QWK: {val_qwk:.6f}")

    X_full, encoder_full = prepare_dataframe(X_raw_train, fit_encoder=True)
    X_test, _ = prepare_dataframe(X_raw_test, encoder=encoder_full)
    X_test = X_test.reindex(columns=X_full.columns, fill_value=-1)

    final_models = make_models()
    for model in final_models:
        model.fit(X_full, y)

    test_proba = average_probabilities(final_models, X_test, all_classes)
    test_pred = probabilities_to_labels(test_proba, all_classes)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_pred), dtype=int),
            "AdoptionSpeed": test_pred.astype(int),
        }
    )
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()