import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/titanic/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/titanic/data/test.parquet"


class DenseOneHot:
    def __init__(self):
        try:
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    def fit(self, X, y=None):
        self.encoder.fit(X)
        return self

    def transform(self, X):
        out = self.encoder.transform(X)
        if hasattr(out, "toarray"):
            out = out.toarray()
        return out

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


def get_title(name):
    if pd.isna(name):
        return "Unknown"
    try:
        title = str(name).split(",")[1].split(".")[0].strip()
    except Exception:
        return "Unknown"
    if title in {"Mr", "Mrs", "Miss", "Master"}:
        return title
    return "Rare"


def get_cabin_deck(cabin):
    if pd.isna(cabin):
        return "Unknown"
    cabin = str(cabin).strip()
    if not cabin:
        return "Unknown"
    return cabin[0]


def add_features(df):
    df = df.copy()

    if "Name" in df.columns:
        df["Title"] = df["Name"].apply(get_title)
    else:
        df["Title"] = "Unknown"

    if "Cabin" in df.columns:
        df["CabinDeck"] = df["Cabin"].apply(get_cabin_deck)
    else:
        df["CabinDeck"] = "Unknown"

    sib = df["SibSp"].fillna(0) if "SibSp" in df.columns else 0
    parch = df["Parch"].fillna(0) if "Parch" in df.columns else 0
    df["FamilySize"] = sib + parch
    df["IsAlone"] = (df["FamilySize"] == 0).astype(int)

    if "Pclass" in df.columns and "Sex" in df.columns:
        df["Pclass_Sex"] = (
            df["Pclass"].astype(str) + "_" + df["Sex"].fillna("Unknown").astype(str)
        )
    else:
        df["Pclass_Sex"] = "Unknown"

    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    return df


def make_preprocessor(numeric_cols, categorical_cols):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", DenseOneHot()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def make_models():
    lgbm = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        num_leaves=15,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )

    cat = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=4,
        l2_leaf_reg=3.0,
        random_state=42,
        verbose=False,
    )

    return [lgbm, xgb, cat]


def ensemble_predict(X_fit, y_fit, X_pred, numeric_cols, categorical_cols):
    pipelines = []
    for model in make_models():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(numeric_cols, categorical_cols)),
                ("model", model),
            ]
        )
        pipeline.fit(X_fit, y_fit)
        pipelines.append(pipeline)

    proba = np.mean(
        [p.predict_proba(X_pred)[:, 1] for p in pipelines],
        axis=0,
    )
    return (proba >= 0.5).astype(int)


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if "Survived" in test_df.columns:
        test_df = test_df.drop(columns=["Survived"])

    y_train = train_df["Survived"].astype(int)
    X_train = train_df.drop(columns=["Survived"])

    X_train = add_features(X_train)
    X_test = add_features(test_df)

    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    categorical_cols = [
        c
        for c in ["Sex", "Embarked", "Title", "CabinDeck", "Pclass_Sex"]
        if c in common_cols
    ]
    numeric_cols = [c for c in common_cols if c not in categorical_cols]

    for col in categorical_cols:
        X_train[col] = X_train[col].astype("object").fillna("Unknown")
        X_test[col] = X_test[col].astype("object").fillna("Unknown")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    val_pred = ensemble_predict(X_tr, y_tr, X_val, numeric_cols, categorical_cols)
    _ = accuracy_score(y_val, val_pred)

    final_pred = ensemble_predict(X_train, y_train, X_test, numeric_cols, categorical_cols)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(final_pred), dtype=int),
            "Survived": final_pred.astype(int),
        }
    )
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()