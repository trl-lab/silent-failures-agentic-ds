import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/churn/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/churn/data/test.parquet"
TARGET = "class"
DROP_COLS = ["phone_number"]


def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def encode_target(y):
    y = y.astype(str).str.strip()
    unique = sorted(y.unique())
    if len(unique) == 2:
        positive = "1" if "1" in unique else unique[-1]
        return (y == positive).astype(int)
    mapping = {v: i for i, v in enumerate(unique)}
    return y.map(mapping).astype(int)


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    train_df = train_df.dropna(subset=[TARGET])
    y = encode_target(train_df[TARGET])

    X = train_df.drop(columns=[TARGET])
    X_test = test_df.drop(columns=[TARGET], errors="ignore")

    X = X.drop(columns=[c for c in DROP_COLS if c in X.columns], errors="ignore")
    X_test = X_test.drop(columns=[c for c in DROP_COLS if c in X_test.columns], errors="ignore")

    X_test = X_test.reindex(columns=X.columns)

    X = X.loc[:, X.notna().any()]
    X = X.loc[:, X.nunique(dropna=True) > 1]
    X_test = X_test.reindex(columns=X.columns)

    categorical_cols = [c for c in X.columns if X[c].dtype.name in ("category", "object")]
    if "state" in X.columns and "state" not in categorical_cols:
        categorical_cols.append("state")
    categorical_cols = [c for c in categorical_cols if c in X.columns]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    for col in categorical_cols:
        X[col] = X[col].astype(object)
        X_test[col] = X_test[col].astype(object)

    transformers = []
    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_cols
        ))
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", make_onehot())
            ]),
            categorical_cols
        ))

    if not transformers:
        transformers = [(
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            list(X.columns)
        )]

    preprocessor = ColumnTransformer(transformers, remainder="drop")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "lgbm": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        "xgb": XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist"
        ),
        "cat": CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=0
        )
    }

    val_scores = {}
    val_preds = {}

    for name, model in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_val)[:, 1]
        if len(np.unique(y_val)) < 2:
            score = 0.5
        else:
            score = roc_auc_score(y_val, proba)
        val_scores[name] = score
        val_preds[name] = proba

    weights = np.array([max(score - 0.5, 0.0) for score in val_scores.values()])
    if weights.sum() <= 0:
        weights = np.ones(len(models)) / len(models)
    else:
        weights = weights / weights.sum()

    final_preds = []
    for name, model in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])
        pipe.fit(X, y)
        proba = pipe.predict_proba(X_test)[:, 1]
        final_preds.append(proba)

    final_proba = np.average(np.vstack(final_preds), axis=0, weights=weights)
    final_proba = np.nan_to_num(final_proba, nan=0.5, posinf=1.0, neginf=0.0)
    final_proba = np.clip(final_proba, 0.0, 1.0)

    submission = pd.DataFrame({
        "id": np.arange(len(test_df), dtype=int),
        "class": final_proba
    })
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()