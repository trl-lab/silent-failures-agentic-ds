import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42
NUMERIC_FEATURES = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "TicketLength",
]
CATEGORICAL_FEATURES = ["Sex", "Embarked", "Title", "CabinDeck"]
KNOWN_TITLES = [
    "Mr",
    "Mrs",
    "Miss",
    "Master",
    "Dr",
    "Rev",
    "Col",
    "Major",
    "Mlle",
    "Ms",
    "Lady",
    "Sir",
    "Mme",
    "Don",
    "Jon",
    "Capt",
]


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    if "Name" in d.columns:
        d["Title"] = d["Name"].fillna("Unknown").str.extract(r", ([A-Za-z]+)\.")[0]
    else:
        d["Title"] = "Unknown"

    d["FamilySize"] = d["SibSp"].fillna(0) + d["Parch"].fillna(0) + 1
    d["CabinDeck"] = d["Cabin"].fillna("Unknown").astype(str).str[:1]
    d["TicketLength"] = d["Ticket"].fillna("").astype(str).str.len()

    d["Title"] = d["Title"].where(d["Title"].isin(KNOWN_TITLES), "Other")

    drop_cols = [
        c
        for c in ["PassengerId", "Name", "Ticket", "Cabin", "Survived"]
        if c in d.columns
    ]
    if drop_cols:
        d = d.drop(columns=drop_cols)

    return d


def make_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def get_params() -> dict:
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "min_data_in_leaf": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "seed": RANDOM_STATE,
        "verbose": -1,
    }


train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

y = train["Survived"].astype(int)
X_raw = extract_features(train)
X_test_raw = extract_features(test)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scores = []
best_iterations = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_raw, y), start=1):
    preprocessor = make_preprocessor()

    X_tr = preprocessor.fit_transform(X_raw.iloc[tr_idx], y.iloc[tr_idx])
    X_va = preprocessor.transform(X_raw.iloc[va_idx])
    feature_names = preprocessor.get_feature_names_out().tolist()

    y_tr = y.iloc[tr_idx]
    y_va = y.iloc[va_idx]

    train_set = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names)
    valid_set = lgb.Dataset(
        X_va, label=y_va, reference=train_set, feature_name=feature_names
    )

    model = lgb.train(
        get_params(),
        train_set,
        num_boost_round=1000,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None or best_iteration <= 0:
        best_iteration = 1000

    pred_va = (model.predict(X_va, num_iteration=int(best_iteration)) >= 0.5).astype(
        int
    )
    fold_acc = accuracy_score(y_va, pred_va)

    scores.append(fold_acc)
    best_iterations.append(int(best_iteration))
    print(f"Fold {fold} accuracy: {fold_acc:.5f}")

cv_accuracy = float(np.mean(scores))
print(f"CV accuracy: {cv_accuracy:.5f}")

final_preprocessor = make_preprocessor()
X_final = final_preprocessor.fit_transform(X_raw, y)
final_feature_names = final_preprocessor.get_feature_names_out().tolist()

final_rounds = int(np.mean(best_iterations)) if best_iterations else 300
final_rounds = max(1, final_rounds)

final_model = lgb.train(
    get_params(),
    lgb.Dataset(X_final, label=y, feature_name=final_feature_names),
    num_boost_round=final_rounds,
    callbacks=[lgb.log_evaluation(-1)],
)

X_test = final_preprocessor.transform(X_test_raw)
test_pred = (final_model.predict(X_test, num_iteration=final_rounds) >= 0.5).astype(int)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test), dtype=int), "Survived": test_pred.astype(int)}
)
submission.to_csv("./working/submission.csv", index=False)
