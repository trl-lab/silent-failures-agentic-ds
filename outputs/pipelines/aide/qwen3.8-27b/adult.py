import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from pandas.api.types import is_string_dtype, is_categorical_dtype

os.makedirs("./working", exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

TARGET = "income"

if "id" in train.columns:
    train = train.drop(columns=["id"])

train = train.dropna(subset=[TARGET]).reset_index(drop=True)

if (
    train[TARGET].dtype == object
    or is_string_dtype(train[TARGET])
    or is_categorical_dtype(train[TARGET])
):
    y = train[TARGET].astype(str).str.startswith(">50K").astype(int)
else:
    y = train[TARGET].astype(int)

X = train.drop(columns=[TARGET])
test_X = test.reindex(columns=X.columns)

known_numeric = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

for col in known_numeric:
    if col in X.columns:
        X[col] = pd.to_numeric(X[col].astype(object), errors="coerce")
        test_X[col] = pd.to_numeric(test_X[col].astype(object), errors="coerce")

missing_tokens = {
    "nan": np.nan,
    "NaN": np.nan,
    "NAN": np.nan,
    "null": np.nan,
    "NULL": np.nan,
    "None": np.nan,
    "none": np.nan,
    "": np.nan,
    "<NA>": np.nan,
    "NA": np.nan,
    "N/A": np.nan,
}

for col in X.columns:
    if (
        X[col].dtype == object
        or is_string_dtype(X[col])
        or is_categorical_dtype(X[col])
    ):
        X[col] = X[col].astype(object)
        test_X[col] = test_X[col].astype(object)

        X[col] = X[col].replace("?", np.nan)
        test_X[col] = test_X[col].replace("?", np.nan)

        X[col] = X[col].astype(str).str.strip()
        test_X[col] = test_X[col].astype(str).str.strip()

        X[col] = X[col].replace(missing_tokens)
        test_X[col] = test_X[col].replace(missing_tokens)
    else:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        test_X[col] = pd.to_numeric(test_X[col], errors="coerce")

categorical_cols = []
for col in X.columns:
    if (
        X[col].dtype == object
        or is_string_dtype(X[col])
        or is_categorical_dtype(X[col])
    ):
        cats = list(pd.unique(X[col].dropna()))
        if len(cats) == 0:
            cats = ["unknown"]

        X[col] = pd.Categorical(X[col], categories=cats)
        test_X[col] = pd.Categorical(test_X[col], categories=cats)
        categorical_cols.append(col)

all_nan_cols = X.columns[X.isna().all()]
if len(all_nan_cols) > 0:
    X = X.drop(columns=all_nan_cols)
    test_X = test_X.drop(columns=all_nan_cols, errors="ignore")
    categorical_cols = [c for c in categorical_cols if c not in all_nan_cols]

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "feature_fraction": 0.9,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "seed": 42,
    "n_jobs": -1,
    "verbosity": -1,
}

num_boost_round = 3000
early_stopping_rounds = 50

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_proba = np.zeros(len(y), dtype=float)
best_iters = []

for tr_idx, val_idx in skf.split(X, y):
    dtrain = lgb.Dataset(
        X.iloc[tr_idx],
        label=y.iloc[tr_idx],
        categorical_feature=categorical_cols,
        free_raw_data=True,
    )
    dvalid = lgb.Dataset(
        X.iloc[val_idx],
        label=y.iloc[val_idx],
        categorical_feature=categorical_cols,
        reference=dtrain,
        free_raw_data=True,
    )

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dvalid],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    best_iter = getattr(booster, "best_iteration", 0) or 0
    if best_iter <= 0:
        best_iter = num_boost_round

    best_iters.append(best_iter)
    oof_proba[val_idx] = booster.predict(X.iloc[val_idx], num_iteration=best_iter)

oof_pred = (oof_proba >= 0.5).astype(int)
cv_accuracy = accuracy_score(y, oof_pred)
print(f"5-fold CV accuracy: {cv_accuracy:.6f}")

final_rounds = max(1, int(round(np.mean(best_iters))))

dtrain_full = lgb.Dataset(
    X,
    label=y,
    categorical_feature=categorical_cols,
    free_raw_data=True,
)

final_booster = lgb.train(
    params,
    dtrain_full,
    num_boost_round=final_rounds,
    callbacks=[lgb.log_evaluation(-1)],
)

test_proba = final_booster.predict(test_X, num_iteration=final_rounds)
test_pred = np.where(test_proba >= 0.5, ">50K", "<=50K")

submission = pd.DataFrame(
    {
        "id": np.arange(len(test_pred), dtype=int),
        "income": test_pred,
    }
)
submission.to_csv("./working/submission.csv", index=False)
