import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.metrics import root_mean_squared_error

warnings.filterwarnings("ignore")
os.makedirs("./working", exist_ok=True)

TARGET = "NADAC Rate"

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

train = train.copy()
test = test.copy()

if TARGET not in train.columns:
    raise ValueError("Target column 'NADAC Rate' not found in train data.")

train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
train[TARGET] = train[TARGET].replace([np.inf, -np.inf], np.nan)
train = train.dropna(subset=[TARGET])

y = train[TARGET].astype(float)

id_like = {"id", "index", "row_id", "rowid", "identifier", "uid", "uuid"}
feature_cols = [
    col
    for col in train.columns
    if col != TARGET and str(col).strip().lower() not in id_like
]
feature_cols = [col for col in feature_cols if train[col].notna().any()]

X = train[feature_cols].copy()
X_test = test.reindex(columns=feature_cols).copy()


def clean_matrix(arr):
    arr = np.asarray(arr, dtype=float)
    return np.where(np.isfinite(arr), arr, np.nan)


def build_preprocessor(X_fit):
    numeric_cols = []
    cat_cols = []

    for col in X_fit.columns:
        s = X_fit[col]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            numeric_cols.append(col)
        else:
            converted = pd.to_numeric(s, errors="coerce")
            non_null = s.notna().sum()
            if non_null > 0 and converted.notna().sum() >= 0.8 * non_null:
                numeric_cols.append(col)
            else:
                cat_cols.append(col)

    transformers = []

    if numeric_cols:
        num_pipe = FunctionTransformer(
            lambda df: df.apply(pd.to_numeric, errors="coerce"), validate=False
        )
        transformers.append(("num", num_pipe, numeric_cols))

    if cat_cols:
        cat_pipe = Pipeline(
            [
                (
                    "to_str",
                    FunctionTransformer(
                        lambda df: df.astype(str).replace(
                            {"nan": "missing", "None": "missing", "<NA>": "missing"}
                        ),
                        validate=False,
                    ),
                ),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", cat_pipe, cat_cols))

    if not transformers:
        return None

    return ColumnTransformer(transformers, remainder="drop")


def train_and_predict(X_fit, y_fit, X_eval, valid=None, num_boost_round=1000):
    pre = build_preprocessor(X_fit)

    if pre is None:
        return np.full(len(X_eval), float(np.mean(y_fit))), 0

    X_fit_p = clean_matrix(pre.fit_transform(X_fit))
    X_eval_p = clean_matrix(pre.transform(X_eval))

    if X_fit_p.shape[1] == 0 or not np.isfinite(X_fit_p).any():
        return np.full(len(X_eval), float(np.mean(y_fit))), 0

    params = {
        "objective": "regression_l2",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 5,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }

    dtrain = lgb.Dataset(X_fit_p, label=np.asarray(y_fit, dtype=float))

    if valid is None:
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            callbacks=[lgb.log_evaluation(-1)],
        )
        pred = booster.predict(X_eval_p)
        return pred, num_boost_round

    dvalid = lgb.Dataset(
        X_eval_p, label=np.asarray(valid, dtype=float), reference=dtrain
    )

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    best_iter = getattr(booster, "best_iteration", 0)
    if best_iter is None or best_iter <= 0:
        best_iter = num_boost_round

    pred = booster.predict(X_eval_p, num_iteration=best_iter)
    return pred, best_iter


kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []
best_iters = []

for tr_idx, va_idx in kf.split(X):
    X_tr = X.iloc[tr_idx]
    y_tr = y.iloc[tr_idx]
    X_va = X.iloc[va_idx]
    y_va = y.iloc[va_idx]

    pred_va, best_iter = train_and_predict(X_tr, y_tr, X_va, valid=y_va)

    if y.min() >= 0:
        pred_va = np.clip(pred_va, 0.0, None)

    fold_scores.append(root_mean_squared_error(y_va, pred_va))

    if best_iter > 0:
        best_iters.append(best_iter)

val_rmse = float(np.mean(fold_scores))
print(f"Validation RMSE (5-fold CV): {val_rmse:.6f}")

final_rounds = int(np.clip(np.mean(best_iters), 50, 1000)) if best_iters else 500
test_pred, _ = train_and_predict(X, y, X_test, num_boost_round=final_rounds)

if y.min() >= 0:
    test_pred = np.clip(test_pred, 0.0, None)

submission = pd.DataFrame(
    {"id": np.arange(len(test), dtype=int), "NADAC Rate": test_pred}
)

submission.to_csv("./working/submission.csv", index=False)
print(f"Saved ./working/submission.csv with {len(submission)} rows.")
