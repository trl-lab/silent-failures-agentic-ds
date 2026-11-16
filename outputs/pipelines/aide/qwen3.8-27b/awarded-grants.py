import os
import re
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

warnings.filterwarnings("ignore")

TARGET = "Financial Assistance"
TRAIN_PATH = "./input/train.parquet"
TEST_PATH = "./input/test.parquet"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_parquet(TRAIN_PATH)
test = pd.read_parquet(TEST_PATH)


def robust_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype(float)

    s = series.astype(object)
    s = s.where(s.notna(), None)
    s = s.astype(str)
    s = (
        s.str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    s = s.replace({"": None, "nan": None, "None": None, "null": None, "NULL": None})
    return pd.to_numeric(s, errors="coerce").astype(float)


def is_identifier_column(col: str) -> bool:
    c = str(col).strip().lower()
    if c in {"id", "index", "row_id", "rowid", "unnamed"}:
        return True
    if c.startswith("unnamed"):
        return True
    if (
        c.endswith("_id")
        or c.endswith(" id")
        or c.endswith("_index")
        or c.endswith(" index")
    ):
        return True
    if re.search(r"(^|[\s_\-])id($|[\s_\-])", c):
        return True
    return False


def clean_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if TARGET in df.columns:
        df[TARGET] = robust_numeric(df[TARGET])
        df = df.dropna(subset=[TARGET])
    return df


train = clean_target(train)
test = clean_target(test)

feature_cols = [
    col for col in train.columns if col != TARGET and not is_identifier_column(col)
]

feature_types = {}
date_derived_cols = []

for col in feature_cols:
    s = train[col]

    if pd.api.types.is_bool_dtype(s):
        feature_types[col] = "numeric"
        continue

    if pd.api.types.is_numeric_dtype(s):
        feature_types[col] = "numeric"
        continue

    if pd.api.types.is_datetime64_any_dtype(s):
        feature_types[col] = "date"
        continue

    # Object/category/string-like column.
    numeric_like = robust_numeric(s)
    non_null_count = int(s.notna().sum())

    if non_null_count > 0:
        numeric_ratio = numeric_like.notna().sum() / non_null_count
        if numeric_ratio > 0.5:
            feature_types[col] = "numeric"
            continue

        date_like = pd.to_datetime(s, errors="coerce")
        date_ratio = date_like.notna().sum() / non_null_count
        if date_ratio > 0.5:
            feature_types[col] = "date"
            continue

    feature_types[col] = "categorical"

for col, typ in feature_types.items():
    if typ == "date":
        for attr in ["year", "month", "day", "dayofweek"]:
            date_derived_cols.append(f"{col}_{attr}")

final_feature_cols = [
    col for col in feature_cols if feature_types[col] != "date"
] + date_derived_cols
numeric_cols = [
    col
    for col in final_feature_cols
    if col in feature_types and feature_types[col] == "numeric"
]
categorical_cols = [
    col
    for col in final_feature_cols
    if col in feature_types and feature_types[col] == "categorical"
]


def transform_base(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in feature_cols:
        if col not in out.columns:
            out[col] = np.nan

        typ = feature_types[col]

        if typ == "numeric":
            out[col] = robust_numeric(out[col])

        elif typ == "date":
            dt = pd.to_datetime(out[col], errors="coerce")
            out[f"{col}_year"] = dt.dt.year
            out[f"{col}_month"] = dt.dt.month
            out[f"{col}_day"] = dt.dt.day
            out[f"{col}_dayofweek"] = dt.dt.dayofweek
            out = out.drop(columns=[col])

        else:
            s = out[col].astype(object)
            s = s.where(s.notna(), "missing")
            s = s.astype(str)
            s = s.replace({"": "missing", "nan": "missing", "None": "missing"})
            out[col] = s

    for col in final_feature_cols:
        if col not in out.columns:
            out[col] = np.nan

    return out[final_feature_cols]


X_train_base = transform_base(train)
X_test_base = transform_base(test)
y_train = train[TARGET].to_numpy(dtype=float)


def prepare_fold(
    X_tr: pd.DataFrame,
    X_val: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
):
    X_tr = X_tr.copy()
    X_val = X_val.copy()

    for col in numeric_cols:
        tr_series = X_tr[col].replace([np.inf, -np.inf], np.nan)
        median = tr_series.median()
        if pd.isna(median):
            median = 0.0

        X_tr[col] = tr_series.fillna(median)
        X_val[col] = X_val[col].replace([np.inf, -np.inf], np.nan).fillna(median)

    for col in categorical_cols:
        cats = list(pd.unique(X_tr[col]))
        if "missing" not in cats:
            cats.append("missing")
        cats = sorted(cats)

        X_tr[col] = pd.Categorical(X_tr[col], categories=cats)
        X_val[col] = X_val[col].where(X_val[col].isin(cats), "missing")
        X_val[col] = pd.Categorical(X_val[col], categories=cats)

    return X_tr, X_val


def prepare_final(
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
):
    X_tr = X_tr.copy()
    X_te = X_te.copy()

    for col in numeric_cols:
        tr_series = X_tr[col].replace([np.inf, -np.inf], np.nan)
        median = tr_series.median()
        if pd.isna(median):
            median = 0.0

        X_tr[col] = tr_series.fillna(median)
        X_te[col] = X_te[col].replace([np.inf, -np.inf], np.nan).fillna(median)

    for col in categorical_cols:
        cats = list(pd.unique(X_tr[col]))
        if "missing" not in cats:
            cats.append("missing")
        cats = sorted(cats)

        X_tr[col] = pd.Categorical(X_tr[col], categories=cats)
        X_te[col] = X_te[col].where(X_te[col].isin(cats), "missing")
        X_te[col] = pd.Categorical(X_te[col], categories=cats)

    return X_tr, X_te


params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}

NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 50

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(y_train), dtype=float)
best_iterations = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_base), start=1):
    X_tr, X_val = prepare_fold(
        X_train_base.iloc[train_idx],
        X_train_base.iloc[val_idx],
        numeric_cols,
        categorical_cols,
    )
    y_tr = y_train[train_idx]
    y_val = y_train[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=categorical_cols)
    valid_data = lgb.Dataset(
        X_val,
        label=y_val,
        categorical_feature=categorical_cols,
        reference=train_data,
    )

    model = lgb.train(
        params,
        train_data,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[valid_data],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(-1)],
    )

    best_iter = model.best_iteration
    if best_iter is None or best_iter <= 0:
        best_iter = NUM_BOOST_ROUND

    best_iterations.append(int(best_iter))
    oof_predictions[val_idx] = model.predict(X_val, num_iteration=best_iter)

    print(f"Fold {fold}: best_iteration={best_iter}")

if y_train.min() >= 0:
    oof_predictions = np.clip(oof_predictions, 0.0, None)

validation_rmse = root_mean_squared_error(y_train, oof_predictions)
print(f"Validation RMSE: {validation_rmse:.6f}")

final_rounds = int(np.clip(np.mean(best_iterations), 100, NUM_BOOST_ROUND))

X_train_final, X_test_final = prepare_final(
    X_train_base,
    X_test_base,
    numeric_cols,
    categorical_cols,
)

final_model = lgb.train(
    params,
    lgb.Dataset(X_train_final, label=y_train, categorical_feature=categorical_cols),
    num_boost_round=final_rounds,
    callbacks=[lgb.log_evaluation(-1)],
)

test_predictions = final_model.predict(X_test_final, num_iteration=final_rounds)

if y_train.min() >= 0:
    test_predictions = np.clip(test_predictions, 0.0, None)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        TARGET: test_predictions.astype(float),
    }
)

submission_path = os.path.join(WORK_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)

print(f"Saved submission to {submission_path}")
print(submission.head())
