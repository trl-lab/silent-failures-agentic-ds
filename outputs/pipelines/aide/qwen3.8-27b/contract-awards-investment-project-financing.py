import os
import re
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

warnings.filterwarnings("ignore")

TARGET = "Supplier Contract Amount (USD)"
RANDOM_STATE = 42
N_SPLITS = 5
EARLY_STOPPING_ROUNDS = 50
MAX_ROUNDS = 5000

os.makedirs("./working", exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Robust target cleaning.
train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
train = train.dropna(subset=[TARGET]).reset_index(drop=True)
train = train[np.isfinite(train[TARGET])].reset_index(drop=True)

y_raw = train[TARGET].astype(float)
use_log = bool((y_raw >= 0).all())
y = np.log1p(y_raw) if use_log else y_raw.values


def is_id_like(col: str) -> bool:
    c = str(col).lower().strip()
    if c in {"id", "index", "row_id", "rowid", "uuid", "guid", "key", "identifier"}:
        return True
    if (
        c.endswith("_id")
        or c.endswith(" id")
        or c.endswith("_index")
        or c.endswith(" index")
    ):
        return True
    if "contract number" in c or "contract id" in c:
        return True
    return False


# Use only columns present in both train and test, excluding target and obvious IDs.
feature_cols = [
    c for c in train.columns if c != TARGET and c in test.columns and not is_id_like(c)
]

# Fallback if no usable features are found.
if len(feature_cols) == 0:
    feature_cols = ["__dummy__"]
    train["__dummy__"] = 0
    test["__dummy__"] = 0


def make_feature_names(cols):
    names = []
    seen = {}
    for i, c in enumerate(cols):
        n = re.sub(r"[^0-9a-zA-Z]+", "_", str(c)).strip("_")
        if not n:
            n = f"feature_{i}"
        if n in seen:
            seen[n] += 1
            n = f"{n}_{seen[n]}"
        else:
            seen[n] = 0
        names.append(n)
    return names


feature_names = make_feature_names(feature_cols)


def clean_numeric_series(series: pd.Series) -> pd.Series:
    """
    Try to convert a series to numeric, removing common currency/format characters.
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    s = series.astype("string")
    # Convert accounting-style negatives, e.g. "(1,234)" -> "-1234".
    s = s.str.replace(r"\((.+?)\)", r"-\1", regex=True)
    # Remove currency symbols, commas, and whitespace.
    s = s.str.replace(r"[$,\s]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def is_numeric_like(series: pd.Series) -> bool:
    s = series.dropna()
    if len(s) == 0:
        return False
    sample = s.sample(min(200, len(s)), random_state=RANDOM_STATE)
    parsed = clean_numeric_series(sample)
    return parsed.notna().mean() > 0.8


def is_date_like(series: pd.Series) -> bool:
    s = series.dropna()
    if len(s) == 0:
        return False
    sample = s.astype(str).sample(min(200, len(s)), random_state=RANDOM_STATE)
    try:
        parsed = pd.to_datetime(sample, errors="coerce")
        return parsed.notna().mean() > 0.8
    except Exception:
        return False


def datetime_to_days(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    if dt.notna().any():
        days = dt.astype("int64") // 86_400_000_000_000
        return days.where(dt.notna(), np.nan).astype(float)
    return pd.Series(np.nan, index=series.index, dtype=float)


def preprocess(X_train: pd.DataFrame, X_other: pd.DataFrame):
    """
    Fit preprocessing on X_train and transform both X_train and X_other.
    Returns processed DataFrames and list of categorical feature names.
    """
    Xtr = X_train[feature_cols].copy()
    Xot = X_other[feature_cols].copy()

    categorical_cols = []

    for orig, new in zip(feature_cols, feature_names):
        tr = Xtr[orig]
        ot = Xot[orig]

        # Already datetime.
        if pd.api.types.is_datetime64_any_dtype(
            tr
        ) or pd.api.types.is_datetime64_any_dtype(ot):
            Xtr[new] = datetime_to_days(tr)
            Xot[new] = datetime_to_days(ot)
            continue

        # Numeric dtype.
        if pd.api.types.is_numeric_dtype(tr) and pd.api.types.is_numeric_dtype(ot):
            Xtr[new] = pd.to_numeric(tr, errors="coerce")
            Xot[new] = pd.to_numeric(ot, errors="coerce")
            continue

        # Object / string / category.
        if (
            tr.dtype == object
            or ot.dtype == object
            or pd.api.types.is_string_dtype(tr)
            or pd.api.types.is_string_dtype(ot)
            or pd.api.types.is_categorical_dtype(tr)
            or pd.api.types.is_categorical_dtype(ot)
        ):
            # Try numeric conversion first.
            if is_numeric_like(tr) or is_numeric_like(ot):
                Xtr[new] = clean_numeric_series(tr)
                Xot[new] = clean_numeric_series(ot)
                continue

            # Try date conversion.
            if is_date_like(tr) or is_date_like(ot):
                Xtr[new] = datetime_to_days(tr)
                Xot[new] = datetime_to_days(ot)
                continue

            # Categorical ordinal encoding fitted on training fold only.
            tr_vals = tr.astype("string")
            ot_vals = ot.astype("string")

            categories = sorted(tr_vals.dropna().unique())
            mapping = {cat: i for i, cat in enumerate(categories)}

            Xtr[new] = tr_vals.map(mapping).fillna(-1).astype("int32")
            Xot[new] = ot_vals.map(mapping).fillna(-1).astype("int32")
            categorical_cols.append(new)
            continue

        # Fallback: force numeric.
        Xtr[new] = pd.to_numeric(tr, errors="coerce")
        Xot[new] = pd.to_numeric(ot, errors="coerce")

    # Impute numeric features using training-fold medians.
    for col in feature_names:
        if col in categorical_cols:
            Xtr[col] = Xtr[col].fillna(-1).astype("int32")
            Xot[col] = Xot[col].fillna(-1).astype("int32")
        else:
            Xtr[col] = Xtr[col].replace([np.inf, -np.inf], np.nan)
            Xot[col] = Xot[col].replace([np.inf, -np.inf], np.nan)

            median = Xtr[col].median()
            if pd.isna(median):
                median = 0.0

            Xtr[col] = Xtr[col].fillna(median).astype(float)
            Xot[col] = Xot[col].fillna(median).astype(float)

    return Xtr[feature_names], Xot[feature_names], categorical_cols


params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "num_threads": -1,
    "verbosity": -1,
}


def inverse_transform(pred: np.ndarray) -> np.ndarray:
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    if use_log:
        pred = np.clip(pred, -700.0, 700.0)
        pred = np.expm1(pred)
        pred = np.clip(pred, 0.0, None)
    return pred


def make_lgb_dataset(X: pd.DataFrame, label: np.ndarray, cat_cols: list):
    categorical_feature = cat_cols if len(cat_cols) > 0 else "auto"
    return lgb.Dataset(
        X,
        label=label,
        categorical_feature=categorical_feature,
    )


kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
rmse_scores = []
best_iterations = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(train), start=1):
    X_tr_raw = train.loc[tr_idx]
    X_va_raw = train.loc[va_idx]

    X_tr, X_va, cat_cols = preprocess(X_tr_raw, X_va_raw)

    dtrain = make_lgb_dataset(X_tr, y.values[tr_idx], cat_cols)
    dvalid = make_lgb_dataset(X_va, y.values[va_idx], cat_cols)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    va_pred = model.predict(X_va, num_iteration=model.best_iteration)
    va_pred = inverse_transform(va_pred)

    rmse = root_mean_squared_error(y_raw.values[va_idx], va_pred)
    rmse_scores.append(rmse)

    bi = getattr(model, "best_iteration", 0)
    if bi <= 0:
        bi = 1500
    best_iterations.append(bi)

    print(f"Fold {fold} RMSE: {rmse:.6f}")

mean_rmse = float(np.mean(rmse_scores))
std_rmse = float(np.std(rmse_scores))
print(f"Validation RMSE: {mean_rmse:.6f} (+/- {std_rmse:.6f})")

# Final model on full training data.
X_train_full, X_test, cat_cols_full = preprocess(train, test)

full_rounds = int(np.clip(np.mean(best_iterations), 100, MAX_ROUNDS))

dtrain_full = make_lgb_dataset(X_train_full, y.values, cat_cols_full)

final_model = lgb.train(
    params,
    dtrain_full,
    num_boost_round=full_rounds,
    callbacks=[lgb.log_evaluation(-1)],
)

test_pred = final_model.predict(X_test)
test_pred = inverse_transform(test_pred)

submission = pd.DataFrame({"id": np.arange(len(test)), TARGET: test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Saved ./working/submission.csv")
