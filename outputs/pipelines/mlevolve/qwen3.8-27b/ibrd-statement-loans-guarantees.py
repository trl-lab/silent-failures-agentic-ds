import os
import re
import warnings
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

import lightgbm as lgb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# -----------------------------
# Reproducibility and paths
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

TARGET = "Original Principal Amount (US$)"
RANDOM_STATE = 42
VAL_FRACTION = 0.15
SMOOTHING = 20.0
TE_FOLDS = 5
LOW_CARD_LIMIT = 10
HIGH_MISSING_DROP = 0.95

# -----------------------------
# Load data
# -----------------------------
train = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
test = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

# Robust target discovery
if TARGET not in train.columns:
    candidates = [c for c in train.columns if "Original Principal" in c]
    if candidates:
        TARGET = candidates[0]
    else:
        raise ValueError("Target column not found in train data.")

# Clean target and drop unusable training rows before splitting
train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
train = train.replace([np.inf, -np.inf], np.nan)
train = train.dropna(subset=[TARGET]).reset_index(drop=True)

# Test ids must be 0-based row index per task statement
test_ids = np.arange(len(test))

# Identify id-like columns to exclude from features
id_like = {
    "id",
    "index",
    "row_id",
    "rowid",
    "loan_id",
    "guarantee_id",
    "application_id",
    "request_id",
    "statement_id",
}


def is_id_col(col: str) -> bool:
    c = str(col).strip().lower()
    return c in id_like or c.startswith("id_") or c.endswith("_id")


drop_cols = [c for c in train.columns if is_id_col(c) or c == TARGET]
drop_cols = [c for c in drop_cols if c in train.columns]

X_all = train.drop(columns=drop_cols, errors="ignore").copy()
y_all = train[TARGET].copy()

# -----------------------------
# Split FIRST: train / validation
# -----------------------------
try:
    strat = pd.qcut(y_all, q=10, duplicates="drop")
except Exception:
    strat = pd.cut(y_all.rank(method="first"), bins=10, duplicates="drop")

if len(y_all) < 20:
    val_size = max(1, int(round(len(y_all) * VAL_FRACTION)))
    val_size = min(max(1, val_size), len(y_all) - 1)
    train_idx, val_idx = train_test_split(
        np.arange(len(y_all)), test_size=val_size, random_state=RANDOM_STATE
    )
else:
    train_idx, val_idx = train_test_split(
        np.arange(len(y_all)),
        test_size=VAL_FRACTION,
        random_state=RANDOM_STATE,
        stratify=strat,
    )

X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
X_val_raw = X_all.iloc[val_idx].reset_index(drop=True)
y_train = y_all.iloc[train_idx].reset_index(drop=True)
y_val = y_all.iloc[val_idx].reset_index(drop=True)


# -----------------------------
# Helper utilities
# -----------------------------
def clean_col_name(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower() if s else "col"


def is_date_like(series: pd.Series) -> bool:
    if series.dtype == "datetime64[ns]":
        return True
    sample = series.dropna().astype(str).head(200)
    if len(sample) == 0:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() > 0.8


def is_numeric_like(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    sample = series.dropna().astype(str).head(500)
    if len(sample) == 0:
        return False
    parsed = pd.to_numeric(sample, errors="coerce")
    return parsed.notna().mean() > 0.8


def add_date_features(
    df: pd.DataFrame, date_cols: list, date_min: pd.Series
) -> pd.DataFrame:
    out = df.copy()
    for col in date_cols:
        if col not in out.columns:
            continue
        dt = pd.to_datetime(out[col], errors="coerce")
        base = clean_col_name(col)
        out[f"{base}_year"] = dt.dt.year
        out[f"{base}_month"] = dt.dt.month
        out[f"{base}_day"] = dt.dt.day
        out[f"{base}_dayofweek"] = dt.dt.dayofweek
        out[f"{base}_quarter"] = dt.dt.quarter
        # Days since train minimum date, fitted only on train
        ref = date_min.get(col, pd.Timestamp.min)
        out[f"{base}_days_since_train_min"] = (dt - ref).dt.days
        out = out.drop(columns=[col], errors="ignore")
    return out


def target_encode_fit_transform(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: pd.Series,
    col: str,
):
    """
    Smoothed K-fold target encoding for train rows.
    Validation/test use full train statistics only.
    """
    global_mean = float(y_train.mean())
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_vals = train_df[col].astype("string")
    val_vals = (
        val_df[col].astype("string")
        if col in val_df.columns
        else pd.Series(["missing"] * len(val_df), index=val_df.index)
    )
    test_vals = (
        test_df[col].astype("string")
        if col in test_df.columns
        else pd.Series(["missing"] * len(test_df), index=test_df.index)
    )

    # Full train stats for val/test
    tmp = pd.DataFrame(
        {"cat": train_vals.fillna("missing").values, "y": y_train.values}
    )
    agg = tmp.groupby("cat")["y"].agg(["count", "sum"])
    full_stats = (agg["sum"] + SMOOTHING * global_mean) / (agg["count"] + SMOOTHING)

    val_df[f"{col}_te"] = (
        val_vals.fillna("missing").map(full_stats).fillna(global_mean).astype(float)
    )
    test_df[f"{col}_te"] = (
        test_vals.fillna("missing").map(full_stats).fillna(global_mean).astype(float)
    )

    # K-fold train encoding to avoid row-level target leakage
    train_te = np.full(len(train_df), global_mean, dtype=float)
    kf = KFold(n_splits=TE_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for tr_fold, va_fold in kf.split(train_df):
        fold_tmp = pd.DataFrame(
            {
                "cat": train_vals.iloc[tr_fold].fillna("missing").values,
                "y": y_train.iloc[tr_fold].values,
            }
        )
        fold_agg = fold_tmp.groupby("cat")["y"].agg(["count", "sum"])
        fold_stats = (fold_agg["sum"] + SMOOTHING * global_mean) / (
            fold_agg["count"] + SMOOTHING
        )
        train_te[va_fold] = (
            train_vals.iloc[va_fold]
            .fillna("missing")
            .map(fold_stats)
            .fillna(global_mean)
            .astype(float)
            .values
        )

    train_df[f"{col}_te"] = train_te
    stats = {
        "global_mean": global_mean,
        "category_stats": full_stats.to_dict(),
    }
    return train_df, val_df, test_df, stats


def frequency_encode(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, col: str
):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    freq = train_df[col].astype("string").fillna("missing").value_counts()
    train_df[f"{col}_freq"] = (
        train_df[col]
        .astype("string")
        .fillna("missing")
        .map(freq)
        .fillna(0)
        .astype(float)
    )
    val_df[f"{col}_freq"] = (
        val_df[col].astype("string").fillna("missing").map(freq).fillna(0).astype(float)
    )
    test_df[f"{col}_freq"] = (
        test_df[col]
        .astype("string")
        .fillna("missing")
        .map(freq)
        .fillna(0)
        .astype(float)
    )
    return train_df, val_df, test_df


def one_hot_encode(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, col: str
):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df[col] = train_df[col].astype("string").fillna("missing")
    val_df[col] = (
        val_df[col].astype("string").fillna("missing")
        if col in val_df.columns
        else pd.Series(["missing"] * len(val_df), index=val_df.index)
    )
    test_df[col] = (
        test_df[col].astype("string").fillna("missing")
        if col in test_df.columns
        else pd.Series(["missing"] * len(test_df), index=test_df.index)
    )

    dummies = pd.get_dummies(train_df[col], prefix=clean_col_name(col), dtype=float)
    train_df = pd.concat([train_df.drop(columns=[col]), dummies], axis=1)
    val_d = pd.get_dummies(val_df[col], prefix=clean_col_name(col), dtype=float)
    test_d = pd.get_dummies(test_df[col], prefix=clean_col_name(col), dtype=float)
    val_d = val_d.reindex(columns=dummies.columns, fill_value=0.0)
    test_d = test_d.reindex(columns=dummies.columns, fill_value=0.0)
    val_df = pd.concat([val_df.drop(columns=[col]), val_d], axis=1)
    test_df = pd.concat([test_df.drop(columns=[col]), test_d], axis=1)
    return train_df, val_df, test_df


# -----------------------------
# Classify columns using TRAIN split only
# -----------------------------
date_cols = []
numeric_cols = []
categorical_cols = []

for col in X_train_raw.columns:
    s = X_train_raw[col]
    if is_date_like(s):
        date_cols.append(col)
    elif is_numeric_like(s):
        numeric_cols.append(col)
    else:
        categorical_cols.append(col)

# Convert numeric columns in all datasets
for col in numeric_cols:
    if col not in X_val_raw.columns:
        X_val_raw[col] = np.nan
    if col not in test.columns:
        test[col] = np.nan
    X_train_raw[col] = pd.to_numeric(X_train_raw[col], errors="coerce")
    X_val_raw[col] = pd.to_numeric(X_val_raw[col], errors="coerce")
    test[col] = pd.to_numeric(test[col], errors="coerce")

# Convert categorical columns to string/category
for col in categorical_cols:
    if col not in X_val_raw.columns:
        X_val_raw[col] = "missing"
    if col not in test.columns:
        test[col] = "missing"
    X_train_raw[col] = X_train_raw[col].astype("string")
    X_val_raw[col] = X_val_raw[col].astype("string")
    test[col] = test[col].astype("string")

# Drop columns that are almost entirely missing in train, after missing indicators
high_missing_cols = [
    col
    for col in X_train_raw.columns
    if X_train_raw[col].isna().mean() > HIGH_MISSING_DROP
]

# -----------------------------
# Missing indicators
# -----------------------------
X_train = X_train_raw.copy()
X_val = X_val_raw.copy()
X_test = test.copy()

for col in X_train.columns:
    if col in drop_cols:
        continue
    miss = X_train[col].isna()
    if miss.any():
        name = f"{clean_col_name(col)}_missing"
        X_train[name] = miss.astype(float)
        X_val[name] = X_val[col].isna().astype(float) if col in X_val.columns else 1.0
        X_test[name] = (
            X_test[col].isna().astype(float) if col in X_test.columns else 1.0
        )

# -----------------------------
# Date features
# -----------------------------
date_min = pd.Series(dtype="datetime64[ns]")
for col in date_cols:
    parsed = pd.to_datetime(X_train[col], errors="coerce")
    if parsed.notna().any():
        date_min[col] = parsed.min()

X_train = add_date_features(X_train, date_cols, date_min)
X_val = add_date_features(X_val, date_cols, date_min)
X_test = add_date_features(X_test, date_cols, date_min)

# Remove original date columns if still present
for col in date_cols:
    X_train = X_train.drop(columns=[col], errors="ignore")
    X_val = X_val.drop(columns=[col], errors="ignore")
    X_test = X_test.drop(columns=[col], errors="ignore")

# -----------------------------
# Numeric feature engineering
# -----------------------------
numeric_features = [
    c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])
]

# Impute numeric columns with train median
numeric_medians = X_train[numeric_features].median(numeric_only=True)
numeric_medians = numeric_medians.fillna(0.0)

for df_name, df in [("train", X_train), ("val", X_val), ("test", X_test)]:
    for col in numeric_features:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(numeric_medians.get(col, 0.0))

# Add log and signed-log transforms for numeric features
for col in numeric_features:
    base = clean_col_name(col)
    X_train[f"{base}_log1p"] = np.log1p(np.maximum(X_train[col], 0.0))
    X_val[f"{base}_log1p"] = np.log1p(np.maximum(X_val[col], 0.0))
    X_test[f"{base}_log1p"] = np.log1p(np.maximum(X_test[col], 0.0))

    signed = np.sign(X_train[col]) * np.log1p(np.abs(X_train[col]))
    X_train[f"{base}_signed_log"] = signed
    X_val[f"{base}_signed_log"] = np.sign(X_val[col]) * np.log1p(np.abs(X_val[col]))
    X_test[f"{base}_signed_log"] = np.sign(X_test[col]) * np.log1p(np.abs(X_test[col]))

# Winsorized features using train quantiles
quantiles = X_train[numeric_features].quantile([0.01, 0.99], numeric_only=True)
for col in numeric_features:
    base = clean_col_name(col)
    q01 = quantiles.loc[0.01, col]
    q99 = quantiles.loc[0.99, col]
    if np.isfinite(q01) and np.isfinite(q99) and q99 > q01:
        X_train[f"{base}_winsor"] = X_train[col].clip(q01, q99)
        X_val[f"{base}_winsor"] = X_val[col].clip(q01, q99)
        X_test[f"{base}_winsor"] = X_test[col].clip(q01, q99)

# -----------------------------
# Categorical feature engineering
# -----------------------------
cat_card = X_train[categorical_cols].nunique(dropna=True)

for col in categorical_cols:
    if col not in X_train.columns:
        continue
    n_unique = int(cat_card.get(col, 0))

    # Frequency encoding for all categoricals
    X_train, X_val, X_test = frequency_encode(X_train, X_val, X_test, col)

    if n_unique <= LOW_CARD_LIMIT:
        X_train, X_val, X_test = one_hot_encode(X_train, X_val, X_test, col)
    else:
        X_train, X_val, X_test, _ = target_encode_fit_transform(
            X_train, X_val, X_test, y_train, col
        )
        X_train = X_train.drop(columns=[col], errors="ignore")
        X_val = X_val.drop(columns=[col], errors="ignore")
        X_test = X_test.drop(columns=[col], errors="ignore")

# Drop remaining original categorical columns
remaining_cats = [c for c in categorical_cols if c in X_train.columns]
X_train = X_train.drop(columns=remaining_cats, errors="ignore")
X_val = X_val.drop(columns=remaining_cats, errors="ignore")
X_test = X_test.drop(columns=remaining_cats, errors="ignore")

# Drop original high-missing columns if still present
for col in high_missing_cols:
    X_train = X_train.drop(columns=[col], errors="ignore")
    X_val = X_val.drop(columns=[col], errors="ignore")
    X_test = X_test.drop(columns=[col], errors="ignore")

# -----------------------------
# Final numeric cleanup and scaling
# -----------------------------
# Ensure all columns are numeric
X_train = X_train.apply(pd.to_numeric, errors="coerce")
X_val = X_val.apply(pd.to_numeric, errors="coerce")
X_test = X_test.apply(pd.to_numeric, errors="coerce")

# Drop all-NaN and constant columns based on train
cols_to_drop = []
for col in X_train.columns:
    if X_train[col].isna().all():
        cols_to_drop.append(col)
    elif X_train[col].nunique(dropna=True) <= 1:
        cols_to_drop.append(col)

X_train = X_train.drop(columns=cols_to_drop, errors="ignore")
X_val = X_val.reindex(columns=X_train.columns, fill_value=np.nan)
X_test = X_test.reindex(columns=X_train.columns, fill_value=np.nan)

# Fill any residual NaN with 0 after scaling-ready cleanup
X_train = X_train.fillna(0.0)
X_val = X_val.fillna(0.0)
X_test = X_test.fillna(0.0)

# Standard scaling fitted only on train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(
    X_train_scaled, columns=X_train.columns, index=X_train.index
)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_train.columns, index=X_val.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_train.columns, index=X_test.index)

# Replace any remaining non-finite values
X_train_scaled = (
    X_train_scaled.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)
)
X_val_scaled = (
    X_val_scaled.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)
)
X_test_scaled = (
    X_test_scaled.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)
)

# -----------------------------
# Persist engineered artifacts
# -----------------------------
X_train_scaled.to_parquet(os.path.join(WORKING_DIR, "X_train.parquet"))
X_val_scaled.to_parquet(os.path.join(WORKING_DIR, "X_val.parquet"))
X_test_scaled.to_parquet(os.path.join(WORKING_DIR, "X_test.parquet"))

y_train.to_frame(TARGET).to_parquet(os.path.join(WORKING_DIR, "y_train.parquet"))
y_val.to_frame(TARGET).to_parquet(os.path.join(WORKING_DIR, "y_val.parquet"))

pd.DataFrame({"id": test_ids}).to_parquet(os.path.join(WORKING_DIR, "test_ids.parquet"))
pd.DataFrame({"feature_names": X_train_scaled.columns.tolist()}).to_csv(
    os.path.join(WORKING_DIR, "feature_names.csv"), index=False
)


# -----------------------------
# Model design
# -----------------------------
LGBM_BASE_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 31,
    "min_child_samples": 10,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 3000,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}


def build_lgbm_model(seed: int) -> lgb.LGBMRegressor:
    params = LGBM_BASE_PARAMS.copy()
    params["random_state"] = int(seed)
    return lgb.LGBMRegressor(**params)


def postprocess(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(pred, 0.0, None)


# -----------------------------
# Training and evaluation
# -----------------------------
def main():
    X_train = pd.read_parquet(os.path.join(WORKING_DIR, "X_train.parquet"))
    X_val = pd.read_parquet(os.path.join(WORKING_DIR, "X_val.parquet"))
    X_test = pd.read_parquet(os.path.join(WORKING_DIR, "X_test.parquet"))
    y_train = pd.read_parquet(os.path.join(WORKING_DIR, "y_train.parquet"))[
        TARGET
    ].astype(float)
    y_val = pd.read_parquet(os.path.join(WORKING_DIR, "y_val.parquet"))[TARGET].astype(
        float
    )

    try:
        test_ids = (
            pd.read_parquet(os.path.join(WORKING_DIR, "test_ids.parquet"))["id"]
            .astype(int)
            .to_numpy()
        )
    except Exception:
        test_ids = np.arange(len(X_test), dtype=int)

    y_train_np = y_train.to_numpy(dtype=np.float32)
    y_val_np = y_val.to_numpy(dtype=np.float32)

    seeds = [42, 43, 44, 45, 46]
    val_preds = []
    test_preds = []

    for seed in seeds:
        model = build_lgbm_model(seed)
        model.fit(
            X_train,
            y_train_np,
            eval_set=[(X_val, y_val_np)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )

        best_iter = getattr(model, "best_iteration_", None)
        if best_iter is None or best_iter <= 0:
            best_iter = model.n_estimators

        val_pred = model.predict(X_val, num_iteration=best_iter)
        test_pred = model.predict(X_test, num_iteration=best_iter)

        val_preds.append(val_pred)
        test_preds.append(test_pred)
        print(f"Seed {seed} | best_iteration={best_iter}")

    val_pred = np.mean(val_preds, axis=0)
    test_pred = np.mean(test_preds, axis=0)

    val_pred = postprocess(val_pred)
    test_pred = postprocess(test_pred)

    score = float(root_mean_squared_error(y_val_np, val_pred))

    submission = pd.DataFrame(
        {
            "id": test_ids.astype(int),
            TARGET: test_pred.astype(float),
        }
    )
    submission.to_csv(os.path.join(SUBMISSION_DIR, "submission.csv"), index=False)

    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()
