import os
import re
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
from itertools import combinations

warnings.filterwarnings("ignore")

WORKING = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

TARGET = "Bus age"
SEED = 42
np.random.seed(SEED)

MISSING_MAP = {
    "nan": "__MISSING__",
    "None": "__MISSING__",
    "": "__MISSING__",
    "null": "__MISSING__",
    "NULL": "__MISSING__",
}


def sanitize(name):
    s = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_")
    return s if s else "feature"


def to_numeric_series(s):
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        return s

    s2 = s.astype(str).str.strip()
    s2 = s2.replace(
        {
            "": np.nan,
            "nan": np.nan,
            "NaN": np.nan,
            "None": np.nan,
            "null": np.nan,
            "NULL": np.nan,
        }
    )
    s2 = s2.str.replace(r"[\$,]", "", regex=True)
    s2 = s2.str.replace(",", "", regex=False)
    return pd.to_numeric(s2, errors="coerce")


def is_id_col(c):
    lc = str(c).lower().strip()
    return (
        lc
        in {
            "id",
            "index",
            "idx",
            "row_id",
            "rowid",
            "key",
            "identifier",
            "bus_id",
            "vehicle_id",
            "fleet_id",
            "school_id",
        }
        or lc.startswith("id_")
        or lc.endswith("_id")
    )


train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

if TARGET not in train.columns:
    for c in train.columns:
        if str(c).lower().strip() == "bus age":
            TARGET = c
            break

train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce").replace(
    [np.inf, -np.inf], np.nan
)
train = train.dropna(subset=[TARGET]).reset_index(drop=True)

train_id_cols = [c for c in train.columns if is_id_col(c) and c != TARGET]
test_id_cols = [c for c in test.columns if is_id_col(c)]

train = train.drop(columns=train_id_cols, errors="ignore").reset_index(drop=True)
test = test.drop(columns=test_id_cols, errors="ignore").reset_index(drop=True)

# Task specification: id is the 0-based integer row index of the test file.
test_id = pd.DataFrame({"id": np.arange(len(test), dtype=int)})

feature_cols = [c for c in train.columns if c != TARGET]
if len(feature_cols) == 0:
    feature_cols = ["dummy_feature"]
    train["dummy_feature"] = 0
    test["dummy_feature"] = 0

test = test.reindex(columns=feature_cols)

# Split first, then fit all feature-engineering statistics on train_part only.
if len(train) > 1:
    val_size = 0.2
    try:
        strat = pd.qcut(train[TARGET], q=5, labels=False, duplicates="drop")
        if strat.nunique() > 1:
            train_part, val_part = train_test_split(
                train,
                test_size=val_size,
                random_state=SEED,
                stratify=strat,
            )
        else:
            raise ValueError
    except Exception:
        train_part, val_part = train_test_split(
            train,
            test_size=val_size,
            random_state=SEED,
        )

    train_part = train_part.reset_index(drop=True)
    val_part = val_part.reset_index(drop=True)
else:
    train_part = train.copy().reset_index(drop=True)
    val_part = train.iloc[:0].copy().reset_index(drop=True)

if len(val_part) == 0:
    val_part = train_part.copy()

y_train = train_part[TARGET].astype(float).reset_index(drop=True)
y_val = val_part[TARGET].astype(float).reset_index(drop=True)

# -----------------------------------------------------------------------------
# Type detection and train-only statistics.
# -----------------------------------------------------------------------------
temp_train = train_part[feature_cols].copy()
numeric_cols = []
categorical_cols = []

for c in feature_cols:
    s = temp_train[c]
    if pd.api.types.is_bool_dtype(s):
        numeric_cols.append(c)
        temp_train[c] = s.astype(int)
    elif pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(c)
    else:
        conv = to_numeric_series(s)
        non_missing = s.notna().sum()
        if conv.notna().sum() > 0 and (
            non_missing == 0 or conv.notna().sum() / max(non_missing, 1) >= 0.5
        ):
            numeric_cols.append(c)
            temp_train[c] = conv
        else:
            categorical_cols.append(c)

if numeric_cols:
    temp_train[numeric_cols] = temp_train[numeric_cols].replace(
        [np.inf, -np.inf], np.nan
    )

cat_train = {}
for c in categorical_cols:
    s = train_part[c].astype(str).str.strip().replace(MISSING_MAP)
    cat_train[c] = s

numeric_stats = {}
for c in numeric_cols:
    s = temp_train[c]
    finite = s.dropna()

    median = float(finite.median()) if len(finite) else 0.0
    q25 = float(finite.quantile(0.25)) if len(finite) else 0.0
    q75 = float(finite.quantile(0.75)) if len(finite) else 0.0
    iqr = q75 - q25
    if not np.isfinite(iqr) or iqr == 0:
        iqr = 1.0

    min_val = float(finite.min()) if len(finite) else 0.0
    max_val = float(finite.max()) if len(finite) else 0.0
    skew = (
        float(finite.skew()) if len(finite) > 2 and np.isfinite(finite.skew()) else 0.0
    )

    numeric_stats[c] = {
        "median": median,
        "iqr": iqr,
        "min": min_val,
        "max": max_val,
        "skew": skew,
        "missing": float(s.isna().mean()),
    }

global_mean = float(y_train.mean()) if len(y_train) else 0.0

cat_stats = {}
for c in categorical_cols:
    s = cat_train[c]
    counts = s.value_counts(dropna=False)
    ord_map = counts.rank(method="first", ascending=False).astype(int)
    freq_map = counts / max(len(s), 1)

    grouped = y_train.groupby(s)
    cat_mean = grouped.mean()
    cat_count = grouped.size()

    m = 20.0
    target_enc_map = (cat_count * cat_mean + m * global_mean) / (cat_count + m)
    target_enc_map = target_enc_map.reindex(counts.index, fill_value=global_mean)

    cat_stats[c] = {
        "ord": ord_map,
        "freq": freq_map,
        "target": target_enc_map,
        "nunique": len(counts),
    }

top_numeric = []
if numeric_cols and len(y_train) > 1:
    corr = temp_train[numeric_cols].corrwith(y_train).abs()
    corr = corr.dropna()
    if len(corr) > 0:
        top_numeric = corr[corr > 0.05].nlargest(min(5, len(corr))).index.tolist()

year_cols = []
for c in numeric_cols:
    s = temp_train[c].dropna()
    if len(s) == 0:
        continue

    name = str(c).lower()
    if "year" in name or "date" in name:
        year_cols.append(c)
    elif ((s >= 1800) & (s <= 2100)).mean() > 0.5:
        year_cols.append(c)

date_cols = []
for c in categorical_cols:
    if "date" in str(c).lower() or "year" in str(c).lower():
        parsed = pd.to_datetime(train_part[c], errors="coerce")
        if parsed.notna().mean() > 0.5:
            date_cols.append(c)

# Precompute train-only age estimate offsets for strongly negatively correlated year-like features.
year_age_est = {}
for c in year_cols:
    corr_val = temp_train[c].corr(y_train) if len(y_train) > 1 else np.nan
    if np.isfinite(corr_val) and corr_val < -0.2:
        offset = np.nanmedian(y_train.values + temp_train[c].values)
        if np.isfinite(offset):
            year_age_est[c] = float(offset)


def transform_df(df):
    df = df.reindex(columns=feature_cols)
    out = pd.DataFrame(index=df.index)

    for c in numeric_cols:
        s = to_numeric_series(df[c])
        s = s.replace([np.inf, -np.inf], np.nan)
        st = numeric_stats[c]

        missing = s.isna().astype(int)
        s_imp = s.fillna(st["median"])

        out[c] = s_imp
        out[c + "_missing"] = missing
        out[c + "_scaled"] = (s_imp - st["median"]) / st["iqr"]

        if abs(st["skew"]) > 0.5:
            shift = max(0.0, st["min"] - 1.0) if st["min"] < 0 else 0.0
            vals = (s_imp - shift).clip(lower=0.0)
            out[c + "_log"] = np.log1p(vals)

    for c in categorical_cols:
        s = df[c].astype(str).str.strip().replace(MISSING_MAP)
        st = cat_stats[c]

        out[c + "_missing"] = (s == "__MISSING__").astype(int)
        out[c + "_ord"] = s.map(st["ord"]).fillna(-1).astype(int)
        out[c + "_freq"] = s.map(st["freq"]).fillna(0.0).astype(float)
        out[c + "_target_enc"] = s.map(st["target"]).fillna(global_mean).astype(float)
        out[c + "_len"] = s.str.len().fillna(0).astype(int)
        out[c + "_words"] = s.str.split().str.len().fillna(0).astype(int)

        if c in date_cols:
            parsed = pd.to_datetime(df[c], errors="coerce")
            out[c + "_year"] = parsed.dt.year.fillna(0).astype(int)
            out[c + "_month"] = parsed.dt.month.fillna(0).astype(int)
            out[c + "_day"] = parsed.dt.day.fillna(0).astype(int)

    for c in year_cols:
        s = out[c]
        out[c + "_decade"] = (s // 10) * 10
        out[c + "_mod100"] = s % 100

        if c in year_age_est:
            out[c + "_age_est"] = year_age_est[c] - s

    for a, b in combinations(year_cols, 2):
        out[f"{a}_minus_{b}"] = out[a] - out[b]

    for a, b in combinations(top_numeric, 2):
        out[f"{a}_plus_{b}"] = out[a] + out[b]
        out[f"{a}_times_{b}"] = out[a] * out[b]
        denom = out[b].replace(0, np.nan)
        out[f"{a}_div_{b}"] = (out[a] / denom).fillna(0.0)

    return out


def sanitize_columns(df):
    df = df.copy()
    df.columns = [sanitize(c) for c in df.columns]

    cols = []
    seen = set()
    for c in df.columns:
        if c in seen:
            i = 1
            new = f"{c}_{i}"
            while new in seen:
                i += 1
                new = f"{c}_{i}"
            c = new
        seen.add(c)
        cols.append(c)

    df.columns = cols
    return df


X_train = sanitize_columns(transform_df(train_part))
X_val = sanitize_columns(transform_df(val_part))
X_test = sanitize_columns(transform_df(test))

all_cols = list(X_train.columns)
existing = set(all_cols)

for c in X_val.columns:
    if c not in existing:
        all_cols.append(c)
        existing.add(c)

for c in X_test.columns:
    if c not in existing:
        all_cols.append(c)
        existing.add(c)

if len(all_cols) == 0:
    X_train = pd.DataFrame({"constant_feature": 1.0}, index=X_train.index)
    X_val = pd.DataFrame({"constant_feature": 1.0}, index=X_val.index)
    X_test = pd.DataFrame({"constant_feature": 1.0}, index=X_test.index)
else:
    X_train = X_train.reindex(columns=all_cols, fill_value=0)
    X_val = X_val.reindex(columns=all_cols, fill_value=0)
    X_test = X_test.reindex(columns=all_cols, fill_value=0)

X_train = X_train.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
X_val = X_val.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
X_test = X_test.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

# Persist intermediate artifacts for auditability.
X_train.to_parquet(os.path.join(WORKING, "X_train.parquet"), index=False)
X_val.to_parquet(os.path.join(WORKING, "X_val.parquet"), index=False)
X_test.to_parquet(os.path.join(WORKING, "X_test.parquet"), index=False)
y_train.to_frame(TARGET).to_parquet(
    os.path.join(WORKING, "y_train.parquet"), index=False
)
y_val.to_frame(TARGET).to_parquet(os.path.join(WORKING, "y_val.parquet"), index=False)
test_id.to_parquet(os.path.join(WORKING, "test_id.parquet"), index=False)

MODEL_DIR = os.path.join(WORKING, "model_design")
TRAINING_DIR = os.path.join(WORKING, "training_evaluation")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TRAINING_DIR, exist_ok=True)


def build_monotone_constraints(feature_names):
    """
    Domain constraint:
    - Larger calendar/model/purchase year should reduce bus age.
    - Explicit age estimates should increase bus age.
    - Missing indicators, cyclic modulo terms, and interaction terms are left unconstrained.
    """
    constraints = [0] * len(feature_names)
    derived_tokens = (
        "_missing",
        "_minus_",
        "_plus_",
        "_times_",
        "_div_",
        "_ord",
        "_freq",
        "_target_enc",
        "_len",
        "_words",
        "_mod",
        "cycle",
    )

    for i, name in enumerate(feature_names):
        n = str(name).lower()

        if any(tok in n for tok in derived_tokens):
            constraints[i] = 0
            continue

        if "age_est" in n or "bus_age" in n or n.endswith("_age") or "_age" in n:
            constraints[i] = 1
            continue

        year_like = (
            "year" in n
            or "manufacture" in n
            or "model_year" in n
            or "purchase" in n
            or "acquisition" in n
        )

        if year_like:
            constraints[i] = -1

    return constraints


feature_names = list(X_train.columns)
monotone_constraints = build_monotone_constraints(feature_names)
n_constraints = int(sum(1 for c in monotone_constraints if c != 0))

model = lgb.LGBMRegressor(
    objective="regression_l2",
    n_estimators=3000,
    learning_rate=0.02,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
    monotone_constraints=monotone_constraints,
    monotone_constraints_method="advanced",
    feature_pre_filter=False,
)

with open(os.path.join(MODEL_DIR, "model_config.json"), "w") as f:
    json.dump(
        {
            "architecture": "LightGBM GBDT regressor",
            "loss": "regression_l2 (squared error, aligned with RMSE)",
            "optimizer": "early stopping + learning rate schedule via LightGBM boosting",
            "monotone_constraints": dict(zip(feature_names, monotone_constraints)),
            "n_monotone_constraints": n_constraints,
        },
        f,
        indent=2,
        default=str,
    )

metric_history = []
best_metric = {"val_rmse": np.inf, "iteration": 0}


def training_callback(env):
    if env.evaluation_result_list:
        val_rmse = float(env.evaluation_result_list[0][2])
    else:
        val_rmse = np.nan

    metric_history.append({"iteration": int(env.iteration), "val_rmse": val_rmse})

    if np.isfinite(val_rmse) and val_rmse < best_metric["val_rmse"] - 1e-12:
        best_metric["val_rmse"] = val_rmse
        best_metric["iteration"] = int(env.iteration)


early_stopping = lgb.early_stopping(
    stopping_rounds=50, first_metric_only=True, verbose=False
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    callbacks=[early_stopping, training_callback, lgb.log_evaluation(-1)],
)

booster = getattr(model, "booster_", model)
booster.save_model(os.path.join(TRAINING_DIR, "model_best.json"))

best_iteration = getattr(model, "best_iteration_", None)
if best_iteration is None or best_iteration <= 0:
    best_iteration = getattr(model, "n_estimators_", None)

training_summary = {
    "task": "regression",
    "target": TARGET,
    "metric": "rmse",
    "split": "stratified 80/20 holdout prepared before feature engineering",
    "model": "LightGBM GBDT regressor with monotonic constraints",
    "best_iteration": int(best_iteration) if best_iteration is not None else None,
    "best_iteration_from_tracker": int(best_metric["iteration"]),
    "final_validation_rmse": None,
    "n_train": int(len(X_train)),
    "n_val": int(len(X_val)),
    "n_test": int(len(X_test)),
    "n_features": int(X_train.shape[1]),
}

with open(os.path.join(TRAINING_DIR, "training_summary.json"), "w") as f:
    json.dump(training_summary, f, indent=2, default=str)


def predict_with_best_iteration(model, X):
    booster = getattr(model, "booster_", model)
    best_iter = getattr(model, "best_iteration_", None)

    if best_iter is not None and best_iter > 0:
        try:
            max_iter = booster.num_trees()
        except Exception:
            max_iter = None

        if max_iter is not None and max_iter > 0:
            best_iter = min(int(best_iter), int(max_iter))

        if best_iter > 0:
            return booster.predict(X, num_iteration=int(best_iter))

    return booster.predict(X)


def postprocess_predictions(pred, y_ref):
    pred = np.asarray(pred, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)
    finite_y = y_ref[np.isfinite(y_ref)]

    if len(finite_y) > 0:
        nan_fill = float(np.mean(finite_y))
        posinf_fill = float(np.max(finite_y))
        neginf_fill = float(np.min(finite_y))
    else:
        nan_fill = 0.0
        posinf_fill = 0.0
        neginf_fill = 0.0

    pred = np.nan_to_num(pred, nan=nan_fill, posinf=posinf_fill, neginf=neginf_fill)

    # Bus age is non-negative when observed target is non-negative.
    if len(finite_y) > 0 and np.min(finite_y) >= 0:
        pred = np.clip(pred, 0.0, None)

    return pred


val_pred = postprocess_predictions(predict_with_best_iteration(model, X_val), y_train)
test_pred = postprocess_predictions(predict_with_best_iteration(model, X_test), y_train)

score = root_mean_squared_error(y_val, val_pred)

training_summary["final_validation_rmse"] = float(score)
with open(os.path.join(TRAINING_DIR, "training_summary.json"), "w") as f:
    json.dump(training_summary, f, indent=2, default=str)

submission = pd.DataFrame(
    {
        "id": test_id["id"].astype(int).values,
        TARGET: np.asarray(test_pred, dtype=float),
    }
)
submission.to_csv(os.path.join(SUBMISSION_DIR, "submission.csv"), index=False)

print(f"Final Validation Score: {score}")
