import os
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

warnings.filterwarnings("ignore")

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

SEED = 42
TARGET = "number_of_officers"
np.random.seed(SEED)


def _sanitize(col: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]", "_", str(col))


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Preserve test ids if present; otherwise use 0-based row indices.
if "id" in test.columns:
    test_ids = (
        pd.to_numeric(test["id"], errors="coerce")
        .fillna(np.arange(len(test)))
        .astype(int)
        .to_numpy()
    )
else:
    test_ids = np.arange(len(test), dtype=int)
test = test.drop(columns=["id"], errors="ignore")

# Sanitize and de-duplicate column names.
train = train.rename(columns=_sanitize)
test = test.rename(columns=_sanitize)
train = train.loc[:, ~train.columns.duplicated()]
test = test.loc[:, ~test.columns.duplicated()]

# ---------------------------------------------------------------------------
# Target handling
# ---------------------------------------------------------------------------
if TARGET not in train.columns:
    target_candidates = [c for c in train.columns if c.lower() == TARGET.lower()]
    if len(target_candidates) == 1:
        train = train.rename(columns={target_candidates[0]: TARGET})
    else:
        raise ValueError(f"Target column '{TARGET}' not found in train data.")

y_all = pd.to_numeric(train[TARGET], errors="coerce").replace([np.inf, -np.inf], np.nan)
train = train.drop(columns=[TARGET], errors="ignore")

valid_target_mask = y_all.notna()
train = train.loc[valid_target_mask].copy()
y_all = y_all.loc[valid_target_mask].reset_index(drop=True)
train = train.reset_index(drop=True)

test = test.drop(columns=[TARGET], errors="ignore")

# ---------------------------------------------------------------------------
# Align feature columns between train and test
# ---------------------------------------------------------------------------
common_cols = [c for c in train.columns if c in test.columns]
if len(common_cols) == 0:
    common_cols = train.columns.tolist()

train = train[common_cols]
test = test.reindex(columns=common_cols)

n_train = len(train)

# Extremely small-data fallback: keep script executable.
if n_train == 0:
    submission = pd.DataFrame({"id": test_ids.astype(int), "number_of_officers": 0.0})
    submission.to_csv("./submission/submission.csv", index=False)
    print(f"Final Validation Score: {float('nan')}")
    raise SystemExit

# ---------------------------------------------------------------------------
# Validation split FIRST
# ---------------------------------------------------------------------------
if n_train >= 10:
    val_size = 0.2
elif n_train >= 2:
    val_size = 1
else:
    val_size = 0

stratify = None
if n_train >= 20 and y_all.nunique() > 1:
    try:
        stratify = pd.qcut(y_all, q=5, duplicates="drop")
    except Exception:
        stratify = None

try:
    if val_size > 0:
        train_idx, val_idx = train_test_split(
            np.arange(n_train), test_size=val_size, random_state=SEED, stratify=stratify
        )
    else:
        train_idx = np.arange(n_train)
        val_idx = np.array([], dtype=int)
except Exception:
    if val_size > 0:
        train_idx, val_idx = train_test_split(
            np.arange(n_train), test_size=val_size, random_state=SEED
        )
    else:
        train_idx = np.arange(n_train)
        val_idx = np.array([], dtype=int)

# Edge-case fallback: if no holdout is possible, use train as validation
# only to keep the pipeline executable. This should not occur in normal use.
if len(val_idx) == 0:
    val_idx = train_idx

X_tr_raw = train.iloc[train_idx].copy()
X_va_raw = train.iloc[val_idx].copy()
X_te_raw = test.copy()
X_full_raw = train.copy()

y_train = y_all.iloc[train_idx].astype(float).to_numpy()
y_val = y_all.iloc[val_idx].astype(float).to_numpy()

# ---------------------------------------------------------------------------
# Identify numeric / categorical columns using the training split only
# ---------------------------------------------------------------------------
numeric_cols = []
cat_cols = []

for col in X_tr_raw.columns:
    s = X_tr_raw[col]
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(col)
    else:
        cat_cols.append(col)

# Clean numeric columns.
for col in numeric_cols:
    X_tr_raw[col] = pd.to_numeric(X_tr_raw[col], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    X_va_raw[col] = pd.to_numeric(X_va_raw[col], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    X_te_raw[col] = pd.to_numeric(X_te_raw[col], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    X_full_raw[col] = pd.to_numeric(X_full_raw[col], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )

# Clean categorical columns.
for col in cat_cols:
    X_tr_raw[col] = X_tr_raw[col].astype(object).where(X_tr_raw[col].notna(), "missing")
    X_va_raw[col] = X_va_raw[col].astype(object).where(X_va_raw[col].notna(), "missing")
    X_te_raw[col] = X_te_raw[col].astype(object).where(X_te_raw[col].notna(), "missing")
    X_full_raw[col] = (
        X_full_raw[col].astype(object).where(X_full_raw[col].notna(), "missing")
    )

# Drop columns that are constant / all-NaN in the training split.
keep_cols = [c for c in X_tr_raw.columns if X_tr_raw[c].nunique(dropna=True) > 1]
if len(keep_cols) == 0:
    keep_cols = X_tr_raw.columns.tolist()

X_tr_raw = X_tr_raw[keep_cols]
X_va_raw = X_va_raw[keep_cols]
X_te_raw = X_te_raw[keep_cols]
X_full_raw = X_full_raw[keep_cols]

numeric_cols = [c for c in numeric_cols if c in keep_cols]
cat_cols = [c for c in cat_cols if c in keep_cols]

# ---------------------------------------------------------------------------
# Fit stateful transformers ONLY on the training split
# ---------------------------------------------------------------------------
if numeric_cols:
    num_imputer = SimpleImputer(strategy="median")
    X_tr_num_imp_fit = pd.DataFrame(
        num_imputer.fit_transform(X_tr_raw[numeric_cols]),
        index=X_tr_raw.index,
        columns=numeric_cols,
    )

    scaler = StandardScaler()
    scaler.fit(X_tr_num_imp_fit)

    variances = X_tr_num_imp_fit.var().sort_values(ascending=False)
    top_numeric_cols = variances.index[: min(6, len(variances))].tolist()
else:
    num_imputer = None
    scaler = None
    top_numeric_cols = []

cat_artifacts = {}
for col in cat_cols:
    counts = X_tr_raw[col].value_counts(normalize=True)
    dummy_cols = []
    if 1 < len(counts) <= 10:
        d_tr = pd.get_dummies(
            X_tr_raw[col], prefix=col, dummy_na=False, drop_first=True
        ).astype(float)
        dummy_cols = d_tr.columns.tolist()
    cat_artifacts[col] = {"counts": counts, "dummy_cols": dummy_cols}


# ---------------------------------------------------------------------------
# Feature builders using only fitted artifacts
# ---------------------------------------------------------------------------
def build_numeric_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame(index=df_raw.index)

    num_imp = pd.DataFrame(
        num_imputer.transform(df_raw[numeric_cols]),
        index=df_raw.index,
        columns=numeric_cols,
    )
    num_scaled = pd.DataFrame(
        scaler.transform(num_imp), index=df_raw.index, columns=numeric_cols
    )

    out = num_scaled.copy()

    for col in numeric_cols:
        x = num_imp[col].astype(float).to_numpy()
        out[f"{col}__slog"] = np.sign(x) * np.log1p(np.abs(x))
        if x.size > 0 and np.nanmin(x) >= 0:
            out[f"{col}__sqrt"] = np.sqrt(x)
        out[f"{col}__sq"] = x * x
        out[f"{col}__abs"] = np.abs(x)

    out["row__num_mean"] = num_imp.mean(axis=1).fillna(0.0).to_numpy()
    out["row__num_std"] = num_imp.std(axis=1).fillna(0.0).to_numpy()
    out["row__num_min"] = num_imp.min(axis=1).fillna(0.0).to_numpy()
    out["row__num_max"] = num_imp.max(axis=1).fillna(0.0).to_numpy()
    out["row__num_range"] = out["row__num_max"] - out["row__num_min"]
    out["row__num_sum"] = num_imp.sum(axis=1).fillna(0.0).to_numpy()
    out["row__num_skew"] = num_imp.skew(axis=1).fillna(0.0).to_numpy()

    if len(top_numeric_cols) >= 2:
        for i in range(len(top_numeric_cols)):
            for j in range(i + 1, len(top_numeric_cols)):
                a = num_imp[top_numeric_cols[i]].to_numpy(dtype=float)
                b = num_imp[top_numeric_cols[j]].to_numpy(dtype=float)
                out[f"{top_numeric_cols[i]}__prod__{top_numeric_cols[j]}"] = a * b
                out[f"{top_numeric_cols[i]}__ratio__{top_numeric_cols[j]}"] = a / (
                    np.abs(b) + 1e-6
                )
                out[f"{top_numeric_cols[i]}__diff__{top_numeric_cols[j]}"] = a - b

    return out


def build_categorical_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df_raw.index)

    for col in cat_cols:
        artifact = cat_artifacts[col]
        counts = artifact["counts"]
        dummy_cols = artifact["dummy_cols"]

        out[f"{col}__freq"] = df_raw[col].map(counts).fillna(0.0)

        if dummy_cols:
            d = pd.get_dummies(
                df_raw[col], prefix=col, dummy_na=False, drop_first=True
            ).astype(float)
            d = d.reindex(columns=dummy_cols, fill_value=0.0)
            out = pd.concat([out, d], axis=1)

    return out


X_tr_num = build_numeric_features(X_tr_raw)
X_tr_cat = build_categorical_features(X_tr_raw)
X_train = pd.concat([X_tr_num, X_tr_cat], axis=1)

X_va_num = build_numeric_features(X_va_raw)
X_va_cat = build_categorical_features(X_va_raw)
X_val = pd.concat([X_va_num, X_va_cat], axis=1)

X_te_num = build_numeric_features(X_te_raw)
X_te_cat = build_categorical_features(X_te_raw)
X_test = pd.concat([X_te_num, X_te_cat], axis=1)

X_full_num = build_numeric_features(X_full_raw)
X_full_cat = build_categorical_features(X_full_raw)
X_full = pd.concat([X_full_num, X_full_cat], axis=1)

# Align all matrices to the training feature schema.
X_train = X_train.loc[:, ~X_train.columns.duplicated()]
all_feature_cols = X_train.columns.tolist()

X_val = X_val.reindex(columns=all_feature_cols, fill_value=0.0)
X_test = X_test.reindex(columns=all_feature_cols, fill_value=0.0)
X_full = X_full.reindex(columns=all_feature_cols, fill_value=0.0)

if len(all_feature_cols) == 0:
    X_train["dummy"] = 0.0
    X_val["dummy"] = 0.0
    X_test["dummy"] = 0.0
    X_full["dummy"] = 0.0
    all_feature_cols = X_train.columns.tolist()

X_train = X_train.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)
X_val = X_val.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)
X_test = X_test.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)
X_full = X_full.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)


def _sanitize_feature_names(cols):
    seen = {}
    out = []
    for c in cols:
        base = re.sub(r"[^0-9a-zA-Z_]", "_", str(c)).strip("_")
        if not base:
            base = "feature"
        if base[0].isdigit():
            base = f"feature_{base}"
        base = base[:50]
        if base in seen:
            seen[base] += 1
            name = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
            name = base
        out.append(name)
    return out


_sanitized_feature_cols = _sanitize_feature_names(all_feature_cols)
X_train.columns = _sanitized_feature_cols
X_val.columns = _sanitized_feature_cols
X_test.columns = _sanitized_feature_cols
X_full.columns = _sanitized_feature_cols
all_feature_cols = _sanitized_feature_cols

# Optional audit artifacts.
X_train.to_parquet("./working/train_features.parquet")
X_val.to_parquet("./working/val_features.parquet")
X_test.to_parquet("./working/test_features.parquet")
X_full.to_parquet("./working/full_train_features.parquet")
pd.DataFrame({"id": np.arange(len(y_train)), "number_of_officers": y_train}).to_parquet(
    "./working/train_targets.parquet"
)
pd.DataFrame({"id": np.arange(len(y_val)), "number_of_officers": y_val}).to_parquet(
    "./working/val_targets.parquet"
)
pd.DataFrame({"id": test_ids.astype(int)}).to_parquet("./working/test_ids.parquet")

# ---------------------------------------------------------------------------
# Model design and training
# ---------------------------------------------------------------------------
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": -1,
}

base_n_estimators = 3000

# Edge-case fallback for extremely small training sets.
if len(X_train) < 2:
    val_pred = np.full(len(y_val), float(np.mean(y_train)))
    test_pred = np.full(len(X_test), float(np.mean(y_train)))
else:
    model = lgb.LGBMRegressor(n_estimators=base_n_estimators, **lgb_params)

    if len(y_val) > 0:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
    else:
        model.fit(X_train, y_train)

    best_iteration = getattr(model, "best_iteration_", None)
    if best_iteration is None or best_iteration <= 0:
        best_iteration = base_n_estimators

    val_pred = model.predict(X_val, num_iteration=best_iteration)

    # Retrain on the complete training dataset for final test inference.
    final_model = lgb.LGBMRegressor(n_estimators=int(best_iteration), **lgb_params)
    final_model.fit(X_full, y_all)
    test_pred = final_model.predict(X_test)

# ---------------------------------------------------------------------------
# Post-processing: identical for validation and test
# ---------------------------------------------------------------------------
integer_target = len(y_all) > 0 and np.allclose(y_all, np.round(y_all), atol=1e-6)


def postprocess(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

    # Officer counts are non-negative.
    pred = np.clip(pred, 0.0, None)

    if integer_target:
        pred = np.round(pred)

    return pred


val_pred = postprocess(val_pred)
test_pred = postprocess(test_pred)

# ---------------------------------------------------------------------------
# Official metric: RMSE on the local holdout
# ---------------------------------------------------------------------------
if len(y_val) > 0:
    score = float(np.sqrt(np.mean((y_val - val_pred) ** 2)))
else:
    score = float("nan")

submission = pd.DataFrame({"id": test_ids.astype(int), "number_of_officers": test_pred})
submission.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {score}")
