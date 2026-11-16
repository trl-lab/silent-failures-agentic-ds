import os
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb

warnings.filterwarnings("ignore")

WORK_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

TARGET = "credit_amount"
ID_COL = "id"
SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------
train_raw = pd.read_parquet("./input/train.parquet")
test_raw = pd.read_parquet("./input/test.parquet")

if TARGET not in train_raw.columns:
    raise ValueError(f"Target column '{TARGET}' not found in train.parquet")

train_raw[TARGET] = pd.to_numeric(train_raw[TARGET], errors="coerce")
train_raw = train_raw[np.isfinite(train_raw[TARGET])].reset_index(drop=True)

if len(train_raw) == 0:
    raise ValueError("No usable training rows after removing non-finite target values.")

# Required submission id is the 0-based integer row index.
test_raw[ID_COL] = np.arange(len(test_raw), dtype=int)

# ---------------------------------------------------------------------------
# 2. Split FIRST: raw train -> train split + validation split.
#    All feature statistics below are fit only on the train split.
# ---------------------------------------------------------------------------
y_all = train_raw[TARGET].to_numpy(dtype=float)

stratify = None
if len(np.unique(y_all)) > 5:
    try:
        stratify = pd.qcut(y_all, q=5, duplicates="drop")
    except Exception:
        stratify = None

used_stratify = False
if len(train_raw) < 2:
    train_idx = np.arange(len(train_raw))
    val_idx = train_idx
else:
    try:
        train_idx, val_idx = train_test_split(
            np.arange(len(train_raw)),
            test_size=0.2,
            random_state=SEED,
            stratify=stratify,
        )
        used_stratify = stratify is not None
    except Exception:
        train_idx, val_idx = train_test_split(
            np.arange(len(train_raw)),
            test_size=0.2,
            random_state=SEED,
            stratify=None,
        )
        used_stratify = False

train_split_raw = train_raw.iloc[train_idx].copy().reset_index(drop=True)
val_split_raw = train_raw.iloc[val_idx].copy().reset_index(drop=True)

y_train = train_split_raw[TARGET].to_numpy(dtype=float)
y_val = val_split_raw[TARGET].to_numpy(dtype=float)

# ---------------------------------------------------------------------------
# 3. Select safe feature columns: columns present in both train and test.
# ---------------------------------------------------------------------------
train_feature_cols = [c for c in train_split_raw.columns if c not in {TARGET, ID_COL}]
test_feature_cols = [c for c in test_raw.columns if c not in {TARGET, ID_COL}]

feature_cols = [c for c in train_feature_cols if c in test_feature_cols]

# Fallback: if there is no overlap, use train columns and fill missing test columns as NaN.
if not feature_cols:
    feature_cols = train_feature_cols


def project_frame(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        if c in df.columns:
            out[c] = df[c].to_numpy()
        else:
            out[c] = np.nan
    return out


Xtr_raw = project_frame(train_split_raw, feature_cols)
Xva_raw = project_frame(val_split_raw, feature_cols)
Xte_raw = project_frame(test_raw, feature_cols)

# ---------------------------------------------------------------------------
# 4. Infer feature types using only the training split.
# ---------------------------------------------------------------------------
numeric_cols = []
categorical_cols = []

for c in feature_cols:
    s = Xtr_raw[c]

    if s.dtype == bool:
        numeric_cols.append(c)
        continue

    if pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(c)
        continue

    coerced = pd.to_numeric(s, errors="coerce")
    if coerced.notna().sum() > 0 and coerced.notna().mean() >= 0.5:
        numeric_cols.append(c)
    else:
        categorical_cols.append(c)

# Convert string-like numeric columns consistently.
for c in numeric_cols:
    Xtr_raw[c] = pd.to_numeric(Xtr_raw[c], errors="coerce")
    Xva_raw[c] = pd.to_numeric(Xva_raw[c], errors="coerce")
    Xte_raw[c] = pd.to_numeric(Xte_raw[c], errors="coerce")

# Drop columns that are entirely missing in the training split.
usable_numeric = [c for c in numeric_cols if Xtr_raw[c].notna().any()]
usable_categorical = [c for c in categorical_cols if Xtr_raw[c].notna().any()]

# ---------------------------------------------------------------------------
# 5. Missingness indicators (fit from train split only).
# ---------------------------------------------------------------------------
for c in usable_numeric + usable_categorical:
    if Xtr_raw[c].isna().any():
        Xtr_raw[f"{c}__missing"] = Xtr_raw[c].isna().astype(np.int8)
        Xva_raw[f"{c}__missing"] = Xva_raw[c].isna().astype(np.int8)
        Xte_raw[f"{c}__missing"] = Xte_raw[c].isna().astype(np.int8)

# ---------------------------------------------------------------------------
# 6. Numeric imputation and transforms.
# ---------------------------------------------------------------------------
for c in usable_numeric:
    med = float(Xtr_raw[c].median())
    if not np.isfinite(med):
        med = 0.0

    Xtr_raw[c] = Xtr_raw[c].fillna(med)
    Xva_raw[c] = Xva_raw[c].fillna(med)
    Xte_raw[c] = Xte_raw[c].fillna(med)

    # Signed log transform: monotonic, stable for zero/negative/skewed values.
    Xtr_raw[f"{c}__slog"] = np.sign(Xtr_raw[c]) * np.log1p(np.abs(Xtr_raw[c]))
    Xva_raw[f"{c}__slog"] = np.sign(Xva_raw[c]) * np.log1p(np.abs(Xva_raw[c]))
    Xte_raw[f"{c}__slog"] = np.sign(Xte_raw[c]) * np.log1p(np.abs(Xte_raw[c]))

# ---------------------------------------------------------------------------
# 7. Categorical encoding and frequency encoding (fit from train split only).
# ---------------------------------------------------------------------------
if usable_categorical:
    cat_train_str = Xtr_raw[usable_categorical].astype(str)
    cat_val_str = Xva_raw[usable_categorical].astype(str)
    cat_test_str = Xte_raw[usable_categorical].astype(str)

    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cat_train = oe.fit_transform(cat_train_str)
    cat_val = oe.transform(cat_val_str)
    cat_test = oe.transform(cat_test_str)

    for i, c in enumerate(usable_categorical):
        Xtr_raw[f"{c}__code"] = cat_train[:, i].astype(np.int32)
        Xva_raw[f"{c}__code"] = cat_val[:, i].astype(np.int32)
        Xte_raw[f"{c}__code"] = cat_test[:, i].astype(np.int32)

        freq = cat_train_str[c].value_counts()
        Xtr_raw[f"{c}__freq"] = cat_train_str[c].map(freq).fillna(0.0).astype(float)
        Xva_raw[f"{c}__freq"] = cat_val_str[c].map(freq).fillna(0.0).astype(float)
        Xte_raw[f"{c}__freq"] = cat_test_str[c].map(freq).fillna(0.0).astype(float)

# ---------------------------------------------------------------------------
# 8. Interaction features from top train-correlated numeric columns.
#    Correlation is computed only on the training split, so no validation/test leakage.
# ---------------------------------------------------------------------------
y_std = float(np.std(y_train))
corr_map = {}
for c in usable_numeric:
    x = Xtr_raw[c].to_numpy(dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)

    if np.std(x) > 0 and y_std > 0:
        corr = float(np.corrcoef(x, y_train)[0, 1])
        if not np.isfinite(corr):
            corr = 0.0
        corr_map[c] = corr
    else:
        corr_map[c] = 0.0

top_numeric = sorted(corr_map, key=lambda k: abs(corr_map[k]), reverse=True)[:4]

eps = 1e-8
for i in range(len(top_numeric)):
    for j in range(i + 1, len(top_numeric)):
        a = top_numeric[i]
        b = top_numeric[j]

        for df in (Xtr_raw, Xva_raw, Xte_raw):
            xa = df[a].to_numpy(dtype=float)
            xb = df[b].to_numpy(dtype=float)
            xa = np.where(np.isfinite(xa), xa, 0.0)
            xb = np.where(np.isfinite(xb), xb, 0.0)

            df[f"{a}__x__{b}"] = xa * xb
            df[f"{a}__div__{b}"] = xa / (np.abs(xb) + eps)

# ---------------------------------------------------------------------------
# 9. Final numeric cleanup, scaling, and zero-variance filtering.
# ---------------------------------------------------------------------------
for df in (Xtr_raw, Xva_raw, Xte_raw):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill any remaining NaNs with train-split medians.
for c in Xtr_raw.columns:
    if pd.api.types.is_numeric_dtype(Xtr_raw[c]):
        med = float(Xtr_raw[c].median())
        if not np.isfinite(med):
            med = 0.0

        Xtr_raw[c] = Xtr_raw[c].fillna(med)
        Xva_raw[c] = Xva_raw[c].fillna(med)
        Xte_raw[c] = Xte_raw[c].fillna(med)

# Standardize numeric features using train-split statistics only.
scale_cols = [c for c in Xtr_raw.columns if pd.api.types.is_numeric_dtype(Xtr_raw[c])]
if scale_cols:
    scaler = StandardScaler()
    Xtr_raw[scale_cols] = scaler.fit_transform(Xtr_raw[scale_cols])
    Xva_raw[scale_cols] = scaler.transform(Xva_raw[scale_cols])
    Xte_raw[scale_cols] = scaler.transform(Xte_raw[scale_cols])

# Remove zero-variance columns in the training split.
final_feature_cols = [
    c
    for c in Xtr_raw.columns
    if pd.api.types.is_numeric_dtype(Xtr_raw[c])
    and np.std(Xtr_raw[c].to_numpy(dtype=float)) > 1e-12
]

if not final_feature_cols:
    # Defensive fallback: keep one safe row-position feature rather than failing.
    final_feature_cols = ["__row_index"]
    Xtr_raw[final_feature_cols[0]] = np.arange(len(Xtr_raw), dtype=float)
    Xva_raw[final_feature_cols[0]] = np.arange(len(Xva_raw), dtype=float)
    Xte_raw[final_feature_cols[0]] = np.arange(len(Xte_raw), dtype=float)

X_train = Xtr_raw[final_feature_cols].astype(float)
X_val = Xva_raw[final_feature_cols].astype(float)
X_test = Xte_raw[final_feature_cols].astype(float)

# ---------------------------------------------------------------------------
# 10. Persist processed artifacts for later pipeline steps.
# ---------------------------------------------------------------------------
X_train.to_parquet(os.path.join(WORK_DIR, "processed_train.parquet"), index=False)
X_val.to_parquet(os.path.join(WORK_DIR, "processed_val.parquet"), index=False)
X_test.to_parquet(os.path.join(WORK_DIR, "processed_test.parquet"), index=False)

pd.DataFrame({"id": train_split_raw.index, "credit_amount": y_train}).to_parquet(
    os.path.join(WORK_DIR, "train_target.parquet"), index=False
)
pd.DataFrame({"id": val_split_raw.index, "credit_amount": y_val}).to_parquet(
    os.path.join(WORK_DIR, "val_target.parquet"), index=False
)

with open(os.path.join(WORK_DIR, "feature_names.json"), "w") as f:
    json.dump(final_feature_cols, f, indent=2)

# ---------------------------------------------------------------------------
# 11. Consistent post-processing and metric helpers.
# ---------------------------------------------------------------------------
clip_non_negative = bool(np.min(y_train) >= 0.0)


def _apply_postprocess(pred):
    pred = np.asarray(pred, dtype=float)
    if clip_non_negative:
        pred = np.clip(pred, 0.0, np.inf)
    return pred


def _clipped_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if clip_non_negative:
        y_pred = np.clip(y_pred, 0.0, np.inf)

    err = y_true - y_pred
    return "clipped_rmse", float(np.sqrt(np.mean(err * err))), False


def _predict_with_best_iteration(model, X, best_iter):
    if best_iter is None or best_iter < 0:
        return model.predict(X)
    return model.predict(X, num_iteration=max(int(best_iter), 1))


# ---------------------------------------------------------------------------
# 12. Minimal sanity model for this step:
#     - validates that the engineered features are usable;
#     - produces a task-faithful official RMSE on the holdout validation split;
#     - writes a consistent test submission.
#     Later steps can replace this sanity model with the full competition model.
# ---------------------------------------------------------------------------
sanity_model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=15,
    max_depth=-1,
    min_child_samples=5,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)

sanity_model.fit(X_train, y_train)

sanity_val_pred = _apply_postprocess(sanity_model.predict(X_val))
sanity_test_pred = _apply_postprocess(sanity_model.predict(X_test))
sanity_score = float(root_mean_squared_error(y_val, sanity_val_pred))

submission = pd.DataFrame(
    {
        "id": np.arange(len(sanity_test_pred), dtype=int),
        "credit_amount": sanity_test_pred.astype(float),
    }
)
submission.to_csv(os.path.join(SUBMISSION_DIR, "submission.csv"), index=False)

print(f"[sanity] val_rmse={sanity_score:.4f}")

# ---------------------------------------------------------------------------
# 13. Recover engineered splits / targets from the previous step.
#     Primary source: in-memory variables; fallback: saved parquet artifacts.
# ---------------------------------------------------------------------------
if all(v in globals() for v in ("X_train", "y_train", "X_val", "y_val", "X_test")):
    print("[train] using in-memory artifacts from previous step")
else:
    print("[train] in-memory artifacts missing -> loading saved parquet files")
    X_train = pd.read_parquet(os.path.join(WORK_DIR, "processed_train.parquet"))
    X_val = pd.read_parquet(os.path.join(WORK_DIR, "processed_val.parquet"))
    X_test = pd.read_parquet(os.path.join(WORK_DIR, "processed_test.parquet"))
    y_train = pd.read_parquet(os.path.join(WORK_DIR, "train_target.parquet"))[TARGET]
    y_val = pd.read_parquet(os.path.join(WORK_DIR, "val_target.parquet"))[TARGET]

X_train = np.asarray(X_train, dtype=float)
X_val = np.asarray(X_val, dtype=float)
X_test = np.asarray(X_test, dtype=float)
y_train = np.asarray(y_train, dtype=float)
y_val = np.asarray(y_val, dtype=float)

assert len(X_train) == len(y_train), "train feature/target length mismatch"
assert len(X_val) == len(y_val), "val feature/target length mismatch"
assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], "feature width mismatch"

if len(train_raw) < 2:
    split_desc = "all train (too small to split)"
elif used_stratify:
    split_desc = "80/20 stratified-quintile"
else:
    split_desc = "80/20 random"

print(
    f"[audit] split={split_desc} seed=42 | metric=RMSE | "
    f"rows: train={len(y_train)} val={len(y_val)} test={len(X_test)} feats={X_train.shape[1]}"
)

# ---------------------------------------------------------------------------
# 14. Candidate models: same regularized LightGBM skeleton, two losses.
#     - regression (L2): direct RMSE-aligned objective
#     - huber: robust to heavy-tailed credit outliers (delta = MAD scale of y)
#     Both are early-stopped on the official RMSE over the holdout split.
# ---------------------------------------------------------------------------
mad = float(np.median(np.abs(y_train - np.median(y_train))))
huber_delta = mad if mad > 1e-6 else 1.0

base_params = dict(
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=10,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)

candidates = {
    "lgb_l2": dict(base_params, objective="regression"),
    "lgb_huber": dict(base_params, objective="huber", huber_delta=huber_delta),
}

results = []
for name, params in candidates.items():
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=_clipped_rmse,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    n_iter = getattr(model, "best_iteration_", None)
    pred = _predict_with_best_iteration(model, X_val, n_iter)
    pred = _apply_postprocess(pred)

    rmse = float(root_mean_squared_error(y_val, pred))
    results.append((name, model, rmse, n_iter))
    print(f"[train] {name}: best_iter={n_iter} val_rmse={rmse:.4f}")

# ---------------------------------------------------------------------------
# 15. Select best candidate by official RMSE; persist booster + metadata.
# ---------------------------------------------------------------------------
best_name, best_model, best_rmse, best_iter = min(results, key=lambda t: t[2])

best_model.booster_.save_model(os.path.join(WORK_DIR, "best_model.txt"))

best_iter_meta = None if best_iter is None or best_iter < 0 else max(int(best_iter), 1)
with open(os.path.join(WORK_DIR, "best_model_meta.json"), "w") as f:
    json.dump(
        {
            "candidate": best_name,
            "best_iteration": best_iter_meta,
            "val_rmse": float(best_rmse),
        },
        f,
        indent=2,
    )

print(
    f"[select] best candidate={best_name} best_iter={best_iter_meta} val_rmse={best_rmse:.4f}"
)

# ---------------------------------------------------------------------------
# 16. Final inference: IDENTICAL path for validation and test.
#     Same booster, same num_iteration, same support clip (credits >= 0).
# ---------------------------------------------------------------------------
val_pred = _predict_with_best_iteration(best_model, X_val, best_iter)
test_pred = _predict_with_best_iteration(best_model, X_test, best_iter)

val_pred = _apply_postprocess(val_pred)
test_pred = _apply_postprocess(test_pred)

score = float(root_mean_squared_error(y_val, val_pred))

submission = pd.DataFrame(
    {
        "id": np.arange(len(test_pred), dtype=int),
        "credit_amount": test_pred.astype(float),
    }
)
submission.to_csv(os.path.join(SUBMISSION_DIR, "submission.csv"), index=False)

print(
    f"[output] submission.csv written: rows={len(submission)} cols={list(submission.columns)}"
)

print(f"Final Validation Score: {score}")
