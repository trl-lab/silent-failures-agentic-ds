import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

INPUT_DIR = "./input"
WORK_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

TARGET = "outcome"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# 1. Load raw data
# -----------------------------------------------------------------------------
train_raw_all = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
test_raw = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

if TARGET not in train_raw_all.columns:
    raise ValueError(f"Target column '{TARGET}' not found in train data.")


def normalize_target(s):
    if s.dtype == object:
        mapping = {
            "fail": 0,
            "failure": 0,
            "failed": 0,
            "crash": 0,
            "crashes": 0,
            "succeed": 1,
            "success": 1,
            "succeeded": 1,
            "ok": 1,
            "pass": 1,
        }
        s = s.astype(str).str.lower().str.strip().map(mapping)
    return pd.to_numeric(s, errors="coerce")


train_raw_all[TARGET] = normalize_target(train_raw_all[TARGET])
train_raw_all = train_raw_all.dropna(subset=[TARGET]).reset_index(drop=True)

# Drop obvious non-parameter metadata columns.
meta_like = {
    "id",
    "index",
    "row_id",
    "sample_id",
    "batch",
    "batch_id",
    "sample",
    "label",
    "target",
}
drop_cols = [c for c in train_raw_all.columns if c.lower() in meta_like and c != TARGET]
train_raw_all = train_raw_all.drop(columns=drop_cols, errors="ignore")

# Identify candidate feature columns.
feature_candidates = [c for c in train_raw_all.columns if c != TARGET]

numeric_feature_candidates = []
for c in feature_candidates:
    tmp = pd.to_numeric(train_raw_all[c], errors="coerce")
    if tmp.notna().sum() > 0:
        numeric_feature_candidates.append(c)

# The climate dataset may include two deactivated batch/sample columns.
# If more than 18 numeric parameter columns exist, keep the last 18, matching
# the documented 18 active model parameters.
if len(numeric_feature_candidates) > 18:
    feature_cols = numeric_feature_candidates[-18:]
else:
    feature_cols = numeric_feature_candidates

# Ensure test has the same feature columns; if not, align as much as possible.
test_feature_cols = [c for c in feature_cols if c in test_raw.columns]
if len(test_feature_cols) != len(feature_cols):
    feature_cols = test_feature_cols

if len(feature_cols) == 0:
    raise ValueError("No numeric feature columns found.")

# -----------------------------------------------------------------------------
# 2. Split FIRST: stratified train/validation split from raw training data
# -----------------------------------------------------------------------------
y_all = train_raw_all[TARGET]
if y_all.nunique() < 2:
    raise ValueError("Target is not binary; cannot perform stratified split.")

train_df, val_df = train_test_split(
    train_raw_all, test_size=0.20, stratify=y_all, random_state=SEED, shuffle=True
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_raw.reset_index(drop=True)

X_train_raw = train_df[feature_cols].copy()
X_val_raw = val_df[feature_cols].copy()
X_test_raw = test_df[[c for c in feature_cols if c in test_df.columns]].copy()

y_train = train_df[TARGET].astype(int).values
y_val = val_df[TARGET].astype(int).values


# -----------------------------------------------------------------------------
# 3. Cleaning / imputation fitted ONLY on training split
# -----------------------------------------------------------------------------
def clean_numeric(df, cols):
    out = pd.DataFrame(index=df.index)
    for c in cols:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            out[c] = np.nan
    return out


X_train_clean = clean_numeric(X_train_raw, feature_cols)
X_val_clean = clean_numeric(X_val_raw, feature_cols)
X_test_clean = clean_numeric(X_test_raw, feature_cols)

train_median = X_train_clean.median(numeric_only=True)


def impute_with_train_stats(df):
    out = df.copy()
    for c in feature_cols:
        if c in out.columns:
            fill = train_median[c] if pd.notna(train_median[c]) else 0.0
            out[c] = out[c].fillna(fill)
    return out


X_train_clean = impute_with_train_stats(X_train_clean)
X_val_clean = impute_with_train_stats(X_val_clean)
X_test_clean = impute_with_train_stats(X_test_clean)

# Decide clipping from the training split only, then apply identically to all splits.
clip_cols = []
for c in feature_cols:
    s = X_train_clean[c]
    if s.notna().any():
        in_range = ((s >= -1e-6) & (s <= 1 + 1e-6)).mean()
        if in_range > 0.95:
            clip_cols.append(c)


def apply_train_clip(df):
    out = df.copy()
    for c in clip_cols:
        if c in out.columns:
            out[c] = out[c].clip(0.0, 1.0)
    return out


X_train_clean = apply_train_clip(X_train_clean)
X_val_clean = apply_train_clip(X_val_clean)
X_test_clean = apply_train_clip(X_test_clean)

train_mean = X_train_clean.mean(numeric_only=True).fillna(0.0)
train_std = X_train_clean.std(numeric_only=True).replace(0.0, 1.0).fillna(1.0)


# -----------------------------------------------------------------------------
# 4. Feature engineering (all statistics from training split only)
# -----------------------------------------------------------------------------
def univariate_auc(x, y):
    if np.unique(y).size < 2:
        return 0.5
    try:
        auc = roc_auc_score(y, x)
    except ValueError:
        return 0.5
    return max(auc, 1.0 - auc)


univ_scores = {}
for c in feature_cols:
    univ_scores[c] = univariate_auc(X_train_clean[c].values, y_train)

top_k = min(8, len(feature_cols))
top_features = sorted(univ_scores, key=lambda c: univ_scores[c], reverse=True)[:top_k]


def engineer_features(X, top_feats):
    out = pd.DataFrame(index=X.index)

    # Base features
    for c in feature_cols:
        out[c] = X[c].values

    # Standardized features (train statistics)
    for c in feature_cols:
        out[f"z_{c}"] = (X[c].values - train_mean[c]) / train_std[c]

    # Squared terms for all base features
    for c in feature_cols:
        out[f"sq_{c}"] = X[c].values ** 2

    # Pairwise products among top features (interaction hypothesis)
    for i in range(len(top_feats)):
        for j in range(i + 1, len(top_feats)):
            a, b = top_feats[i], top_feats[j]
            out[f"inter_{a}_{b}"] = X[a].values * X[b].values

    # Extreme-value indicators (parameters are scaled [0,1])
    for c in feature_cols:
        out[f"hi_{c}"] = (X[c].values > 0.8).astype(np.float32)
        out[f"lo_{c}"] = (X[c].values < 0.2).astype(np.float32)

    # Aggregate extreme counts
    hi_cols = [f"hi_{c}" for c in feature_cols]
    lo_cols = [f"lo_{c}" for c in feature_cols]
    out["n_high"] = out[hi_cols].sum(axis=1)
    out["n_low"] = out[lo_cols].sum(axis=1)
    out["n_extreme"] = out["n_high"] + out["n_low"]

    # Pairwise extreme conjunctions among top features
    for i in range(len(top_feats)):
        for j in range(i + 1, len(top_feats)):
            a, b = top_feats[i], top_feats[j]
            out[f"hihi_{a}_{b}"] = out[f"hi_{a}"] * out[f"hi_{b}"]
            out[f"lolo_{a}_{b}"] = out[f"lo_{a}"] * out[f"lo_{b}"]
            out[f"hilo_{a}_{b}"] = out[f"hi_{a}"] * out[f"lo_{b}"]

    # Aggregate parameter-space descriptors
    vals = X[feature_cols].values
    out["mean_param"] = vals.mean(axis=1)
    out["std_param"] = vals.std(axis=1)
    out["min_param"] = vals.min(axis=1)
    out["max_param"] = vals.max(axis=1)
    out["range_param"] = out["max_param"] - out["min_param"]
    out["l1_norm"] = np.abs(vals).sum(axis=1)
    out["l2_norm"] = np.sqrt((vals**2).sum(axis=1))
    out["median_param"] = np.median(vals, axis=1)
    out["iqr_param"] = np.percentile(vals, 75, axis=1) - np.percentile(vals, 25, axis=1)

    # Distances to train centroid (multivariate outlier / risk indicator)
    centroid = train_mean[feature_cols].values
    out["dist_mean_euclidean"] = np.linalg.norm(vals - centroid, axis=1)
    out["dist_mean_manhattan"] = np.abs(vals - centroid).sum(axis=1)
    out["dist_mean_chebyshev"] = np.abs(vals - centroid).max(axis=1)

    # Mahalanobis distance using train covariance (pseudo-inverse for stability)
    try:
        cov = np.cov(X_train_clean[feature_cols].values, rowvar=False)
        cov = np.atleast_2d(cov)
        cov_inv = np.linalg.pinv(cov)
        diff = vals - centroid
        mahalanobis2 = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
        out["dist_mahalanobis"] = np.sqrt(np.maximum(mahalanobis2, 0.0))
    except Exception:
        out["dist_mahalanobis"] = out["dist_mean_euclidean"]

    return out


X_train_fe = engineer_features(X_train_clean, top_features)
X_val_fe = engineer_features(X_val_clean, top_features)
X_test_fe = engineer_features(X_test_clean, top_features)

# Replace any residual non-finite values with 0 (safe for tree/NN models).
X_train_fe = X_train_fe.replace([np.inf, -np.inf], np.nan).fillna(0.0)
X_val_fe = X_val_fe.replace([np.inf, -np.inf], np.nan).fillna(0.0)
X_test_fe = X_test_fe.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# -----------------------------------------------------------------------------
# 5. Persist engineered artifacts for later steps
# -----------------------------------------------------------------------------
def make_id_frame(df):
    if "id" in df.columns:
        return df[["id"]].copy()
    return pd.DataFrame({"__row_id": np.arange(len(df))}, index=df.index)


train_out = make_id_frame(train_df)
train_out[TARGET] = y_train
train_out = pd.concat([train_out, X_train_fe], axis=1)

val_out = make_id_frame(val_df)
val_out[TARGET] = y_val
val_out = pd.concat([val_out, X_val_fe], axis=1)

test_out = make_id_frame(test_df)
test_out = pd.concat([test_out, X_test_fe], axis=1)

train_out.to_parquet(os.path.join(WORK_DIR, "train_engineered.parquet"), index=False)
val_out.to_parquet(os.path.join(WORK_DIR, "val_engineered.parquet"), index=False)
test_out.to_parquet(os.path.join(WORK_DIR, "test_engineered.parquet"), index=False)

feature_meta = {
    "target": TARGET,
    "base_features": feature_cols,
    "top_interaction_features": top_features,
    "engineered_features": list(X_train_fe.columns),
    "split": {
        "train_rows": int(len(train_out)),
        "val_rows": int(len(val_out)),
        "test_rows": int(len(test_out)),
        "train_pos_rate": float(y_train.mean()),
        "val_pos_rate": float(y_val.mean()),
    },
}
with open(os.path.join(WORK_DIR, "feature_meta.json"), "w") as f:
    json.dump(feature_meta, f, indent=2)

# -----------------------------------------------------------------------------
# 6. Model design and training: failure-risk MLP with AUC surrogate
# -----------------------------------------------------------------------------
train_eng = pd.read_parquet(os.path.join(WORK_DIR, "train_engineered.parquet"))
val_eng = pd.read_parquet(os.path.join(WORK_DIR, "val_engineered.parquet"))
test_eng = pd.read_parquet(os.path.join(WORK_DIR, "test_engineered.parquet"))

feature_cols = None
meta_path = os.path.join(WORK_DIR, "feature_meta.json")
if os.path.exists(meta_path):
    with open(meta_path, "r") as f:
        meta = json.load(f)
    feature_cols = meta.get("engineered_features", None)

exclude = {TARGET, "id", "index", "row_id", "sample_id", "label", "__row_id"}
if feature_cols is None or len(feature_cols) == 0:
    feature_cols = [c for c in train_eng.columns if c not in exclude]

feature_cols = [c for c in feature_cols if c in train_eng.columns]
if len(feature_cols) == 0:
    feature_cols = [c for c in train_eng.columns if c not in exclude]
if len(feature_cols) == 0:
    raise ValueError("No engineered features found.")


def prepare_X(df):
    X = df[feature_cols].astype(np.float32).values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def prepare_y(df):
    if TARGET not in df.columns:
        return np.zeros(len(df), dtype=int)
    y = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int).values
    return np.clip(y, 0, 1)


X_train = prepare_X(train_eng)
X_val = prepare_X(val_eng)
X_test = prepare_X(test_eng)

y_success_train = prepare_y(train_eng)
y_success_val = prepare_y(val_eng)

# The target is 0=failure, 1=success.
# We model P(failure) and submit P(success) = 1 - P(failure).
y_fail_train = 1 - y_success_train

# Fit input normalization only on the training split.
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train).astype(np.float32)
X_val_s = scaler.transform(X_val).astype(np.float32)
X_test_s = scaler.transform(X_test).astype(np.float32)

X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0)
X_val_s = np.nan_to_num(X_val_s, nan=0.0, posinf=0.0, neginf=0.0)
X_test_s = np.nan_to_num(X_test_s, nan=0.0, posinf=0.0, neginf=0.0)

X_train_t = torch.tensor(X_train_s, dtype=torch.float32, device=DEVICE)
X_val_t = torch.tensor(X_val_s, dtype=torch.float32, device=DEVICE)
X_test_t = torch.tensor(X_test_s, dtype=torch.float32, device=DEVICE)
y_fail_train_t = torch.tensor(y_fail_train, dtype=torch.float32, device=DEVICE)


# LightGBM binary failure-risk ensemble.
# Regularized trees are better suited to this small imbalanced tabular set and
# can be stopped directly on validation ROC AUC.
y_fail_val = 1 - y_success_val
n_fail = int(y_fail_train.sum())
n_success = int((y_success_train == 1).sum())
scale_pos_weight = float(np.clip(n_success / max(n_fail, 1), 1.0, 10.0))

val_fail_ranks = []
test_fail_ranks = []

if n_fail > 0 and n_success > 0 and len(np.unique(y_fail_val)) > 1:
    lgbm_configs = [
        {
            "seed": SEED,
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 20,
            "learning_rate": 0.02,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "reg_lambda": 10.0,
        },
        {
            "seed": SEED + 1,
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 15,
            "learning_rate": 0.01,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_lambda": 5.0,
        },
        {
            "seed": SEED + 2,
            "num_leaves": 7,
            "max_depth": 4,
            "min_child_samples": 20,
            "learning_rate": 0.01,
            "subsample": 0.85,
            "colsample_bytree": 0.7,
            "reg_lambda": 10.0,
        },
        {
            "seed": SEED + 3,
            "num_leaves": 15,
            "max_depth": 3,
            "min_child_samples": 15,
            "learning_rate": 0.02,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 5.0,
        },
    ]

    for cfg in lgbm_configs:
        model = lgb.LGBMClassifier(
            objective="binary",
            eval_metric="auc",
            n_estimators=3000,
            scale_pos_weight=scale_pos_weight,
            num_leaves=cfg["num_leaves"],
            max_depth=cfg["max_depth"],
            min_child_samples=cfg["min_child_samples"],
            learning_rate=cfg["learning_rate"],
            subsample=cfg["subsample"],
            subsample_freq=1,
            colsample_bytree=cfg["colsample_bytree"],
            reg_lambda=cfg["reg_lambda"],
            random_state=cfg["seed"],
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            X_train_s,
            y_fail_train,
            eval_set=[(X_val_s, y_fail_val)],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        best_iter = getattr(model, "best_iteration_", None)
        if best_iter is not None and best_iter > 0:
            val_fail_pred = model.predict_proba(X_val_s, num_iteration=best_iter)[:, 1]
            test_fail_pred = model.predict_proba(X_test_s, num_iteration=best_iter)[:, 1]
        else:
            val_fail_pred = model.predict_proba(X_val_s)[:, 1]
            test_fail_pred = model.predict_proba(X_test_s)[:, 1]

        val_fail_pred = np.nan_to_num(val_fail_pred, nan=0.5, posinf=1.0, neginf=0.0)
        test_fail_pred = np.nan_to_num(test_fail_pred, nan=0.5, posinf=1.0, neginf=0.0)

        val_fail_ranks.append(pd.Series(val_fail_pred).rank(pct=True).values)
        test_fail_ranks.append(pd.Series(test_fail_pred).rank(pct=True).values)

        try:
            model_val_auc = float(roc_auc_score(y_fail_val, val_fail_pred))
        except ValueError:
            model_val_auc = 0.5

        print(
            f"LightGBM seed={cfg['seed']} | "
            f"best_iteration={best_iter} | val_fail_auc={model_val_auc:.4f}"
        )

if len(val_fail_ranks) == 0:
    val_success_pred = np.full(len(y_success_val), 0.5, dtype=np.float64)
    test_success_pred = np.full(len(X_test_s), 0.5, dtype=np.float64)
else:
    val_fail_rank_avg = np.mean(val_fail_ranks, axis=0)
    test_fail_rank_avg = np.mean(test_fail_ranks, axis=0)
    val_success_pred = np.clip(1.0 - val_fail_rank_avg, 0.0, 1.0)
    test_success_pred = np.clip(1.0 - test_fail_rank_avg, 0.0, 1.0)

# -----------------------------------------------------------------------------
# 7. Final inference and submission
# -----------------------------------------------------------------------------
try:
    score = float(roc_auc_score(y_success_val, val_success_pred))
except ValueError:
    score = 0.5

if not np.isfinite(score):
    score = 0.5

submission = pd.DataFrame(
    {
        "id": np.arange(len(test_success_pred), dtype=int),
        "outcome": test_success_pred.astype(np.float64),
    }
)
submission.to_csv(os.path.join(SUBMISSION_DIR, "submission.csv"), index=False)

print(f"Final Validation Score: {score}")
