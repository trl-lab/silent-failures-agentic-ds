import os
import json
import copy
import random
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Global configuration
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
TARGET = "antibody_truth"
RANDOM_STATE = 42
SEED = RANDOM_STATE
VAL_FRACTION = 0.20

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# 1. Load raw data
# -----------------------------------------------------------------------------
train_raw = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
test_raw = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

# The task specifies id as the 0-based integer row index for submission.
test_id = np.arange(len(test_raw), dtype=int)

if TARGET not in train_raw.columns:
    raise ValueError(f"Target column {TARGET} not found in train data.")

train_raw = train_raw.copy()


def normalize_binary_target_series(s: pd.Series) -> pd.Series:
    if s is None or len(s) == 0:
        return s

    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)

    if pd.api.types.is_numeric_dtype(s):
        non_missing_mask = s.notna()
        if not non_missing_mask.any():
            return s.astype(float)

        uniq = s[non_missing_mask].unique()
        if len(uniq) == 1:
            out = pd.Series(np.nan, index=s.index, dtype=float)
            out[non_missing_mask] = int(uniq[0])
            return out

        if len(uniq) == 2:
            mapping = {v: i for i, v in enumerate(sorted(uniq))}
            return s.map(mapping).astype(float)

        median = float(np.nanmedian(s[non_missing_mask]))
        out = pd.Series(np.nan, index=s.index, dtype=float)
        out[non_missing_mask] = (s[non_missing_mask] > median).astype(int)
        return out

    converted = pd.to_numeric(s, errors="coerce")
    non_missing = s.notna()
    if non_missing.sum() > 0 and converted.notna().sum() >= 0.7 * non_missing.sum():
        return normalize_binary_target_series(converted)

    label_map = {
        "positive": 1, "pos": 1, "pos.": 1, "pos?": 1, "+": 1,
        "detected": 1, "reactive": 1, "yes": 1, "true": 1, "1": 1,
        "antibody positive": 1,
        "negative": 0, "neg": 0, "neg.": 0, "neg?": 0, "-": 0,
        "not detected": 0, "non-reactive": 0, "no": 0, "false": 0, "0": 0,
        "antibody negative": 0,
    }

    lowered = s.astype("string").str.strip().str.lower()
    mapped = lowered.map(label_map)
    if non_missing.sum() > 0 and mapped.notna().sum() >= 0.7 * non_missing.sum():
        return mapped.astype(float)

    uniq = s.dropna().unique()
    if len(uniq) == 2:
        mapping = {}
        for v in uniq:
            key = str(v).strip().lower()
            if key in label_map:
                mapping[v] = label_map[key]
        if len(mapping) == 2:
            return s.map(mapping).astype(float)
        mapping = {v: i for i, v in enumerate(sorted(uniq, key=str))}
        return s.map(mapping).astype(float)

    if len(uniq) > 2:
        freq = s.value_counts()
        top_two = freq.index[:2].tolist()
        mapping = {v: i for i, v in enumerate(top_two)}
        return s.map(mapping).astype(float)

    return s.astype(float)


train_raw[TARGET] = normalize_binary_target_series(train_raw[TARGET])
train_raw[TARGET] = pd.to_numeric(train_raw[TARGET], errors="coerce")
train_raw = train_raw.dropna(subset=[TARGET])
train_raw[TARGET] = train_raw[TARGET].astype(int)

# Drop exact duplicate training rows after target cleaning.
train_raw = train_raw.drop_duplicates().reset_index(drop=True)

# -----------------------------------------------------------------------------
# 2. Split first: train -> train/val, test remains untouched for inference.
# -----------------------------------------------------------------------------
n_total = len(train_raw)
if n_total == 0:
    raise ValueError("No training rows with non-missing target remain after cleaning.")

if n_total < 2:
    train_idx = np.array([0], dtype=int)
    val_idx = np.array([0], dtype=int)
elif n_total < 20:
    val_size = max(1, int(round(n_total * VAL_FRACTION)))
    val_size = min(val_size, n_total - 1)
    train_idx, val_idx = train_test_split(
        np.arange(n_total),
        test_size=val_size,
        random_state=RANDOM_STATE,
    )
elif train_raw[TARGET].nunique() > 1:
    train_idx, val_idx = train_test_split(
        np.arange(n_total),
        test_size=VAL_FRACTION,
        random_state=RANDOM_STATE,
        stratify=train_raw[TARGET],
    )
else:
    train_idx, val_idx = train_test_split(
        np.arange(n_total),
        test_size=VAL_FRACTION,
        random_state=RANDOM_STATE,
    )

train_split_raw = train_raw.iloc[train_idx].copy().reset_index(drop=True)
val_split_raw = train_raw.iloc[val_idx].copy().reset_index(drop=True)
test_split_raw = test_raw.copy().reset_index(drop=True)

y_train = train_split_raw[TARGET].to_numpy(dtype=int)
y_val = val_split_raw[TARGET].to_numpy(dtype=int)

id_cols = {"id"}
feature_cols = [c for c in train_split_raw.columns if c not in id_cols and c != TARGET]

train_feat_raw = train_split_raw[feature_cols].copy()
val_feat_raw = val_split_raw[feature_cols].copy()
test_feat_raw = test_split_raw[
    [c for c in feature_cols if c in test_split_raw.columns]
].copy()

# If test lacks train features, align feature spaces with NaN.
for c in feature_cols:
    if c not in test_feat_raw.columns:
        test_feat_raw[c] = np.nan

# -----------------------------------------------------------------------------
# 3. Column type detection and basic coercion
# -----------------------------------------------------------------------------
def try_convert_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s
    if s.dtype == object or str(s.dtype) == "string":
        converted = pd.to_numeric(s, errors="coerce")
        non_missing = s.notna().sum()
        if non_missing > 0 and converted.notna().sum() >= 0.7 * non_missing:
            return converted
    return s


def try_convert_datetime_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if s.dtype == object or str(s.dtype) == "string":
        converted = pd.to_datetime(s, errors="coerce")
        non_missing = s.notna().sum()
        if non_missing > 0 and converted.notna().sum() >= 0.7 * non_missing:
            return converted
    return s


datetime_cols = []
numeric_cols = []
categorical_cols = []

for col in feature_cols:
    train_s = train_feat_raw[col]
    val_s = val_feat_raw[col]
    test_s = test_feat_raw[col]

    dt_train = try_convert_datetime_series(train_s)
    if pd.api.types.is_datetime64_any_dtype(dt_train):
        datetime_cols.append(col)
        train_feat_raw[col] = dt_train
        val_feat_raw[col] = try_convert_datetime_series(val_s)
        test_feat_raw[col] = try_convert_datetime_series(test_s)
        continue

    num_train = try_convert_numeric_series(train_s)
    if pd.api.types.is_numeric_dtype(num_train):
        numeric_cols.append(col)
        train_feat_raw[col] = num_train
        val_feat_raw[col] = try_convert_numeric_series(val_s)
        test_feat_raw[col] = try_convert_numeric_series(test_s)
        continue

    categorical_cols.append(col)
    train_feat_raw[col] = train_s.astype("object").where(train_s.notna(), None)
    val_feat_raw[col] = val_s.astype("object").where(val_s.notna(), None)
    test_feat_raw[col] = test_s.astype("object").where(test_s.notna(), None)

# -----------------------------------------------------------------------------
# 4. Missingness indicators
# -----------------------------------------------------------------------------
for col in feature_cols:
    miss_col = f"{col}__missing"
    train_feat_raw[miss_col] = train_feat_raw[col].isna().astype(np.int8)
    val_feat_raw[miss_col] = val_feat_raw[col].isna().astype(np.int8)
    test_feat_raw[miss_col] = test_feat_raw[col].isna().astype(np.int8)

# -----------------------------------------------------------------------------
# 5. Datetime feature expansion
# -----------------------------------------------------------------------------
if datetime_cols:
    min_ts = None
    for col in datetime_cols:
        finite = train_feat_raw[col].dropna()
        if len(finite) > 0:
            candidate = finite.min()
            if min_ts is None or candidate < min_ts:
                min_ts = candidate
    if min_ts is None:
        min_ts = pd.Timestamp("1970-01-01")

    for col in datetime_cols:
        for df in (train_feat_raw, val_feat_raw, test_feat_raw):
            dt = df[col]
            df[f"{col}__year"] = dt.dt.year
            df[f"{col}__month"] = dt.dt.month
            df[f"{col}__day"] = dt.dt.day
            df[f"{col}__hour"] = dt.dt.hour
            df[f"{col}__minute"] = dt.dt.minute
            df[f"{col}__second"] = dt.dt.second
            df[f"{col}__dayofweek"] = dt.dt.dayofweek
            df[f"{col}__dayofyear"] = dt.dt.dayofyear
            df[f"{col}__weekofyear"] = dt.dt.isocalendar().week.astype("Int64")

            time_since = (dt - min_ts).dt.total_seconds().astype(float)
            df[f"{col}__time_since_min"] = time_since

            hour = dt.dt.hour.fillna(0).astype(float)
            dayofweek = dt.dt.dayofweek.fillna(0).astype(float)
            dayofyear = dt.dt.dayofyear.fillna(0).astype(float)

            df[f"{col}__hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
            df[f"{col}__hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
            df[f"{col}__dow_sin"] = np.sin(2 * np.pi * dayofweek / 7.0)
            df[f"{col}__dow_cos"] = np.cos(2 * np.pi * dayofweek / 7.0)
            df[f"{col}__doy_sin"] = np.sin(2 * np.pi * dayofyear / 365.25)
            df[f"{col}__doy_cos"] = np.cos(2 * np.pi * dayofyear / 365.25)

        train_feat_raw = train_feat_raw.drop(columns=[col])
        val_feat_raw = val_feat_raw.drop(columns=[col])
        test_feat_raw = test_feat_raw.drop(columns=[col])

# -----------------------------------------------------------------------------
# 6. Numeric cleaning: impute, winsorize, log-transform positives
# -----------------------------------------------------------------------------
numeric_bounds = {}
numeric_log_cols = []

for col in numeric_cols:
    s = train_feat_raw[col].replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() == 0:
        train_feat_raw[col] = 0.0
        continue

    median = float(s.median())
    q01 = float(s.quantile(0.01))
    q99 = float(s.quantile(0.99))

    if not np.isfinite(q01) or not np.isfinite(q99):
        q01, q99 = median, median
    if q99 <= q01:
        q01, q99 = median, median

    numeric_bounds[col] = (median, q01, q99)

    if s.min() >= 0 and s.std() > 0:
        numeric_log_cols.append(col)


def clean_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = np.nan
        s = df[col].replace([np.inf, -np.inf], np.nan)
        median, q01, q99 = numeric_bounds.get(col, (0.0, 0.0, 0.0))
        s = s.fillna(median)
        s = s.clip(lower=q01, upper=q99)
        df[col] = s.astype(float)
    return df


train_feat_raw = clean_numeric_frame(train_feat_raw)
val_feat_raw = clean_numeric_frame(val_feat_raw)
test_feat_raw = clean_numeric_frame(test_feat_raw)

for col in numeric_log_cols:
    train_feat_raw[f"{col}__log1p"] = np.log1p(train_feat_raw[col].clip(lower=0.0))
    val_feat_raw[f"{col}__log1p"] = np.log1p(val_feat_raw[col].clip(lower=0.0))
    test_feat_raw[f"{col}__log1p"] = np.log1p(test_feat_raw[col].clip(lower=0.0))

# -----------------------------------------------------------------------------
# 7. Categorical encoding: frequency + ordinal, fit on train only
# -----------------------------------------------------------------------------
for col in categorical_cols:
    train_s = train_feat_raw[col].fillna("__missing__").astype(str)
    freq = train_s.value_counts(normalize=True)
    categories = sorted(train_s.dropna().unique().tolist())

    train_feat_raw[f"{col}__freq"] = train_s.map(freq).fillna(0.0).astype(float)
    val_feat_raw[f"{col}__freq"] = (
        val_feat_raw[col]
        .fillna("__missing__")
        .astype(str)
        .map(freq)
        .fillna(0.0)
        .astype(float)
    )
    test_feat_raw[f"{col}__freq"] = (
        test_feat_raw[col]
        .fillna("__missing__")
        .astype(str)
        .map(freq)
        .fillna(0.0)
        .astype(float)
    )

    code_map = {c: i for i, c in enumerate(categories)}
    train_feat_raw[f"{col}__code"] = train_s.map(code_map).fillna(-1).astype(int)
    val_feat_raw[f"{col}__code"] = (
        val_feat_raw[col]
        .fillna("__missing__")
        .astype(str)
        .map(code_map)
        .fillna(-1)
        .astype(int)
    )
    test_feat_raw[f"{col}__code"] = (
        test_feat_raw[col]
        .fillna("__missing__")
        .astype(str)
        .map(code_map)
        .fillna(-1)
        .astype(int)
    )

    train_feat_raw = train_feat_raw.drop(columns=[col])
    val_feat_raw = val_feat_raw.drop(columns=[col])
    test_feat_raw = test_feat_raw.drop(columns=[col])

# -----------------------------------------------------------------------------
# 8. Group aggregates for device/sample/batch-like columns
# -----------------------------------------------------------------------------
group_col_candidates = []
for col in categorical_cols:
    name = col.lower()
    if any(
        token in name
        for token in [
            "id",
            "device",
            "sample",
            "subject",
            "patient",
            "well",
            "plate",
            "batch",
            "run",
            "lot",
            "serial",
            "unit",
            "sensor",
            "probe",
            "channel",
            "row",
            "column",
            "date",
            "time",
        ]
    ):
        group_col_candidates.append(col)

for col in categorical_cols:
    code_col = f"{col}__code"
    if code_col in train_feat_raw.columns:
        nunique = train_feat_raw[code_col].nunique(dropna=True)
        if 2 <= nunique <= max(20, int(np.sqrt(len(train_feat_raw)) * 2)):
            if col not in group_col_candidates:
                group_col_candidates.append(col)

group_cols = group_col_candidates[:4]

base_numeric_for_groups = [c for c in numeric_cols if c in train_feat_raw.columns]
if len(base_numeric_for_groups) > 0:
    variances = {
        c: (
            float(train_feat_raw[c].var(ddof=0))
            if train_feat_raw[c].notna().any()
            else 0.0
        )
        for c in base_numeric_for_groups
    }
    top_group_numerics = sorted(variances, key=variances.get, reverse=True)[:5]
else:
    top_group_numerics = []

if group_cols and top_group_numerics:
    train_key = train_split_raw[
        [c for c in group_cols if c in train_split_raw.columns]
    ].copy()
    val_key = val_split_raw[
        [c for c in group_cols if c in val_split_raw.columns]
    ].copy()
    test_key = test_split_raw[
        [c for c in group_cols if c in test_split_raw.columns]
    ].copy()

    for col in group_cols:
        if col not in train_key.columns:
            continue
        train_key[col] = train_key[col].fillna("__missing__").astype(str)
        val_key[col] = val_key[col].fillna("__missing__").astype(str)
        test_key[col] = test_key[col].fillna("__missing__").astype(str)

        for num in top_group_numerics:
            if num not in train_feat_raw.columns:
                continue
            temp = pd.DataFrame(
                {
                    "group": train_key[col].to_numpy(),
                    "value": train_feat_raw[num].to_numpy(dtype=float),
                }
            )

            agg = temp.groupby("group")["value"].agg(
                ["mean", "std", "min", "max", "count"]
            )
            global_stats = {
                "mean": float(train_feat_raw[num].mean()),
                "std": float(train_feat_raw[num].std(ddof=0)),
                "min": float(train_feat_raw[num].min()),
                "max": float(train_feat_raw[num].max()),
                "count": float(len(train_feat_raw)),
            }

            for split_key, split_feat in [
                (train_key, train_feat_raw),
                (val_key, val_feat_raw),
                (test_key, test_feat_raw),
            ]:
                keys = split_key[col].to_numpy()
                for stat in ["mean", "std", "min", "max", "count"]:
                    feat_name = f"grp_{col}__{num}__{stat}"
                    mapped = pd.Series(keys).map(agg[stat])
                    mapped = mapped.fillna(global_stats[stat])
                    split_feat[feat_name] = mapped.to_numpy(dtype=float)

# -----------------------------------------------------------------------------
# 9. Low-order interactions among strongest numeric signals
# -----------------------------------------------------------------------------
interaction_base = [c for c in numeric_cols if c in train_feat_raw.columns]
if len(interaction_base) > 0:
    variances = {
        c: (
            float(train_feat_raw[c].var(ddof=0))
            if train_feat_raw[c].notna().any()
            else 0.0
        )
        for c in interaction_base
    }
    top_interactions = sorted(variances, key=variances.get, reverse=True)[:5]
else:
    top_interactions = []

for i in range(len(top_interactions)):
    for j in range(i + 1, len(top_interactions)):
        a = top_interactions[i]
        b = top_interactions[j]
        for df in (train_feat_raw, val_feat_raw, test_feat_raw):
            da = df[a].astype(float)
            db = df[b].astype(float)
            df[f"inter__{a}__{b}__prod"] = da * db
            df[f"inter__{a}__{b}__ratio"] = da / (db + 1e-6)
            df[f"inter__{a}__{b}__diff"] = da - db

# -----------------------------------------------------------------------------
# 10. Final numeric matrix, scaling, and feature selection
#     Imputer is fit on train only to avoid split-specific imputation leakage.
# -----------------------------------------------------------------------------
train_feat = train_feat_raw.select_dtypes(include=[np.number]).copy()
train_feat = train_feat.replace([np.inf, -np.inf], np.nan)

final_imputer = SimpleImputer(strategy="median")
train_feat = pd.DataFrame(
    final_imputer.fit_transform(train_feat),
    columns=train_feat.columns,
    index=train_feat.index,
)

def finalize_with_train_imputer(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.reindex(columns=train_feat.columns).copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(
        final_imputer.transform(df),
        columns=train_feat.columns,
        index=df.index,
    )

val_feat = finalize_with_train_imputer(val_feat_raw)
test_feat = finalize_with_train_imputer(test_feat_raw)

# Drop constant features using train only.
train_var = train_feat.var(ddof=0)
non_constant = train_var[train_var > 1e-12].index.tolist()
train_feat = train_feat[non_constant]
val_feat = val_feat[non_constant]
test_feat = test_feat[non_constant]

# Correlation deduplication using train only.
if len(non_constant) > 1:
    corr = train_feat.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        highly_corr = upper.index[upper[col] > 0.995].tolist()
        if highly_corr:
            to_drop.add(col)
            to_drop.update(highly_corr)
    keep = [c for c in non_constant if c not in to_drop]
    if len(keep) > 0:
        train_feat = train_feat[keep]
        val_feat = val_feat[keep]
        test_feat = test_feat[keep]

# If still very large, keep top-variance features to keep downstream model stable.
if train_feat.shape[1] > 220:
    top_features = (
        train_feat.var(ddof=0).sort_values(ascending=False).head(220).index.tolist()
    )
    train_feat = train_feat[top_features]
    val_feat = val_feat[top_features]
    test_feat = test_feat[top_features]

# Fit scaler on train only.
scaler = StandardScaler()
train_feat_scaled = pd.DataFrame(
    scaler.fit_transform(train_feat),
    columns=train_feat.columns,
    index=train_feat.index,
)
val_feat_scaled = pd.DataFrame(
    scaler.transform(val_feat),
    columns=train_feat.columns,
    index=val_feat.index,
)
test_feat_scaled = pd.DataFrame(
    scaler.transform(test_feat),
    columns=train_feat.columns,
    index=test_feat.index,
)

train_feat_scaled = train_feat_scaled.replace([np.inf, -np.inf], 0.0).fillna(0.0)
val_feat_scaled = val_feat_scaled.replace([np.inf, -np.inf], 0.0).fillna(0.0)
test_feat_scaled = test_feat_scaled.replace([np.inf, -np.inf], 0.0).fillna(0.0)

# -----------------------------------------------------------------------------
# 11. Save engineered artifacts for auditability
# -----------------------------------------------------------------------------
train_out = train_feat_scaled.copy()
train_out[TARGET] = y_train

val_out = val_feat_scaled.copy()
val_out[TARGET] = y_val

test_out = test_feat_scaled.copy()
test_out["id"] = test_id

train_out.to_parquet(os.path.join(WORKING_DIR, "processed_train.parquet"), index=False)
val_out.to_parquet(os.path.join(WORKING_DIR, "processed_val.parquet"), index=False)
test_out.to_parquet(os.path.join(WORKING_DIR, "processed_test.parquet"), index=False)

metadata = {
    "target": TARGET,
    "random_state": RANDOM_STATE,
    "val_fraction": VAL_FRACTION,
    "train_size": int(len(train_out)),
    "val_size": int(len(val_out)),
    "test_size": int(len(test_out)),
    "n_features": int(train_feat_scaled.shape[1]),
    "feature_names": train_feat_scaled.columns.tolist(),
    "original_numeric_cols": numeric_cols,
    "original_categorical_cols": categorical_cols,
    "datetime_cols": datetime_cols,
    "group_cols": group_cols,
    "id_col": "id",
    "split": "stratified_by_target_if_binary",
    "metric": "roc_auc",
}
with open(os.path.join(WORKING_DIR, "feature_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(
    f"Feature engineering complete: train={len(train_out)}, "
    f"val={len(val_out)}, test={len(test_out)}, features={train_feat_scaled.shape[1]}"
)

# -----------------------------------------------------------------------------
# 12. Model design setup: context-gated tabular MLP
# -----------------------------------------------------------------------------
train_df = train_out
val_df = val_out
test_df = test_out

feature_names = train_feat_scaled.columns.tolist()

for c in feature_names:
    if c not in train_df.columns:
        train_df[c] = 0.0
    if c not in val_df.columns:
        val_df[c] = 0.0
    if c not in test_df.columns:
        test_df[c] = 0.0

X_train = train_df[feature_names].astype(float).to_numpy()
X_val = val_df[feature_names].astype(float).to_numpy()
X_test = test_df[feature_names].astype(float).to_numpy()

y_train = train_df[TARGET].astype(float).to_numpy()
y_val = val_df[TARGET].astype(float).to_numpy()

X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

n_train, n_features = X_train.shape
n_val = X_val.shape[0]
n_test = X_test.shape[0]

context_tokens = (
    "grp_",
    "__freq",
    "__code",
    "__missing",
    "__year",
    "__month",
    "__day",
    "__hour",
    "__minute",
    "__second",
    "__dayofweek",
    "__dayofyear",
    "__weekofyear",
    "__time_since",
    "__dow",
    "__doy",
)

context_idx = [
    i
    for i, name in enumerate(feature_names)
    if any(token in str(name) for token in context_tokens)
]

if len(context_idx) == 0:
    context_idx = [
        i for i, name in enumerate(feature_names) if "__missing" in str(name)
    ]

context_dim = len(context_idx)
context_indices = (
    torch.tensor(context_idx, dtype=torch.long, device=device)
    if context_dim > 0
    else torch.empty(0, dtype=torch.long, device=device)
)


class ContextGatedTabularNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        context_indices: torch.Tensor,
        hidden_sizes: tuple[int, int],
        context_hidden: int,
        dropout: float,
        prior_bias: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.context_indices = context_indices
        self.context_dim = int(context_indices.numel())
        self.context_out_dim = (
            max(8, context_hidden // 2) if self.context_dim > 0 else 0
        )

        h1, h2 = hidden_sizes

        if self.context_dim > 0:
            self.context_encoder = nn.Sequential(
                nn.Linear(self.context_dim, context_hidden),
                nn.LayerNorm(context_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(context_hidden, context_hidden),
                nn.LayerNorm(context_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(context_hidden, self.context_out_dim),
            )

            self.gate_net = nn.Sequential(
                nn.Linear(self.context_dim, context_hidden),
                nn.LayerNorm(context_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(context_hidden, self.input_dim),
            )
            nn.init.zeros_(self.gate_net[-1].bias)
        else:
            self.context_encoder = None
            self.gate_net = None

        self.sensor_net = nn.Sequential(
            nn.Linear(self.input_dim, h1),
            nn.LayerNorm(h1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        head_in = h2 + self.context_out_dim
        head_mid = max(16, h2 // 2)

        self.head = nn.Sequential(
            nn.Linear(head_in, head_mid),
            nn.LayerNorm(head_mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_mid, 1),
        )

        if np.isfinite(prior_bias):
            with torch.no_grad():
                self.head[-1].bias.fill_(float(prior_bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()

        if self.context_dim > 0:
            ctx = x[:, self.context_indices]
            gate = 1.0 + torch.tanh(self.gate_net(ctx))
            sensor = x * gate
            context_repr = self.context_encoder(ctx)
        else:
            sensor = x
            context_repr = torch.zeros(
                x.size(0), self.context_out_dim, device=x.device, dtype=x.dtype
            )

        sensor_repr = self.sensor_net(sensor)
        combined = torch.cat([sensor_repr, context_repr], dim=1)
        logits = self.head(combined).squeeze(-1)
        return logits


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight: float):
        super().__init__()
        self.register_buffer(
            "pos_weight", torch.tensor(float(pos_weight), dtype=torch.float32)
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
        )


pos_count = float(np.sum(y_train == 1))
neg_count = float(np.sum(y_train == 0))

if pos_count > 0 and neg_count > 0:
    pos_weight = neg_count / pos_count
    prior = pos_count / (pos_count + neg_count)
    prior_bias = float(np.log(prior / (1.0 - prior))) if 0.0 < prior < 1.0 else 0.0
else:
    pos_weight = 1.0
    prior_bias = 0.0

pos_weight = float(np.clip(pos_weight, 0.2, 5.0))

if n_features <= 20:
    hidden_sizes = (32, 16)
    context_hidden = 16
    dropout = 0.30
    weight_decay = 0.020
    lr = 5e-4
elif n_features <= 80:
    hidden_sizes = (64, 32)
    context_hidden = 32
    dropout = 0.25
    weight_decay = 0.015
    lr = 8e-4
else:
    hidden_sizes = (128, 64)
    context_hidden = 64
    dropout = 0.20
    weight_decay = 0.010
    lr = 1e-3

if n_train < 500:
    dropout = min(0.40, dropout + 0.05)
    weight_decay = min(0.05, weight_decay + 0.010)
    lr = min(lr, 5e-4)

max_epochs = 400 if n_train < 2000 else 250
patience = 30
batch_size = min(128, max(16, n_train // 2))

model = ContextGatedTabularNet(
    input_dim=n_features,
    context_indices=context_indices,
    hidden_sizes=hidden_sizes,
    context_hidden=context_hidden,
    dropout=dropout,
    prior_bias=prior_bias,
).to(device)

decay_params = []
no_decay_params = []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if param.dim() > 1:
        decay_params.append(param)
    else:
        no_decay_params.append(param)

optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=lr,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max(max_epochs, 1),
    eta_min=lr * 1e-2,
)

criterion = WeightedBCEWithLogitsLoss(pos_weight=pos_weight)

train_ds = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32),
)
val_ds = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.float32),
)
test_ds = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
)

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False
)
val_loader = DataLoader(
    val_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False
)
test_loader = DataLoader(
    test_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False
)

# -----------------------------------------------------------------------------
# 13. Training, evaluation, and submission
# -----------------------------------------------------------------------------
swa_start = max(1, int(max_epochs * 0.5))
swa_eval_every = 5
swa_tolerance = 0.02


@torch.no_grad()
def predict_probabilities(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader
) -> np.ndarray:
    model.eval()
    preds = []
    for batch in loader:
        xb = batch[0].to(device, non_blocking=True)
        logits = model(xb)
        probs = torch.sigmoid(logits).float().cpu().numpy().astype(float)
        preds.append(probs)
    if len(preds) == 0:
        return np.zeros(0, dtype=float)
    return np.concatenate(preds, axis=0)


def official_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)

    if len(y_true) == 0 or len(y_prob) == 0:
        return 0.5
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_prob))


def current_lr() -> float:
    try:
        if optimizer is not None and len(optimizer.param_groups) > 0:
            return float(optimizer.param_groups[0]["lr"])
    except Exception:
        pass
    return 0.0


initial_val_pred = predict_probabilities(model, val_loader)
best_val_auc = official_roc_auc(y_val, initial_val_pred)
best_state = copy.deepcopy(model.state_dict())
best_epoch = 0

swa_state = None
swa_n = 0
best_swa_state = None
best_swa_auc = -1.0

history = []
patience_counter = 0

for epoch in range(1, max_epochs + 1):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    scheduler.step()

    # Validation
    val_pred = predict_probabilities(model, val_loader)
    val_auc = official_roc_auc(y_val, val_pred)
    history.append({"epoch": epoch, "loss": total_loss / max(n_batches, 1), "val_auc": val_auc})

    # Track best
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch
        patience_counter = 0
    else:
        patience_counter += 1

    # SWA accumulation
    if epoch >= swa_start and val_auc >= best_val_auc - swa_tolerance:
        if swa_state is None:
            swa_state = copy.deepcopy(model.state_dict())
            swa_n = 1
        else:
            swa_n += 1
            for k in swa_state:
                swa_state[k] = (swa_state[k] * (swa_n - 1) + model.state_dict()[k]) / swa_n

    # Early stopping
    if patience_counter >= patience:
        break

# Evaluate SWA candidate vs best
if swa_state is not None and swa_n >= 2:
    model.load_state_dict(swa_state)
    swa_val_pred = predict_probabilities(model, val_loader)
    swa_val_auc = official_roc_auc(y_val, swa_val_pred)
    if swa_val_auc > best_val_auc:
        best_state = copy.deepcopy(model.state_dict())
        best_val_auc = swa_val_auc

# Load best model for test predictions
model.load_state_dict(best_state)
test_pred = predict_probabilities(model, test_loader)

# Write submission
submission = pd.DataFrame({
    "id": test_id,
    "antibody_truth": test_pred.astype(float),
})
submission.to_csv(os.path.join(SUBMISSION_DIR, "submission.csv"), index=False)

print(f"Final Validation Score: {best_val_auc:.6f}")