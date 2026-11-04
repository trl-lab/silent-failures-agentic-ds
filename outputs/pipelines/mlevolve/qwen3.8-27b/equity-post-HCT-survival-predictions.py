import os
import re
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# =============================================================================
# 0. Paths, constants, and reproducibility
# =============================================================================
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

TRAIN_PATH = "./input/train.parquet"
TEST_PATH = "./input/test.parquet"
SUBMISSION_PATH = "./submission/submission.csv"

TARGET = "efs"
TIME_COL = "efs_time"
ID_CANDIDATES = ["ID", "id"]
RANDOM_STATE = 42
TEST_SIZE = 0.2

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# =============================================================================
# 1. Load raw data and create validation split FIRST
# =============================================================================
train = pd.read_parquet(TRAIN_PATH)
test = pd.read_parquet(TEST_PATH)

# Identify id column if present, but do not use it as a feature.
id_col = None
for c in ID_CANDIDATES:
    if c in train.columns:
        id_col = c
        break

# Build a stratification key from target + time quartile, using train only.
train["_time_quartile"] = pd.qcut(
    train[TIME_COL], q=4, labels=False, duplicates="drop"
).astype(int)
train["_strat_key"] = train[TARGET].astype(int) * 10 + train["_time_quartile"]

train_idx, val_idx = train_test_split(
    np.arange(len(train)),
    test_size=TEST_SIZE,
    stratify=train["_strat_key"],
    random_state=RANDOM_STATE,
)

# Drop target/time and split helper columns before feature engineering.
drop_cols = [
    c for c in [TARGET, TIME_COL, "_time_quartile", "_strat_key"] if c in train.columns
]
train_raw = train.drop(columns=drop_cols)
test_raw = test.copy()

train_split = train_raw.iloc[train_idx].reset_index(drop=True)
val_split = train_raw.iloc[val_idx].reset_index(drop=True)
test_split = test_raw.reset_index(drop=True)

# Keep target/time aligned with split indices.
y_train_full = train[TARGET].astype(float).values
time_train_full = train[TIME_COL].astype(float).values
y_train = y_train_full[train_idx]
time_train = time_train_full[train_idx]
y_val = y_train_full[val_idx]
time_val = time_train_full[val_idx]


# =============================================================================
# 2. Feature engineering utilities
# =============================================================================
def clean_col_name(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_")


def ensure_columns(df: pd.DataFrame, cols: list) -> list:
    return [c for c in cols if c in df.columns]


YES_NO_COLS = [
    "arrhythmia",
    "cardiac",
    "diabetes",
    "hepatic_mild",
    "hepatic_severe",
    "in_vivo_tcd",
    "obesity",
    "peptic_ulcer",
    "prior_tumor",
    "psych_disturb",
    "pulm_moderate",
    "pulm_severe",
    "renal_issue",
    "rheum_issue",
    "rituximab",
    "vent_hist",
]


def yes_no_flags(df: pd.DataFrame, cols: list) -> None:
    for c in ensure_columns(df, cols):
        s = df[c].astype("string").str.strip()
        df[f"{c}_yes"] = (s == "Yes").fillna(False).astype(int)
        df[f"{c}_no"] = (s == "No").fillna(False).astype(int)
        df[f"{c}_not_done"] = (s == "Not done").fillna(False).astype(int)
        df[f"{c}_missing"] = s.isna().astype(int)


def disease_group(s) -> str:
    if pd.isna(s):
        return "missing"
    u = str(s).upper()
    if any(x in u for x in ["ALL", "AML", "MDS", "CML", "CLL", "IPA", "LEUKEMIA"]):
        return "leukemia"
    if "MM" in u or "MYELOMA" in u:
        return "myeloma"
    if "LYMPHOMA" in u or "NHL" in u or "HODGKIN" in u or "HL" in u:
        return "lymphoma"
    return "other"


HLA_LOCI = [
    "hla_match_a_high",
    "hla_match_a_low",
    "hla_match_b_high",
    "hla_match_b_low",
    "hla_match_c_high",
    "hla_match_c_low",
    "hla_match_dqb1_high",
    "hla_match_dqb1_low",
    "hla_match_drb1_high",
    "hla_match_drb1_low",
]


def hla_mismatch_sum(df: pd.DataFrame) -> None:
    cols = ensure_columns(df, HLA_LOCI)
    mismatch = pd.Series(0.0, index=df.index)
    available = pd.Series(0.0, index=df.index)
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce")
        mismatch = mismatch + (2.0 - v).fillna(0.0)
        available = available + v.notna().astype(float)
    df["hla_locus_mismatch_sum"] = mismatch
    df["hla_locus_available"] = available
    df["hla_locus_missing"] = len(cols) - available


HLA_SCORE_COLS = [
    ("hla_high_res_10", 10),
    ("hla_high_res_6", 6),
    ("hla_high_res_8", 8),
    ("hla_low_res_10", 10),
    ("hla_low_res_6", 6),
    ("hla_low_res_8", 8),
    ("hla_nmdp_6", 6),
]


def hla_score_mismatches(df: pd.DataFrame) -> None:
    for c, maxv in HLA_SCORE_COLS:
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        df[f"{c}_mismatch"] = (maxv - v).fillna(0.0)
        df[f"{c}_missing"] = v.isna().astype(int)


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Yes/No/Not done flags.
    yes_no_flags(df, YES_NO_COLS)

    # Disease grouping.
    if "prim_disease_hct" in df.columns:
        df["prim_disease_group"] = df["prim_disease_hct"].apply(disease_group)

    # HLA features.
    hla_mismatch_sum(df)
    hla_score_mismatches(df)

    # Age features.
    age = (
        pd.to_numeric(df.get("age_at_hct"), errors="coerce")
        if "age_at_hct" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    df["age_log"] = np.log1p(age.fillna(0.0))
    df["age_sq"] = (age.fillna(0.0) ** 2) / 1000.0
    df["age_pediatric"] = (age < 18).astype(int)
    df["age_elderly"] = (age >= 65).astype(int)
    df["age_missing"] = age.isna().astype(int)

    # Donor age features.
    donor = (
        pd.to_numeric(df.get("donor_age"), errors="coerce")
        if "donor_age" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    df["donor_age_missing"] = donor.isna().astype(int)
    df["donor_age_gt_50"] = (donor > 50).astype(int)
    age_diff = donor - age
    df["age_diff_donor_patient"] = age_diff.fillna(0.0)
    df["age_diff_abs"] = age_diff.abs().fillna(0.0)

    # Comorbidity and frailty features.
    comp = (
        pd.to_numeric(df.get("comorbidity_score"), errors="coerce")
        if "comorbidity_score" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    df["comorbidity_score_missing"] = comp.isna().astype(int)
    df["comorbidity_high"] = (comp >= 4).astype(int)

    karn = (
        pd.to_numeric(df.get("karnofsky_score"), errors="coerce")
        if "karnofsky_score" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    df["karnofsky_missing"] = karn.isna().astype(int)
    df["frailty_100_karnofsky"] = (100.0 - karn).fillna(0.0)
    df["karnofsky_low"] = (karn < 80).astype(int)

    # GVHD prophylaxis features.
    if "gvhd_proph" in df.columns:
        s = df["gvhd_proph"].astype("string").str.strip()
        df["gvhd_fk"] = s.str.contains("FK", case=False, na=False).astype(int)
        df["gvhd_mmf"] = s.str.contains("MMF", case=False, na=False).astype(int)
        df["gvhd_mtx"] = s.str.contains("MTX", case=False, na=False).astype(int)
        df["gvhd_cyclo"] = s.str.contains(
            "Cyclophosphamide", case=False, na=False
        ).astype(int)
        df["gvhd_proph_count"] = df[
            ["gvhd_fk", "gvhd_mmf", "gvhd_mtx", "gvhd_cyclo"]
        ].sum(axis=1)

    # TBI features.
    if "tbi_status" in df.columns:
        s = df["tbi_status"].astype("string").str.strip()
        df["tbi_flag"] = (
            s.str.contains("TBI", case=False, na=False)
            & ~s.str.contains("No TBI", case=False, na=False)
        ).astype(int)
        df["tbi_low_dose"] = s.str.contains("<=cGy", case=False, na=False).astype(int)
        df["tbi_high_dose"] = s.str.contains(">cGy", case=False, na=False).astype(int)
        df["tbi_fractionated"] = s.str.contains(
            "fractionated", case=False, na=False
        ).astype(int)
        df["tbi_single"] = s.str.contains("single", case=False, na=False).astype(int)
        df["tbi_unknown_dose"] = s.str.contains("unknown", case=False, na=False).astype(
            int
        )

    # TCE match features.
    if "tce_match" in df.columns:
        s = df["tce_match"].astype("string").str.strip()
        df["tce_fully_matched"] = (s == "Fully matched").fillna(False).astype(int)
        df["tce_permissive"] = (s == "Permissive").fillna(False).astype(int)
        df["tce_nonpermissive"] = s.str.contains(
            "non-permissive", case=False, na=False
        ).astype(int)

    # Year era features.
    if "year_hct" in df.columns:
        year = pd.to_numeric(df["year_hct"], errors="coerce")
        df["year_hct_num"] = year.fillna(2008.0)
        df["year_era_pre2010"] = (year < 2010).astype(int)
        df["year_era_2010_2015"] = ((year >= 2010) & (year <= 2015)).astype(int)
        df["year_era_post2015"] = (year > 2015).astype(int)
        df["year_missing"] = year.isna().astype(int)

    return df


# =============================================================================
# 3. Apply feature engineering to train/val/test with train-only fitted states
# =============================================================================
train_eng = add_engineered(train_split)
val_eng = add_engineered(val_split)
test_eng = add_engineered(test_split)

exclude_cols = set(ID_CANDIDATES) | {TARGET, TIME_COL, "_time_quartile", "_strat_key"}

numeric_cols = [
    c
    for c in train_eng.columns
    if c not in exclude_cols and pd.api.types.is_numeric_dtype(train_eng[c])
]

onehot_cols = [
    "arrhythmia",
    "cardiac",
    "cmv_status",
    "conditioning_intensity",
    "cyto_score",
    "cyto_score_detail",
    "diabetes",
    "donor_related",
    "dri_score",
    "ethnicity",
    "graft_type",
    "gvhd_proph",
    "hepatic_mild",
    "hepatic_severe",
    "in_vivo_tcd",
    "melphalan_dose",
    "mrd_hct",
    "obesity",
    "peptic_ulcer",
    "prim_disease_hct",
    "prim_disease_group",
    "prior_tumor",
    "prod_type",
    "psych_disturb",
    "pulm_moderate",
    "pulm_severe",
    "race_group",
    "renal_issue",
    "rheum_issue",
    "rituximab",
    "sex_match",
    "tbi_status",
    "tce_div_match",
    "tce_imm_match",
    "tce_match",
    "vent_hist",
]
cat_cols = [c for c in onehot_cols if c in train_eng.columns]

# Numeric imputation using train split medians only.
numeric_medians = {}
for c in numeric_cols:
    med = train_eng[c].median()
    if pd.isna(med):
        med = 0.0
    numeric_medians[c] = float(med)
    train_eng[c] = train_eng[c].fillna(med)
    if c in val_eng.columns:
        val_eng[c] = val_eng[c].fillna(med)
    if c in test_eng.columns:
        test_eng[c] = test_eng[c].fillna(med)

# Defensive finite replacement.
for frame in (train_eng, val_eng, test_eng):
    for c in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[c]):
            frame[c] = frame[c].replace([np.inf, -np.inf], np.nan)

# One-hot encoding using train split categories only.
onehot_categories = {}
for c in cat_cols:
    vals_train = train_eng[c].astype("string").str.strip()
    vals_train = vals_train.where(vals_train.notna(), "missing")
    cats = sorted(vals_train.unique().tolist())
    onehot_categories[c] = cats

    for cat in cats:
        name = clean_col_name(f"{c}_{cat}")
        train_eng[name] = (vals_train == cat).astype(int)

        vals_val = val_eng[c].astype("string").str.strip()
        vals_val = vals_val.where(vals_val.notna(), "missing")
        val_eng[name] = (vals_val == cat).astype(int)

        vals_test = test_eng[c].astype("string").str.strip()
        vals_test = vals_test.where(vals_test.notna(), "missing")
        test_eng[name] = (vals_test == cat).astype(int)

# Drop original categorical columns after one-hotting.
for c in cat_cols:
    train_eng = train_eng.drop(columns=[c], errors="ignore")
    val_eng = val_eng.drop(columns=[c], errors="ignore")
    test_eng = test_eng.drop(columns=[c], errors="ignore")

# Final feature matrix columns.
final_numeric_cols = [
    c
    for c in train_eng.columns
    if c not in exclude_cols and pd.api.types.is_numeric_dtype(train_eng[c])
]
feature_names = final_numeric_cols

# Ensure val/test have exactly the same feature columns.
for frame in (val_eng, test_eng):
    missing = [c for c in feature_names if c not in frame.columns]
    if missing:
        frame[missing] = 0.0

# Scale using train split only.
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_eng[feature_names].astype(float))
val_scaled = scaler.transform(val_eng[feature_names].astype(float))
test_scaled = scaler.transform(test_eng[feature_names].astype(float))

# Build final model-ready frames.
train_out = pd.DataFrame(train_scaled, columns=feature_names)
val_out = pd.DataFrame(val_scaled, columns=feature_names)
test_out = pd.DataFrame(test_scaled, columns=feature_names)

if id_col is not None and id_col in train.columns:
    train_out["id"] = train[id_col].iloc[train_idx].astype(int).values
    val_out["id"] = train[id_col].iloc[val_idx].astype(int).values
else:
    train_out["id"] = np.arange(len(train_out), dtype=int)
    val_out["id"] = np.arange(len(val_out), dtype=int)

train_out[TARGET] = y_train.astype(float)
train_out[TIME_COL] = time_train.astype(float)
val_out[TARGET] = y_val.astype(float)
val_out[TIME_COL] = time_val.astype(float)

if id_col is not None and id_col in test.columns:
    test_out["id"] = test[id_col].astype(int).values
else:
    test_out["id"] = np.arange(len(test_out), dtype=int)

# Persist artifacts for auditability.
train_out.to_parquet("./working/train_features.parquet")
val_out.to_parquet("./working/val_features.parquet")
test_out.to_parquet("./working/test_features.parquet")

with open("./working/feature_names.json", "w") as f:
    json.dump(feature_names, f)

with open("./working/preprocessing.json", "w") as f:
    json.dump(
        {
            "numeric_medians": numeric_medians,
            "onehot_categories": onehot_categories,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        },
        f,
    )

with open("./working/split_info.json", "w") as f:
    json.dump(
        {
            "train_size": int(len(train_out)),
            "val_size": int(len(val_out)),
            "test_size": int(len(test_out)),
            "stratify": "efs + efs_time quartile",
            "random_state": RANDOM_STATE,
        },
        f,
    )

print(
    f"Train features: {train_out.shape}, Val features: {val_out.shape}, Test features: {test_out.shape}"
)
print(f"Feature count: {len(feature_names)}")

# =============================================================================
# 4. Model design from Step 2
# =============================================================================
with open("./working/feature_names.json", "r") as f:
    feature_names = json.load(f)
input_dim = len(feature_names)


class EFSMLP(nn.Module):
    """
    Regularized MLP for EFS binary classification.
    Uses BatchNorm and Dropout to handle tabular feature interactions robustly.
    """

    def __init__(self, input_dim: int):
        super(EFSMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


model = EFSMLP(input_dim)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

print(f"Model architecture defined: EFSMLP (input_dim={input_dim})")

# =============================================================================
# 5. Training and evaluation from Step 3
# =============================================================================
X_train = train_out[feature_names].astype(np.float32).values
y_train = train_out[TARGET].astype(np.float32).values
time_train = train_out[TIME_COL].astype(np.float64).values

X_val = val_out[feature_names].astype(np.float32).values
y_val = val_out[TARGET].astype(np.float32).values
time_val = val_out[TIME_COL].astype(np.float64).values

X_test = test_out[feature_names].astype(np.float32).values

X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
train_loader = DataLoader(
    train_dataset,
    batch_size=512,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
)


# -----------------------------------------------------------------------------
# Survival concordance index implementation.
# Returns standard C-index where higher is better for the supplied risk score.
# -----------------------------------------------------------------------------
def _c_index_from_scores(time, event, score) -> float:
    """
    Compute the standard survival concordance index (higher is better) for
    right-censored data.

    Parameters
    ----------
    time : array-like
        Observed times.
    event : array-like
        Event indicator, 1 = event, 0 = censored.
    score : array-like
        Predicted risk score. Higher score should indicate higher risk.

    Returns
    -------
    float
        C-index in [0, 1]. Returns 0.5 if no comparable pairs exist.
    """
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)

    n = len(time)
    if n < 2:
        return 0.5

    valid = np.isfinite(time) & np.isfinite(score)
    if valid.sum() < 2:
        return 0.5
    time = time[valid]
    event = event[valid]
    score = score[valid]
    n = len(time)

    if event.sum() == 0:
        return 0.5

    # Coordinate-compress scores for Fenwick tree.
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(1, n + 1)

    # Sort by time ascending, then group equal times.
    time_order = np.argsort(time, kind="mergesort")
    sorted_time = time[time_order]
    sorted_event = event[time_order]
    sorted_rank = ranks[time_order]

    class Fenwick:
        def __init__(self, size: int):
            self.n = size
            self.tree = np.zeros(self.n + 1, dtype=np.int64)

        def add(self, i: int, delta: int = 1) -> None:
            while i <= self.n:
                self.tree[i] += delta
                i += i & (-i)

        def sum(self, i: int) -> int:
            s = 0
            while i > 0:
                s += self.tree[i]
                i -= i & (-i)
            return s

    bit = Fenwick(n)
    comparable = 0
    concordant = 0

    later_events = 0
    later_censored = 0

    # Process time groups from latest to earliest.
    i = n - 1
    while i >= 0:
        j = i
        while j >= 0 and sorted_time[j] == sorted_time[i]:
            j -= 1

        group_start = j + 1
        group_end = i

        # For each event in this time group, compare with strictly later observations.
        for k in range(group_start, group_end + 1):
            if sorted_event[k] == 1:
                comparable += later_events + later_censored
                # Concordant if the later observation has strictly lower risk score.
                concordant += int(bit.sum(int(sorted_rank[k]) - 1))

        # Add the whole time group for earlier-time comparisons.
        group_events = int(sorted_event[group_start : group_end + 1].sum())
        group_censored = (group_end - group_start + 1) - group_events
        later_events += group_events
        later_censored += group_censored

        for k in range(group_start, group_end + 1):
            bit.add(int(sorted_rank[k]), 1)

        i = j

    if comparable == 0:
        return 0.5

    return float(concordant) / float(comparable)


def _decide_favorable_y1(y: np.ndarray, time: np.ndarray) -> bool:
    """
    Decide whether y=1 represents the favorable/event-free outcome.

    This is used only to orient the survival concordance metric consistently
    with the target semantics. It is fit on the training split only.
    """
    y = np.asarray(y, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)

    valid = np.isfinite(y) & np.isfinite(time)
    if valid.sum() < 2:
        return True

    yv = y[valid]
    tv = time[valid]

    mean_time_y1 = np.nanmean(tv[yv == 1]) if np.any(yv == 1) else np.nan
    mean_time_y0 = np.nanmean(tv[yv == 0]) if np.any(yv == 0) else np.nan

    if np.isfinite(mean_time_y1) and np.isfinite(mean_time_y0):
        return bool(mean_time_y1 >= mean_time_y0)
    return True


def compute_lower_is_better_concordance(
    y_true: np.ndarray,
    time_true: np.ndarray,
    score_pred: np.ndarray,
    favorable_y1: bool,
) -> float:
    """
    Return the task's lower-is-better concordance score.

    If y=1 is favorable/event-free:
        event = 1 - y
        risk  = 1 - score

    If y=1 is event:
        event = y
        risk  = score

    Then:
        C = survival concordance index(event, time, risk)  # higher is better
        score = 1 - C                                      # lower is better
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    time_true = np.asarray(time_true, dtype=np.float64)
    score_pred = np.asarray(score_pred, dtype=np.float64)

    if favorable_y1:
        event = (1.0 - y_true).astype(np.int8)
        risk = 1.0 - score_pred
    else:
        event = y_true.astype(np.int8)
        risk = score_pred

    c_index = _c_index_from_scores(time_true, event, risk)
    return float(1.0 - c_index)


# Decide target orientation once from the training split only.
favorable_y1 = _decide_favorable_y1(y_train, time_train)


# -----------------------------------------------------------------------------
# Inference helper: identical for validation and test.
# -----------------------------------------------------------------------------
@torch.no_grad()
def predict_proba(
    X_np: np.ndarray, model: nn.Module, device: torch.device, batch_size: int = 1024
) -> np.ndarray:
    model.eval()
    X_t = torch.from_numpy(np.asarray(X_np, dtype=np.float32))
    preds = []
    for start in range(0, len(X_t), batch_size):
        xb = X_t[start : start + batch_size].to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds.append(probs)
    probs = np.concatenate(preds, axis=0) if preds else np.zeros(0, dtype=np.float64)
    return np.clip(probs, 0.0, 1.0)


# -----------------------------------------------------------------------------
# Training loop with validation-based early stopping on the official-style metric.
# -----------------------------------------------------------------------------
try:
    optimizer.zero_grad(set_to_none=True)
except Exception:
    optimizer.zero_grad()

try:
    scheduler.T_max = 100
except Exception:
    pass

max_epochs = 100
patience = 12
best_score = float("inf")
best_state = None
best_epoch = 0
no_improve = 0

for epoch in range(1, max_epochs + 1):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / max(1, n_batches)

    val_probs = predict_proba(X_val, model, device)
    val_score = compute_lower_is_better_concordance(
        y_val,
        time_val,
        val_probs,
        favorable_y1=favorable_y1,
    )

    print(
        f"Epoch {epoch:03d} | loss {avg_loss:.6f} | val_concordance_lower_better {val_score:.6f}"
    )

    if val_score < best_score - 1e-6:
        best_score = val_score
        best_epoch = epoch
        no_improve = 0
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch:03d}; best epoch {best_epoch:03d}")
            break

# -----------------------------------------------------------------------------
# Restore best model and compute final validation score.
# -----------------------------------------------------------------------------
if best_state is not None:
    model.load_state_dict(best_state)
    model.to(device)

val_probs = predict_proba(X_val, model, device)
final_val_score = compute_lower_is_better_concordance(
    y_val,
    time_val,
    val_probs,
    favorable_y1=favorable_y1,
)

with open("./working/best_model_state.pkl", "wb") as f:
    pickle.dump(
        {
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "final_val_score": final_val_score,
            "feature_names": feature_names,
            "device": str(device),
        },
        f,
    )

# -----------------------------------------------------------------------------
# Test inference and submission.
# -----------------------------------------------------------------------------
test_probs = predict_proba(X_test, model, device)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test_probs), dtype=int),
        "efs": test_probs.astype(np.float64),
    }
)

submission.to_csv(SUBMISSION_PATH, index=False)

print(f"Saved submission: {submission.shape[0]} rows")
print(f"Best epoch: {best_epoch}, best validation score: {best_score:.6f}")

score = final_val_score
print(f"Final Validation Score: {score}")
