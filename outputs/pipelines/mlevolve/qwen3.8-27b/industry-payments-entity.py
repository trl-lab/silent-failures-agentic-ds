import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------
# Paths and constants
# -----------------------------
INPUT_DIR = "./input"
WORK_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

TARGET = "entityPayment"
RANDOM_STATE = 42
VAL_FRACTION = 0.20
SMOOTH_M = 10.0
ONEHOT_CARD_LIMIT = 50

SEED = 42
MAX_EPOCHS = 180
PATIENCE = 20
BATCH_SIZE = 128
HIDDEN = 128
DROPOUT = 0.18
WEIGHT_DECAY = 1e-4
LR = 3e-3

np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------
# Load raw data
# -----------------------------
train_path = os.path.join(INPUT_DIR, "train.parquet")
test_path = os.path.join(INPUT_DIR, "test.parquet")

train_raw = pd.read_parquet(train_path)
test_raw = pd.read_parquet(test_path)

# Robust target resolution
if TARGET not in train_raw.columns:
    payment_cols = [c for c in train_raw.columns if "payment" in c.lower()]
    if payment_cols:
        TARGET = payment_cols[0]
    else:
        TARGET = train_raw.columns[-1]

# Preserve ids
test_id_col = "id" if "id" in test_raw.columns else None
train_id_col = "id" if "id" in train_raw.columns else None

# Clean target in train
train_raw[TARGET] = pd.to_numeric(train_raw[TARGET], errors="coerce")
train_raw = train_raw.replace([np.inf, -np.inf], np.nan)
train_raw = train_raw.dropna(subset=[TARGET])

# If target exists in test, drop it
if TARGET in test_raw.columns:
    test_raw = test_raw.drop(columns=[TARGET])

# If no explicit id, use 0-based index
if test_id_col is None:
    test_raw = test_raw.reset_index(drop=True)
    test_ids = np.arange(len(test_raw), dtype=int)
else:
    test_ids = test_raw[test_id_col].astype(int).to_numpy()

# -----------------------------
# Split FIRST
# -----------------------------
if len(train_raw) >= 100:
    try:
        strat = pd.qcut(train_raw[TARGET], q=10, labels=False, duplicates="drop")
    except Exception:
        strat = None
else:
    strat = None

if len(train_raw) > 0:
    train_full, val_full = train_test_split(
        train_raw,
        test_size=VAL_FRACTION,
        random_state=RANDOM_STATE,
        stratify=strat,
    )
else:
    train_full = train_raw.copy()
    val_full = train_raw.iloc[0:0].copy()


# -----------------------------
# Helpers
# -----------------------------
def is_id_like(col: str) -> bool:
    c = str(col).lower()
    tokens = ["id", "index", "row", "uid", "key", "code", "identifier"]
    return any(t in c for t in tokens)


def is_group_like(col: str) -> bool:
    c = str(col).lower()
    tokens = [
        "entity",
        "org",
        "company",
        "merchant",
        "customer",
        "industry",
        "sector",
        "category",
        "class",
        "group",
    ]
    return any(t in c for t in tokens)


def safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)


def add_feature(out: pd.DataFrame, name: str, values) -> None:
    if isinstance(values, pd.Series):
        values = values.to_numpy()
    values = np.asarray(values)
    if values.shape[0] != len(out):
        raise ValueError(f"Feature {name} length mismatch")
    out[name] = values


def _numeric_matrix(df: pd.DataFrame, state: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    medians = state["numeric_medians"]

    for spec in state["numeric_derived_specs"]:
        c = spec["col"]
        vals = pd.to_numeric(df[c], errors="coerce").fillna(medians[c])
        add_feature(out, c, vals)

        if spec["nonnegative"]:
            add_feature(out, f"{c}_log1p", np.log1p(np.maximum(vals, 0.0)))
            add_feature(out, f"{c}_sqrt", np.sqrt(np.maximum(vals, 0.0)))

        add_feature(out, f"{c}_capped", np.clip(vals, spec["q01"], spec["q99"]))
        add_feature(out, f"{c}_abs", np.abs(vals))
        add_feature(out, f"{c}_sign", np.sign(vals))

    return out


def _transform(df: pd.DataFrame, state: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    target_col = state["target_col"]

    # Missingness
    for c in state["missing_cols"]:
        if c in df.columns:
            add_feature(out, f"{c}_missing", df[c].isna().astype(int).to_numpy())
        else:
            add_feature(out, f"{c}_missing", np.zeros(len(df), dtype=int))

    # Numeric features + scaled numeric features
    X_num = _numeric_matrix(df, state)
    for c in X_num.columns:
        add_feature(out, c, X_num[c].to_numpy())

    X_scaled = pd.DataFrame(
        state["scaler"].transform(X_num.to_numpy()),
        columns=state["scaled_numeric_cols"],
        index=df.index,
    )
    for c in X_scaled.columns:
        add_feature(out, f"{c}_scaled", X_scaled[c].to_numpy())

    # Categorical ordinal codes
    for c in state["cat_cols"]:
        if c not in df.columns:
            add_feature(out, f"{c}_code", np.full(len(df), -1, dtype=int))
            continue
        s = df[c].astype("string").fillna(state["cat_modes"][c])
        mapping = state["ordinal_map"].get(c, {})
        codes = s.map(lambda x: mapping.get(str(x), -1)).astype(int).to_numpy()
        add_feature(out, f"{c}_code", codes)

    # One-hot
    for c in state["onehot_cols"]:
        feat_names = state["onehot_feature_names"][c]
        if c not in df.columns:
            for fn in feat_names:
                add_feature(out, fn, np.zeros(len(df), dtype=float))
            continue

        s = df[c].astype("string").fillna(state["cat_modes"][c])
        enc = state["onehot_encoders"][c]
        oh = enc.transform(s.to_numpy().reshape(-1, 1))
        # Handle both sparse (scipy) and dense (numpy) encoder outputs robustly.
        if hasattr(oh, "toarray"):
            oh = oh.toarray()
        else:
            oh = np.asarray(oh)

        for i, fn in enumerate(feat_names):
            if i < oh.shape[1]:
                add_feature(out, fn, oh[:, i])
            else:
                add_feature(out, fn, np.zeros(len(df), dtype=float))

    # Target-encoded group statistics
    global_mean = state["global_target_mean"]
    for c, stats in state["target_stats"].items():
        if c == "__interaction__":
            if len(state["interaction_cols"]) != 2:
                continue
            a, b = state["interaction_cols"]
            if a not in df.columns or b not in df.columns:
                continue
            key = (
                df[a].astype("string").fillna(state["cat_modes"][a])
                + "||"
                + df[b].astype("string").fillna(state["cat_modes"][b])
            )
            prefix = "interaction"
        else:
            if c not in df.columns:
                continue
            key = df[c].astype("string").fillna(state["cat_modes"][c])
            prefix = c

        add_feature(
            out,
            f"{prefix}_te_mean",
            key.map(stats["smooth_mean"]).fillna(global_mean).to_numpy(),
        )
        add_feature(
            out, f"{prefix}_te_count", key.map(stats["count"]).fillna(0.0).to_numpy()
        )
        add_feature(
            out, f"{prefix}_te_std", key.map(stats["std"]).fillna(0.0).to_numpy()
        )
        add_feature(
            out,
            f"{prefix}_te_min",
            key.map(stats["min"]).fillna(global_mean).to_numpy(),
        )
        add_feature(
            out,
            f"{prefix}_te_max",
            key.map(stats["max"]).fillna(global_mean).to_numpy(),
        )
        add_feature(
            out,
            f"{prefix}_te_median",
            key.map(stats["median"]).fillna(global_mean).to_numpy(),
        )

    # Date features
    for c in state["datetime_cols"]:
        if c not in df.columns:
            continue
        dt = safe_to_datetime(df[c])
        origin = state["date_origin"]
        add_feature(out, f"{c}_year", dt.dt.year.fillna(0).astype(int).to_numpy())
        add_feature(out, f"{c}_month", dt.dt.month.fillna(0).astype(int).to_numpy())
        add_feature(out, f"{c}_day", dt.dt.day.fillna(0).astype(int).to_numpy())
        add_feature(
            out, f"{c}_dayofweek", dt.dt.dayofweek.fillna(0).astype(int).to_numpy()
        )
        add_feature(out, f"{c}_quarter", dt.dt.quarter.fillna(0).astype(int).to_numpy())
        add_feature(
            out, f"{c}_dayofyear", dt.dt.dayofyear.fillna(0).astype(int).to_numpy()
        )
        add_feature(
            out,
            f"{c}_is_weekend",
            (dt.dt.dayofweek >= 5).fillna(False).astype(int).to_numpy(),
        )
        if origin is not None:
            days = (dt - origin).dt.total_seconds().fillna(0.0) / 86400.0
            add_feature(out, f"{c}_days_since_origin", days.to_numpy())

    return out


def build_state(train_fit: pd.DataFrame, target_col: str):
    feature_cols = [
        c for c in train_fit.columns if c not in {target_col, train_id_col, test_id_col}
    ]

    numeric_cols = []
    cat_cols = []
    datetime_cols = []

    for c in feature_cols:
        s = train_fit[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            datetime_cols.append(c)
        elif pd.api.types.is_bool_dtype(s):
            cat_cols.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            if is_id_like(c) or is_group_like(c):
                cat_cols.append(c)
            else:
                numeric_cols.append(c)
        else:
            cat_cols.append(c)

    state = {
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
        "datetime_cols": datetime_cols,
        "target_col": target_col,
    }

    # Missingness indicators
    state["missing_cols"] = feature_cols

    # Numeric imputer medians
    medians = {}
    for c in numeric_cols:
        vals = pd.to_numeric(train_fit[c], errors="coerce")
        med = float(vals.median()) if vals.notna().any() else 0.0
        medians[c] = med
    state["numeric_medians"] = medians

    # Categorical mode imputer
    modes = {}
    for c in cat_cols:
        s = train_fit[c].astype("string")
        mode = s.mode(dropna=True)
        modes[c] = mode.iloc[0] if len(mode) > 0 else "__missing__"
    state["cat_modes"] = modes

    # Date origin
    if datetime_cols:
        dt = safe_to_datetime(train_fit[datetime_cols[0]])
        origin = dt.min() if dt.notna().any() else pd.Timestamp("2000-01-01", tz="UTC")
        state["date_origin"] = origin
    else:
        state["date_origin"] = None

    # One-hot encoders for low-cardinality categoricals
    onehot_encoders = {}
    onehot_cols = []
    onehot_feature_names = {}
    for c in cat_cols:
        s = train_fit[c].astype("string").fillna(modes[c])
        if s.nunique(dropna=True) <= ONEHOT_CARD_LIMIT:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            enc.fit(s.to_numpy().reshape(-1, 1))
            onehot_encoders[c] = enc
            onehot_cols.append(c)
            onehot_feature_names[c] = [
                f"{c}_oh_{i}" for i in range(len(enc.categories_[0]))
            ]

    state["onehot_encoders"] = onehot_encoders
    state["onehot_cols"] = onehot_cols
    state["onehot_feature_names"] = onehot_feature_names

    # Ordinal maps from training split only
    ordinal_map = {}
    for c in cat_cols:
        s = train_fit[c].astype("string").fillna(modes[c])
        cats = sorted([str(x) for x in s.unique()])
        ordinal_map[c] = {cat: i for i, cat in enumerate(cats)}
    state["ordinal_map"] = ordinal_map

    # Target statistics from training split only
    y = train_fit[target_col].astype(float)
    global_mean = float(y.mean()) if len(y) > 0 else 0.0
    state["global_target_mean"] = global_mean

    target_stats = {}
    for c in cat_cols:
        s = train_fit[c].astype("string").fillna(modes[c])
        tmp = pd.DataFrame({"key": s, "y": y})
        agg = tmp.groupby("key")["y"].agg(
            ["sum", "count", "std", "min", "max", "median"]
        )
        smooth = (global_mean * SMOOTH_M + agg["sum"]) / (SMOOTH_M + agg["count"])
        target_stats[c] = {
            "smooth_mean": smooth.to_dict(),
            "count": agg["count"].to_dict(),
            "std": agg["std"].fillna(0.0).to_dict(),
            "min": agg["min"].to_dict(),
            "max": agg["max"].to_dict(),
            "median": agg["median"].to_dict(),
        }

    # Interaction target encoding: prefer entity x industry, else top two group-like categoricals
    interaction_cols = []
    entity_col = next((c for c in cat_cols if "entity" in c.lower()), None)
    industry_col = next(
        (c for c in cat_cols if "industry" in c.lower() or "sector" in c.lower()), None
    )
    if entity_col and industry_col:
        interaction_cols = [entity_col, industry_col]
    else:
        group_cols = [c for c in cat_cols if is_group_like(c)]
        if len(group_cols) >= 2:
            interaction_cols = group_cols[:2]
        elif len(cat_cols) >= 2:
            interaction_cols = cat_cols[:2]

    state["interaction_cols"] = interaction_cols

    if interaction_cols:
        a, b = interaction_cols[0], interaction_cols[1]
        key = (
            train_fit[a].astype("string").fillna(modes[a])
            + "||"
            + train_fit[b].astype("string").fillna(modes[b])
        )
        tmp = pd.DataFrame({"key": key, "y": y})
        agg = tmp.groupby("key")["y"].agg(
            ["sum", "count", "std", "min", "max", "median"]
        )
        smooth = (global_mean * SMOOTH_M + agg["sum"]) / (SMOOTH_M + agg["count"])
        target_stats["__interaction__"] = {
            "smooth_mean": smooth.to_dict(),
            "count": agg["count"].to_dict(),
            "std": agg["std"].fillna(0.0).to_dict(),
            "min": agg["min"].to_dict(),
            "max": agg["max"].to_dict(),
            "median": agg["median"].to_dict(),
        }

    state["target_stats"] = target_stats

    # Numeric derived feature specs
    numeric_derived_specs = []
    for c in numeric_cols:
        vals = pd.to_numeric(train_fit[c], errors="coerce").fillna(medians[c])
        min_val = float(vals.min()) if len(vals) > 0 else 0.0
        max_val = float(vals.max()) if len(vals) > 0 else 0.0
        q01 = float(vals.quantile(0.01)) if len(vals) > 0 else 0.0
        q99 = float(vals.quantile(0.99)) if len(vals) > 0 else 0.0
        numeric_derived_specs.append(
            {
                "col": c,
                "min": min_val,
                "max": max_val,
                "q01": q01,
                "q99": q99,
                "nonnegative": min_val >= 0,
            }
        )
    state["numeric_derived_specs"] = numeric_derived_specs

    # Fit scaler on training-derived numeric matrix
    X_num_fit = _numeric_matrix(train_fit, state)
    scaler = RobustScaler()
    scaler.fit(X_num_fit.to_numpy())
    state["scaler"] = scaler
    state["scaled_numeric_cols"] = list(X_num_fit.columns)

    # Build feature names by transforming training fit
    X_fit = _transform(train_fit, state)
    state["feature_names"] = list(X_fit.columns)

    return state


# -----------------------------
# Build and transform features
# -----------------------------
state = build_state(train_full, TARGET)

X_train = _transform(train_full, state)
X_val = _transform(val_full, state)
X_test = _transform(test_raw, state)

# Ensure same feature order
X_train = X_train.reindex(columns=state["feature_names"], fill_value=0.0)
X_val = X_val.reindex(columns=state["feature_names"], fill_value=0.0)
X_test = X_test.reindex(columns=state["feature_names"], fill_value=0.0)

# Replace non-finite values
X_train = X_train.replace([np.inf, -np.inf], 0.0).fillna(0.0)
X_val = X_val.replace([np.inf, -np.inf], 0.0).fillna(0.0)
X_test = X_test.replace([np.inf, -np.inf], 0.0).fillna(0.0)

y_train = train_full[TARGET].astype(float).to_numpy()
y_val = val_full[TARGET].astype(float).to_numpy()

# Save artifacts for auditability
X_train.to_parquet(os.path.join(WORK_DIR, "X_train.parquet"))
X_val.to_parquet(os.path.join(WORK_DIR, "X_val.parquet"))
X_test.to_parquet(os.path.join(WORK_DIR, "X_test.parquet"))
pd.Series(y_train).to_frame(TARGET).to_parquet(
    os.path.join(WORK_DIR, "y_train.parquet")
)
pd.Series(y_val).to_frame(TARGET).to_parquet(os.path.join(WORK_DIR, "y_val.parquet"))
np.save(os.path.join(WORK_DIR, "test_ids.npy"), test_ids)

with open(os.path.join(WORK_DIR, "feature_names.json"), "w") as f:
    json.dump(state["feature_names"], f)

# -----------------------------
# Model input layout: numeric branch + categorical embedding branch
# -----------------------------
feature_names = list(X_train.columns)
cat_cols = [c for c in feature_names if str(c).endswith("_code")]
numeric_cols = [c for c in feature_names if c not in cat_cols]

# Prior features: smoothed target means from entity/industry/interaction features.
prior_candidates = [c for c in numeric_cols if "_te_mean" in str(c)]


def prior_priority(c: str) -> int:
    c = str(c).lower()
    if "interaction" in c:
        return 0
    if "entity" in c:
        return 1
    if "industry" in c or "sector" in c:
        return 2
    return 3


prior_cols = sorted(prior_candidates, key=prior_priority)[:4]

# Categorical embedding specs fitted on training codes only.
cat_specs = []
for c in cat_cols:
    vals = pd.to_numeric(X_train[c], errors="coerce").fillna(-1).astype(int).to_numpy()
    if len(vals) == 0:
        continue
    if np.all(vals == -1):
        continue

    max_code = int(np.nanmax(vals))
    if max_code < 0:
        continue

    # Codes are shifted to codes + 2, with 0 reserved for padding/unknown.
    # Therefore embeddings must have at least max_code + 3 rows.
    max_card = max_code + 1
    if max_card > 65534:
        num_embeddings = 65536
        use_hash = True
    else:
        num_embeddings = max_code + 3
        use_hash = False

    dim = int(np.clip(2 * np.sqrt(max_card + 1), 2, 16))
    cat_specs.append((c, num_embeddings, dim, use_hash))

# -----------------------------
# Target-level bounds from training only
# -----------------------------
global_mean = float(np.mean(y_train)) if len(y_train) > 0 else 0.0
train_min = float(np.min(y_train)) if len(y_train) > 0 else 0.0
train_max = float(np.max(y_train)) if len(y_train) > 0 else 0.0

if train_max <= train_min:
    lower = upper = train_min
else:
    lower = max(0.0, train_min) if train_min >= 0 else train_min
    upper = train_max

# -----------------------------
# Numeric standardization fitted on training only
# -----------------------------
if len(numeric_cols) > 0:
    num_mean = X_train[numeric_cols].mean(axis=0).fillna(0.0).to_numpy(dtype=np.float32)
    num_std = X_train[numeric_cols].std(axis=0).fillna(0.0).to_numpy(dtype=np.float32)
    num_std[num_std == 0.0] = 1.0
else:
    num_mean = np.zeros(0, dtype=np.float32)
    num_std = np.ones(0, dtype=np.float32)


def to_numeric_tensor(df: pd.DataFrame) -> torch.Tensor:
    if len(numeric_cols) == 0:
        return torch.zeros((len(df), 0), dtype=torch.float32)
    arr = df[numeric_cols].to_numpy(dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(arr)


def to_prior_tensor(df: pd.DataFrame) -> torch.Tensor:
    if len(prior_cols) == 0:
        return torch.zeros((len(df), 0), dtype=torch.float32)
    arr = df[prior_cols].to_numpy(dtype=np.float32)
    arr = np.nan_to_num(arr, nan=global_mean, posinf=global_mean, neginf=global_mean)
    return torch.from_numpy(arr)


def to_cat_tensor(df: pd.DataFrame) -> torch.Tensor:
    if len(cat_specs) == 0:
        return torch.zeros((len(df), 0), dtype=torch.int64)

    cols = [c for c, _, _, _ in cat_specs]
    arr = df[cols].to_numpy(dtype=np.int64)
    arr = np.nan_to_num(arr, nan=-1).astype(np.int64)

    out = np.zeros((len(df), len(cat_specs)), dtype=np.int64)
    for j, (c, num_embeddings, dim, use_hash) in enumerate(cat_specs):
        codes = arr[:, j]
        shifted = np.where(codes < 0, 0, codes + 2)
        if use_hash:
            shifted = shifted % num_embeddings
        out[:, j] = shifted

    return torch.from_numpy(out).long()


# -----------------------------
# Model architecture
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.norm(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return h + x


class HierarchicalPaymentNet(nn.Module):
    """
    Output = hierarchical prior + learned residual.

    The prior is a learned convex combination of the global mean and smoothed
    group means. The residual branch uses standardized numeric features and
    embeddings for ordinal categorical codes.
    """

    def __init__(
        self,
        num_numeric: int,
        cat_specs: list,
        n_prior: int,
        global_mean: float,
        lower: float,
        upper: float,
        num_mean: np.ndarray,
        num_std: np.ndarray,
        hidden: int = HIDDEN,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.cat_specs = cat_specs
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings, dim, padding_idx=0)
                for _, num_embeddings, dim, _ in cat_specs
            ]
        )
        cat_dim = sum(dim for _, _, dim, _ in cat_specs)
        in_dim = max(1, num_numeric + cat_dim)

        self.input_proj = nn.Linear(in_dim, hidden)
        self.input_norm = nn.LayerNorm(hidden)
        self.input_dropout = nn.Dropout(dropout)

        self.block1 = ResidualBlock(hidden, dropout)
        self.block2 = ResidualBlock(hidden, dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden, max(hidden // 2, 16)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden // 2, 16), 1),
        )

        # Start residual at zero so initial model output is the hierarchical prior.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

        # Start prior almost entirely at the global mean; learn group weights.
        if n_prior > 0:
            self.prior_logits = nn.Parameter(
                torch.full((n_prior,), -10.0, dtype=torch.float32)
            )
        else:
            self.prior_logits = None

        self.register_buffer(
            "global_mean", torch.tensor(global_mean, dtype=torch.float32)
        )
        self.register_buffer("lower", torch.tensor(lower, dtype=torch.float32))
        self.register_buffer("upper", torch.tensor(upper, dtype=torch.float32))
        self.register_buffer("num_mean", torch.from_numpy(num_mean.astype(np.float32)))
        self.register_buffer("num_std", torch.from_numpy(num_std.astype(np.float32)))

    def _prior(self, x_prior: torch.Tensor) -> torch.Tensor:
        if self.prior_logits is None or x_prior.shape[1] == 0:
            return self.global_mean.expand(x_prior.shape[0])

        # Convex combination over group priors plus a fixed global logit.
        logits = torch.cat(
            [self.prior_logits, torch.zeros(1, device=self.prior_logits.device)]
        )
        weights = F.softmax(logits, dim=0)
        group_weights = weights[:-1]
        return self.global_mean + torch.sum(
            group_weights * (x_prior - self.global_mean), dim=1
        )

    def forward(
        self,
        x_numeric: torch.Tensor,
        x_prior: torch.Tensor,
        x_cat: torch.Tensor,
        apply_clip: bool = True,
    ) -> torch.Tensor:
        x_num = (x_numeric - self.num_mean) / self.num_std

        parts = []
        if x_numeric.shape[1] > 0:
            parts.append(x_num)

        if len(self.embeddings) > 0:
            cat_parts = [
                self.embeddings[i](x_cat[:, i]) for i in range(len(self.embeddings))
            ]
            if cat_parts:
                parts.append(torch.cat(cat_parts, dim=1))

        if len(parts) == 0:
            h = torch.ones(x_numeric.shape[0], 1, device=x_numeric.device)
        else:
            h = torch.cat(parts, dim=1)

        h = self.input_proj(h)
        h = self.input_norm(h)
        h = F.gelu(h)
        h = self.input_dropout(h)
        h = self.block1(h)
        h = self.block2(h)
        residual = self.head(h).squeeze(-1)

        out = self._prior(x_prior) + residual

        if apply_clip:
            out = torch.clamp(out, self.lower, self.upper)
        return out


model = HierarchicalPaymentNet(
    num_numeric=len(numeric_cols),
    cat_specs=cat_specs,
    n_prior=len(prior_cols),
    global_mean=global_mean,
    lower=lower,
    upper=upper,
    num_mean=num_mean,
    num_std=num_std,
)

# -----------------------------
# Loss and optimizer
# -----------------------------
# Official metric is RMSE on raw entityPayment; MSE is the direct training surrogate.
criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max(1, MAX_EPOCHS), eta_min=1e-5
)

# -----------------------------
# Tensors and data loader
# -----------------------------
X_train_num = to_numeric_tensor(X_train)
X_train_prior = to_prior_tensor(X_train)
X_train_cat = to_cat_tensor(X_train)
y_train_t = torch.from_numpy(y_train.astype(np.float32))

X_val_num = to_numeric_tensor(X_val)
X_val_prior = to_prior_tensor(X_val)
X_val_cat = to_cat_tensor(X_val)
y_val_t = torch.from_numpy(y_val.astype(np.float32))

X_test_num = to_numeric_tensor(X_test)
X_test_prior = to_prior_tensor(X_test)
X_test_cat = to_cat_tensor(X_test)

train_ds = TensorDataset(X_train_num, X_train_prior, X_train_cat, y_train_t)
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2 if len(train_ds) > 0 else 0,
)


@torch.no_grad()
def predict(
    df_num: torch.Tensor, df_prior: torch.Tensor, df_cat: torch.Tensor
) -> np.ndarray:
    model.eval()
    preds = []
    bs = 4096
    for i in range(0, len(df_num), bs):
        xn = df_num[i : i + bs]
        xp = df_prior[i : i + bs]
        xc = df_cat[i : i + bs]
        preds.append(model(xn, xp, xc, apply_clip=True).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=float)


# -----------------------------
# Training with official-metric early stopping
# -----------------------------
best_val_rmse = float("inf")
best_state = None
epochs_no_improve = 0

if len(X_train) > 0:
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for xn, xp, xc, yt in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xn, xp, xc, apply_clip=True)
            loss = criterion(pred, yt)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item())
            n_batches += 1

        scheduler.step()

        val_pred = predict(X_val_num, X_val_prior, X_val_cat)
        if len(y_val) > 0:
            val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
        else:
            val_rmse = float("nan")
        train_rmse = float(np.sqrt(train_loss_sum / max(1, n_batches)))

        print(f"epoch={epoch:03d} train_rmse={train_rmse:.6f} val_rmse={val_rmse:.6f}")

        if not np.isnan(val_rmse) and val_rmse < best_val_rmse - 1e-7:
            best_val_rmse = val_rmse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

# -----------------------------
# Final validation and test inference
# -----------------------------
val_pred = predict(X_val_num, X_val_prior, X_val_cat)
if len(y_val) > 0:
    score = float(np.sqrt(mean_squared_error(y_val, val_pred)))
else:
    score = float("nan")

test_pred = predict(X_test_num, X_test_prior, X_test_cat)

submission = pd.DataFrame(
    {
        "id": test_ids.astype(int),
        "entityPayment": test_pred.astype(float),
    }
)
submission.to_csv(os.path.join(SUBMISSION_DIR, "submission.csv"), index=False)

print(f"Final Validation Score: {score}")
