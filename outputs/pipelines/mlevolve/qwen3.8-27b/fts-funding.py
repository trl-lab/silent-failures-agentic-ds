import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

TARGET = "amountUSD"
SEED = 42
VAL_FRACTION = 0.2

torch.manual_seed(SEED)
np.random.seed(SEED)


def find_id_column(df: pd.DataFrame, target: str) -> str:
    preferred = ["id", "Id", "ID", "index", "row_id", "sample_id", "submission_id"]
    for col in preferred:
        if col in df.columns:
            return col

    for col in df.columns:
        if col == target:
            continue
        if pd.api.types.is_integer_dtype(df[col]) and df[col].nunique(
            dropna=True
        ) > 0.8 * len(df):
            return col

    return "id"


def is_date_series(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if s.dtype == object or str(s.dtype).startswith("string"):
        sample = s.dropna().astype(str).head(200)
        if sample.empty:
            return False
        parsed = pd.to_datetime(sample, errors="coerce")
        return parsed.notna().mean() > 0.8
    return False


def is_text_series(s: pd.Series) -> bool:
    if s.dtype != object and not str(s.dtype).startswith("string"):
        return False
    sample = s.dropna().astype(str).head(200)
    if sample.empty:
        return False
    lengths = sample.str.len()
    return lengths.mean() > 20 or lengths.max() > 100


def is_id_like_column(col: str, s: pd.Series) -> bool:
    if "id" in str(col).lower():
        return True
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) > 0.8 * len(s):
        return True
    return False


def is_constant_series(s: pd.Series) -> bool:
    return s.nunique(dropna=True) <= 1


def safe_float_series(s: pd.Series, fill: float = 0.0) -> pd.Series:
    return (
        pd.to_numeric(s, errors="coerce")
        .replace([np.inf, -np.inf], fill)
        .fillna(fill)
        .astype(np.float32)
    )


def clean_target_series(s: pd.Series) -> np.ndarray:
    return (
        pd.to_numeric(s, errors="coerce")
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
        .values.astype(np.float32)
    )


# -----------------------------------------------------------------------------
# 1. Load raw data
# -----------------------------------------------------------------------------
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

if TARGET not in train.columns:
    raise ValueError(f"Target column '{TARGET}' not found in train.parquet")

id_col = find_id_column(train, TARGET)
if id_col not in train.columns:
    train[id_col] = np.arange(len(train), dtype=np.int64)

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)


# -----------------------------------------------------------------------------
# 2. Split training data BEFORE any fitted feature engineering.
# Prefer temporal split if a reliable date column exists; otherwise random split.
# -----------------------------------------------------------------------------
if len(train) <= 1:
    tr = train.iloc[:0].copy()
    va = train.copy()
else:
    date_candidates = [
        c
        for c in train.columns
        if c not in [id_col, TARGET] and is_date_series(train[c])
    ]

    split_done = False
    if date_candidates:
        date_col = date_candidates[0]
        tmp = train.copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        if tmp[date_col].notna().mean() > 0.5:
            tmp = tmp.sort_values(date_col, kind="mergesort").reset_index(drop=True)
            n_val = max(1, int(round(VAL_FRACTION * len(tmp))))
            n_val = min(n_val, len(tmp) - 1 if len(tmp) > 1 else len(tmp))
            va = tmp.iloc[-n_val:].copy()
            tr = tmp.iloc[:-n_val].copy() if n_val < len(tmp) else tmp.iloc[:0].copy()
            split_done = True

    if not split_done:
        tr, va = train_test_split(
            train,
            test_size=VAL_FRACTION,
            random_state=SEED,
            shuffle=True,
        )

    tr = tr.reset_index(drop=True)
    va = va.reset_index(drop=True)


# -----------------------------------------------------------------------------
# 3. Create raw feature frames (target and id removed)
# -----------------------------------------------------------------------------
def strip_id_target(df: pd.DataFrame, include_target: bool = False) -> pd.DataFrame:
    drop_cols = [id_col]
    if not include_target:
        drop_cols.append(TARGET)
    return df.drop(columns=drop_cols, errors="ignore").copy()


tr_raw = strip_id_target(tr)
va_raw = strip_id_target(va)
te_raw = strip_id_target(test)

all_cols = list(
    dict.fromkeys(list(tr_raw.columns) + list(va_raw.columns) + list(te_raw.columns))
)
tr_raw = tr_raw.reindex(columns=all_cols)
va_raw = va_raw.reindex(columns=all_cols)
te_raw = te_raw.reindex(columns=all_cols)

keep_cols = []
for col in all_cols:
    s = tr_raw[col]
    if s.isna().all():
        continue
    if is_constant_series(s):
        continue
    if is_id_like_column(col, s):
        continue
    keep_cols.append(col)

tr_raw = tr_raw[keep_cols].copy()
va_raw = va_raw[keep_cols].copy()
te_raw = te_raw[keep_cols].copy()


# -----------------------------------------------------------------------------
# 4. Assign column roles using training split only.
# -----------------------------------------------------------------------------
numeric_cols = []
categorical_cols = []
date_cols = []
text_cols = []

for col in keep_cols:
    s = tr_raw[col]
    if is_date_series(s):
        date_cols.append(col)
    elif pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(col)
    elif s.dtype == bool:
        categorical_cols.append(col)
    elif (
        s.dtype == object or str(s.dtype).startswith("string") or s.dtype == "category"
    ):
        if is_text_series(s):
            text_cols.append(col)
        else:
            categorical_cols.append(col)
    else:
        categorical_cols.append(col)


# -----------------------------------------------------------------------------
# 5. Missingness indicators (no fitted state needed)
# -----------------------------------------------------------------------------
missing_cols = [col for col in keep_cols if tr_raw[col].isna().any()]


def add_missing_flags(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[f"{col}__missing"] = out[col].isna().astype(np.int8)
    return out


tr_raw = add_missing_flags(tr_raw, missing_cols)
va_raw = add_missing_flags(va_raw, missing_cols)
te_raw = add_missing_flags(te_raw, missing_cols)


# -----------------------------------------------------------------------------
# 6. Date feature engineering
# Fit only the minimum date from the training split.
# -----------------------------------------------------------------------------
date_min = {}
for col in date_cols:
    d = pd.to_datetime(tr_raw[col], errors="coerce")
    date_min[col] = d.min() if d.notna().any() else pd.Timestamp("1970-01-01")


def add_date_features(df: pd.DataFrame, cols, mins: dict) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        d = pd.to_datetime(out[col], errors="coerce")
        valid = d.notna()
        new_cols = []

        out[f"{col}__year"] = d.dt.year.astype(np.float32)
        new_cols.append(f"{col}__year")

        out[f"{col}__month"] = d.dt.month.astype(np.float32)
        new_cols.append(f"{col}__month")

        out[f"{col}__day"] = d.dt.day.astype(np.float32)
        new_cols.append(f"{col}__day")

        out[f"{col}__dayofweek"] = d.dt.dayofweek.astype(np.float32)
        new_cols.append(f"{col}__dayofweek")

        out[f"{col}__dayofyear"] = d.dt.dayofyear.astype(np.float32)
        new_cols.append(f"{col}__dayofyear")

        out[f"{col}__quarter"] = d.dt.quarter.astype(np.float32)
        new_cols.append(f"{col}__quarter")

        out[f"{col}__is_weekend"] = (d.dt.dayofweek >= 5).astype(np.float32)
        new_cols.append(f"{col}__is_weekend")

        days_since_min = (d - mins[col]).dt.days.astype(np.float32)
        out[f"{col}__days_since_min"] = days_since_min
        new_cols.append(f"{col}__days_since_min")

        out[f"{col}__month_sin"] = np.sin(2 * np.pi * d.dt.month / 12).astype(
            np.float32
        )
        new_cols.append(f"{col}__month_sin")

        out[f"{col}__month_cos"] = np.cos(2 * np.pi * d.dt.month / 12).astype(
            np.float32
        )
        new_cols.append(f"{col}__month_cos")

        out[f"{col}__dow_sin"] = np.sin(2 * np.pi * d.dt.dayofweek / 7).astype(
            np.float32
        )
        new_cols.append(f"{col}__dow_sin")

        out[f"{col}__dow_cos"] = np.cos(2 * np.pi * d.dt.dayofweek / 7).astype(
            np.float32
        )
        new_cols.append(f"{col}__dow_cos")

        for c in new_cols:
            out.loc[~valid, c] = 0.0

    return out.drop(columns=cols, errors="ignore")


tr_raw = add_date_features(tr_raw, date_cols, date_min)
va_raw = add_date_features(va_raw, date_cols, date_min)
te_raw = add_date_features(te_raw, date_cols, date_min)


# -----------------------------------------------------------------------------
# 7. Text feature engineering
# -----------------------------------------------------------------------------
def add_text_features(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        s = out[col].fillna("").astype(str)
        out[f"{col}__len"] = s.str.len().astype(np.float32)
        out[f"{col}__words"] = s.str.split().str.len().astype(np.float32)
        out[f"{col}__digits"] = s.str.count(r"\d").astype(np.float32)
        out[f"{col}__upper_ratio"] = (
            s.str.count(r"[A-Z]") / s.str.len().clip(lower=1)
        ).astype(np.float32)
        out[f"{col}__punct_ratio"] = (
            s.str.count(r"[^\w\s]") / s.str.len().clip(lower=1)
        ).astype(np.float32)
    return out.drop(columns=cols, errors="ignore")


tr_raw = add_text_features(tr_raw, text_cols)
va_raw = add_text_features(va_raw, text_cols)
te_raw = add_text_features(te_raw, text_cols)


# -----------------------------------------------------------------------------
# 8. Numeric imputation and transformations
# Fit medians and scaler only on training split.
# -----------------------------------------------------------------------------
numeric_medians = {}
for col in numeric_cols:
    med = tr_raw[col].median()
    if pd.isna(med):
        med = 0.0
    numeric_medians[col] = float(med)

for col in numeric_cols:
    tr_raw[col] = safe_float_series(tr_raw[col], numeric_medians[col])
    va_raw[col] = safe_float_series(va_raw[col], numeric_medians[col])
    te_raw[col] = safe_float_series(te_raw[col], numeric_medians[col])

# Log features for non-negative numeric columns.
for col in numeric_cols:
    if tr_raw[col].min() >= 0:
        tr_raw[f"{col}__log1p"] = np.log1p(tr_raw[col].clip(lower=0)).astype(np.float32)
        va_raw[f"{col}__log1p"] = np.log1p(va_raw[col].clip(lower=0)).astype(np.float32)
        te_raw[f"{col}__log1p"] = np.log1p(te_raw[col].clip(lower=0)).astype(np.float32)

# Robust scaling fitted on training split only.
if numeric_cols and len(tr_raw) > 0:
    scaler = RobustScaler()
    scaler.fit(tr_raw[numeric_cols].values)

    tr_scaled = scaler.transform(tr_raw[numeric_cols].values).astype(np.float32)
    va_scaled = scaler.transform(va_raw[numeric_cols].values).astype(np.float32)
    te_scaled = scaler.transform(te_raw[numeric_cols].values).astype(np.float32)

    for i, col in enumerate(numeric_cols):
        tr_raw[f"{col}__scaled"] = tr_scaled[:, i]
        va_raw[f"{col}__scaled"] = va_scaled[:, i]
        te_raw[f"{col}__scaled"] = te_scaled[:, i]

    tr_raw["numeric__mean"] = tr_raw[numeric_cols].mean(axis=1).astype(np.float32)
    tr_raw["numeric__std"] = (
        tr_raw[numeric_cols].std(axis=1).fillna(0).astype(np.float32)
    )
    tr_raw["numeric__max"] = tr_raw[numeric_cols].max(axis=1).astype(np.float32)
    tr_raw["numeric__min"] = tr_raw[numeric_cols].min(axis=1).astype(np.float32)
    tr_raw["numeric__range"] = (tr_raw["numeric__max"] - tr_raw["numeric__min"]).astype(
        np.float32
    )

    va_raw["numeric__mean"] = va_raw[numeric_cols].mean(axis=1).astype(np.float32)
    va_raw["numeric__std"] = (
        va_raw[numeric_cols].std(axis=1).fillna(0).astype(np.float32)
    )
    va_raw["numeric__max"] = va_raw[numeric_cols].max(axis=1).astype(np.float32)
    va_raw["numeric__min"] = va_raw[numeric_cols].min(axis=1).astype(np.float32)
    va_raw["numeric__range"] = (va_raw["numeric__max"] - va_raw["numeric__min"]).astype(
        np.float32
    )

    te_raw["numeric__mean"] = te_raw[numeric_cols].mean(axis=1).astype(np.float32)
    te_raw["numeric__std"] = (
        te_raw[numeric_cols].std(axis=1).fillna(0).astype(np.float32)
    )
    te_raw["numeric__max"] = te_raw[numeric_cols].max(axis=1).astype(np.float32)
    te_raw["numeric__min"] = te_raw[numeric_cols].min(axis=1).astype(np.float32)
    te_raw["numeric__range"] = (te_raw["numeric__max"] - te_raw["numeric__min"]).astype(
        np.float32
    )


# -----------------------------------------------------------------------------
# 9. Train-only group aggregations for useful categorical columns
# -----------------------------------------------------------------------------
group_cat_cols = []
for col in categorical_cols:
    nuniq = tr_raw[col].nunique(dropna=True)
    if 1 < nuniq <= 500:
        group_cat_cols.append(col)

group_cat_cols = group_cat_cols[:12]
group_num_cols = numeric_cols[:12]

group_stats = {}
for cat in group_cat_cols:
    for num in group_num_cols:
        agg = tr_raw.groupby(cat)[num].agg(["mean", "median", "count"])
        global_mean = tr_raw[num].mean()
        global_median = tr_raw[num].median()

        group_stats[(cat, num)] = {
            "mean": agg["mean"].fillna(global_mean),
            "median": agg["median"].fillna(global_median),
            "count": agg["count"].fillna(0.0),
        }

for cat in group_cat_cols:
    for num in group_num_cols:
        stats = group_stats[(cat, num)]
        for df in [tr_raw, va_raw, te_raw]:
            df[f"{cat}__{num}__mean"] = df[cat].map(stats["mean"]).astype(np.float32)
            df[f"{cat}__{num}__median"] = (
                df[cat].map(stats["median"]).astype(np.float32)
            )
            df[f"{cat}__{num}__count"] = df[cat].map(stats["count"]).astype(np.float32)

        tr_raw[f"{cat}__{num}__mean"] = (
            tr_raw[f"{cat}__{num}__mean"].fillna(tr_raw[num].mean()).astype(np.float32)
        )
        va_raw[f"{cat}__{num}__mean"] = (
            va_raw[f"{cat}__{num}__mean"].fillna(tr_raw[num].mean()).astype(np.float32)
        )
        te_raw[f"{cat}__{num}__mean"] = (
            te_raw[f"{cat}__{num}__mean"].fillna(tr_raw[num].mean()).astype(np.float32)
        )

        tr_raw[f"{cat}__{num}__median"] = (
            tr_raw[f"{cat}__{num}__median"]
            .fillna(tr_raw[num].median())
            .astype(np.float32)
        )
        va_raw[f"{cat}__{num}__median"] = (
            va_raw[f"{cat}__{num}__median"]
            .fillna(tr_raw[num].median())
            .astype(np.float32)
        )
        te_raw[f"{cat}__{num}__median"] = (
            te_raw[f"{cat}__{num}__median"]
            .fillna(tr_raw[num].median())
            .astype(np.float32)
        )


# -----------------------------------------------------------------------------
# 10. Categorical encoding using training-only mappings
# -----------------------------------------------------------------------------
for col in categorical_cols:
    if tr_raw[col].dtype == bool:
        tr_raw[f"{col}__code"] = tr_raw[col].astype(np.int8)
        va_raw[f"{col}__code"] = (
            va_raw[col].map({True: 1, False: 0}).fillna(0).astype(np.int8)
        )
        te_raw[f"{col}__code"] = (
            te_raw[col].map({True: 1, False: 0}).fillna(0).astype(np.int8)
        )
        tr_raw[f"{col}__freq"] = 0.5
        va_raw[f"{col}__freq"] = 0.5
        te_raw[f"{col}__freq"] = 0.5
    else:
        train_values = tr_raw[col].fillna("__MISSING__").astype(str)
        freq = train_values.value_counts()
        mapping = {value: idx for idx, value in enumerate(freq.index)}
        freq_map = {
            value: count / max(1, len(train_values)) for value, count in freq.items()
        }

        def encode_code(series: pd.Series) -> pd.Series:
            return (
                series.fillna("__MISSING__")
                .astype(str)
                .map(mapping)
                .fillna(-1)
                .astype(np.int32)
            )

        def encode_freq(series: pd.Series) -> pd.Series:
            return (
                series.fillna("__MISSING__")
                .astype(str)
                .map(freq_map)
                .fillna(0.0)
                .astype(np.float32)
            )

        tr_raw[f"{col}__code"] = encode_code(tr_raw[col])
        va_raw[f"{col}__code"] = encode_code(va_raw[col])
        te_raw[f"{col}__code"] = encode_code(te_raw[col])

        tr_raw[f"{col}__freq"] = encode_freq(tr_raw[col])
        va_raw[f"{col}__freq"] = encode_freq(va_raw[col])
        te_raw[f"{col}__freq"] = encode_freq(te_raw[col])

    tr_raw.drop(columns=[col], inplace=True, errors="ignore")
    va_raw.drop(columns=[col], inplace=True, errors="ignore")
    te_raw.drop(columns=[col], inplace=True, errors="ignore")


# -----------------------------------------------------------------------------
# 11. Final feature cleanup: remove constants and exact duplicate columns
# -----------------------------------------------------------------------------
def clean_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    constant_cols = [col for col in out.columns if is_constant_series(out[col])]
    out.drop(columns=constant_cols, inplace=True, errors="ignore")

    if len(out.columns) > 0:
        # Detect exact-duplicate columns by transposing so each original
        # column becomes a row; duplicated() only works along axis=0.
        transposed = out.T
        duplicate_mask = transposed.duplicated(keep="first")
        duplicate_cols = transposed.index[duplicate_mask].tolist()
        out.drop(columns=duplicate_cols, inplace=True, errors="ignore")

    return out


tr_feat = clean_feature_frame(tr_raw)
va_feat = clean_feature_frame(va_raw)
te_feat = clean_feature_frame(te_raw)

# Keep only columns present in the training feature matrix to guarantee parity.
final_feature_cols = list(tr_feat.columns)
va_feat = va_feat.reindex(columns=final_feature_cols)
te_feat = te_feat.reindex(columns=final_feature_cols)

# Replace any remaining NaN/Inf with 0 for stable downstream training.
tr_feat = tr_feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
va_feat = va_feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
te_feat = te_feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)

# Safety fallback for degenerate cases where no usable features remain.
if len(final_feature_cols) == 0:
    final_feature_cols = ["__dummy__"]
    tr_feat[final_feature_cols[0]] = 0.0
    va_feat[final_feature_cols[0]] = 0.0
    te_feat[final_feature_cols[0]] = 0.0


# -----------------------------------------------------------------------------
# 12. Model design: gated mixture-of-experts tabular MLP.
# Output semantics: model outputs log1p(amountUSD); inference maps back to USD.
# -----------------------------------------------------------------------------
code_cols = [c for c in final_feature_cols if str(c).endswith("__code")]
code_indices = []
code_dims = []
MAX_EMBED_CARDINALITY = 1000

if code_cols:
    for c in code_cols:
        vals = pd.to_numeric(tr_feat[c], errors="coerce").fillna(-1).astype("int64")
        max_val = int(vals.max()) if len(vals) > 0 else 0
        if max_val <= MAX_EMBED_CARDINALITY:
            code_indices.append(final_feature_cols.index(c))
            code_dims.append(max(max_val, 0) + 2)


class TabularGatedMixtureModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        code_indices,
        code_dims,
        hidden: int = 128,
        n_experts: int = 2,
        emb_dim: int = 8,
        dropout: float = 0.25,
    ):
        super().__init__()

        self.n_features = n_features
        self.hidden = hidden
        self.n_experts = n_experts
        self.emb_dim = emb_dim
        self.dropout_p = dropout

        self.input_norm = nn.LayerNorm(n_features)

        self.embeddings = nn.ModuleList(
            [nn.Embedding(dim, emb_dim) for dim in code_dims]
        )
        self.embed_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "code_idx",
            torch.tensor(code_indices, dtype=torch.long),
            persistent=False,
        )

        total_dim = n_features + len(code_dims) * emb_dim
        self.total_dim = total_dim

        self.gate = nn.Linear(total_dim, n_experts)
        with torch.no_grad():
            self.gate.bias.zero_()
            if n_experts > 1:
                self.gate.bias[0] = 1.0
                self.gate.bias[1] = -1.0
                if n_experts > 2:
                    self.gate.bias[2:] = 0.0

        self.experts = nn.ModuleList()
        mid_dim = max(16, hidden // 2)
        for _ in range(n_experts):
            self.experts.append(
                nn.Sequential(
                    nn.Linear(total_dim, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, mid_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mid_dim, 1),
                )
            )

        self.output_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        raw_x = x
        x = self.input_norm(raw_x)

        parts = [x]

        if self.code_idx.numel() > 0:
            code_vals = torch.nan_to_num(raw_x[:, self.code_idx], nan=0.0).long() + 1
            emb_parts = []
            for i, emb in enumerate(self.embeddings):
                vals = code_vals[:, i].clamp(0, emb.num_embeddings - 1)
                emb_parts.append(emb(vals))
            parts.append(self.embed_dropout(torch.cat(emb_parts, dim=1)))

        h = torch.cat(parts, dim=1)

        gate = F.softmax(self.gate(h), dim=1)
        expert_out = torch.cat([expert(h) for expert in self.experts], dim=1)

        out = (gate * expert_out).sum(dim=1) + self.output_bias
        return out.squeeze(-1)


HIDDEN = 128
N_EXPERTS = 2
EMB_DIM = 8
DROPOUT = 0.25
LR = 1e-3
WEIGHT_DECAY = 1e-2

n_features = max(1, len(final_feature_cols))

model = TabularGatedMixtureModel(
    n_features=n_features,
    code_indices=code_indices,
    code_dims=code_dims,
    hidden=HIDDEN,
    n_experts=N_EXPERTS,
    emb_dim=EMB_DIM,
    dropout=DROPOUT,
)


# -----------------------------------------------------------------------------
# 13. Training data tensors
# -----------------------------------------------------------------------------
y_tr = clean_target_series(tr[TARGET])
y_va = clean_target_series(va[TARGET])

X_tr = tr_feat[final_feature_cols].values.astype(np.float32)
X_va = va_feat[final_feature_cols].values.astype(np.float32)
X_te = te_feat[final_feature_cols].values.astype(np.float32)


def rmse_loss(log_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    log_pred = torch.clamp(log_pred, 0.0, 30.0)
    pred = torch.expm1(log_pred)
    return torch.sqrt(torch.mean((pred - target.float()) ** 2)) + 1e-8


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.999),
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10
)


# -----------------------------------------------------------------------------
# 14. DataLoaders
# -----------------------------------------------------------------------------
BATCH_SIZE = 4096
NUM_WORKERS = 2

tr_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
va_ds = TensorDataset(torch.tensor(X_va), torch.tensor(y_va))
te_ds = TensorDataset(torch.tensor(X_te))

tr_loader = DataLoader(
    tr_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS if len(tr_ds) > 0 else 0,
    pin_memory=False,
)
va_loader = DataLoader(
    va_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS if len(va_ds) > 0 else 0,
    pin_memory=False,
)
te_loader = DataLoader(
    te_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS if len(te_ds) > 0 else 0,
    pin_memory=False,
)


# -----------------------------------------------------------------------------
# 15. Training loop with early stopping on original-scale validation RMSE
# -----------------------------------------------------------------------------
MAX_EPOCHS = 300
PATIENCE = 30
best_val_rmse = float("inf")
best_state = None
patience_counter = 0

for epoch in range(MAX_EPOCHS):
    model.train()
    train_loss = 0.0
    n_batches = 0

    for xb, yb in tr_loader:
        optimizer.zero_grad()
        log_pred = model(xb)
        loss = rmse_loss(log_pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        train_loss += loss.item()
        n_batches += 1

    avg_train = train_loss / max(1, n_batches)

    model.eval()
    with torch.no_grad():
        va_preds = []
        for xb, yb in va_loader:
            log_p = model(xb)
            va_preds.append(torch.expm1(torch.clamp(log_p, 0.0, 30.0)).numpy())

    if va_preds:
        va_preds = np.concatenate(va_preds)
        va_preds = np.nan_to_num(va_preds, nan=0.0, posinf=0.0, neginf=0.0)
        val_rmse = float(
            np.sqrt(
                np.mean((va_preds.astype(np.float64) - y_va.astype(np.float64)) ** 2)
            )
        )
    else:
        val_rmse = float("inf")

    if not np.isfinite(val_rmse):
        val_rmse = float("inf")

    scheduler.step(val_rmse)
    print(
        f"Epoch {epoch+1}/{MAX_EPOCHS} | train_rmse={avg_train:.4f} | val_rmse={val_rmse:.4f}"
    )

    if val_rmse < best_val_rmse or best_state is None:
        best_val_rmse = val_rmse
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            break


# -----------------------------------------------------------------------------
# 16. Load best model and run inference with identical val/test logic
# -----------------------------------------------------------------------------
if best_state is None:
    best_state = model.state_dict()

model.load_state_dict(best_state)
model.eval()


def predict_amounts(loader: DataLoader) -> np.ndarray:
    preds = []
    with torch.no_grad():
        for batch in loader:
            xb = batch[0]
            log_p = model(xb)
            preds.append(torch.expm1(torch.clamp(log_p, 0.0, 30.0)).numpy())

    if not preds:
        return np.array([], dtype=np.float32)

    out = np.concatenate(preds)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


val_preds = predict_amounts(va_loader)
if len(val_preds) > 0:
    val_preds = np.clip(val_preds, 0.0, None)
    score = float(
        np.sqrt(np.mean((val_preds.astype(np.float64) - y_va.astype(np.float64)) ** 2))
    )
else:
    score = 0.0

test_preds = predict_amounts(te_loader)
test_preds = np.clip(test_preds, 0.0, None)
test_preds = np.nan_to_num(test_preds, nan=0.0, posinf=0.0, neginf=0.0)

# -----------------------------------------------------------------------------
# 17. Write submission
# -----------------------------------------------------------------------------
sub = pd.DataFrame(
    {
        "id": np.arange(len(test_preds), dtype=np.int64),
        "amountUSD": test_preds.astype(np.float64),
    }
)
sub.to_csv(os.path.join(SUBMISSION_DIR, "submission.csv"), index=False)

print(f"Final Validation Score: {score}")
