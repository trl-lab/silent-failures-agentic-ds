import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
TARGET = "APPETENCY"
IN, WORK, SUB = Path("./input"), Path("./working"), Path("./submission")
WORK.mkdir(parents=True, exist_ok=True)
SUB.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# Step 1: data loading, split first, then leakage-safe feature engineering
# ------------------------------------------------------------------
train = pd.read_parquet(IN / "train.parquet")
test = pd.read_parquet(IN / "test.parquet")

feat_order = [c for c in train.columns if c != TARGET]
missing_in_test = [c for c in feat_order if c not in test.columns]
assert not missing_in_test, f"test missing columns: {missing_in_test[:5]}"


def _is_cat(s):
    return (
        pd.api.types.is_object_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
        or s.dtype.kind in "OUS"
    )


detected_cats = [c for c in feat_order if _is_cat(train[c])]
if len(detected_cats) == 40:
    cat_cols = detected_cats
else:
    cat_cols = feat_order[-40:]
num_cols = [c for c in feat_order if c not in set(cat_cols)]

y = train[TARGET].astype(float)
y01 = (y > 0).astype(int).values
assert len(np.unique(y01)) == 2

# Split first, then fit all statistics on the train split only.
idx_tr, idx_va = train_test_split(
    np.arange(len(train)), test_size=0.2, stratify=y01, random_state=SEED
)
tr, va = train.iloc[idx_tr], train.iloc[idx_va]
y_tr, y_va = y01[idx_tr], y01[idx_va]


def to_num(df, cols):
    return df[cols].apply(lambda s: pd.to_numeric(s, errors="coerce")).astype(float)


M_tr = to_num(tr, num_cols)

# Winsorization bounds fitted on train split only.
q01, q99 = M_tr.quantile(0.001), M_tr.quantile(0.999)
lo = np.where(q01.values < q99.values, q01.values, -np.inf)
hi = np.where(q99.values > q01.values, q99.values, np.inf)
lo_s, hi_s = pd.Series(lo, index=num_cols), pd.Series(hi, index=num_cols)


def clip_num(df, cols):
    return to_num(df, cols).clip(lower=lo_s, upper=hi_s, axis=1)


X_tr = clip_num(tr, num_cols)

# Drop constant / all-NaN numeric columns.
std_tr = X_tr.std()
dead = [
    c
    for c in num_cols
    if (not np.isfinite(std_tr[c])) or std_tr[c] == 0 or X_tr[c].isna().all()
]
num_used = [c for c in num_cols if c not in set(dead)]

miss_cols = [c for c in num_used if X_tr[c].isna().mean() > 0.05]

# Log1p-shift for skewed numeric columns, parameters from train split only.
skew_params = {}
for c in num_used:
    s = X_tr[c].dropna()
    if len(s) < 100:
        continue
    if abs(float(s.skew())) > 1.0:
        mn = float(s.min())
        skew_params[c] = None if mn >= 0 else -mn


def apply_log(X):
    for c, sh in skew_params.items():
        v = X[c]
        X[c] = (
            np.log1p(v.clip(lower=0))
            if sh is None
            else np.log1p((v - sh).clip(lower=0))
        )
    return X


# Top-k numerics by |corr| with target, fitted on train split only.
corr = {}
for c in num_used:
    a, m = X_tr[c].values, np.isfinite(X_tr[c].values)
    if m.sum() > 10:
        r = np.corrcoef(a[m], y_tr[m])[0, 1]
        corr[c] = abs(float(r)) if np.isfinite(r) else 0.0
    else:
        corr[c] = 0.0
topk = sorted(num_used, key=lambda c: corr[c], reverse=True)[:10]

# Categorical encoders: frequency + smoothed target encoding, fit on train split only.
gm = float(y_tr.mean())
SMOOTH = 50.0
cat_feats = {}
for c in cat_cols:
    s = tr[c].astype(object)
    freq = s.value_counts(normalize=True)
    tmp = pd.Series(y_tr, index=s.index)
    mean_c, cnt_c = tmp.groupby(s.values).mean(), tmp.groupby(s.values).count()
    te = (cnt_c * mean_c + SMOOTH * gm) / (cnt_c + SMOOTH)
    known = set(s.dropna().unique()) | {"__NA__"}
    cat_feats[c] = (freq, te, known)


def apply_cats(df):
    out = {}
    for c, (freq, te, known) in cat_feats.items():
        s = df[c].astype(object)
        s = s.where(s.isin(known), "__NA__")
        out[c] = s.astype("category")
        out[c + "__freq"] = s.map(freq).fillna(0.0).astype(float).values
        out[c + "__te"] = s.map(te).fillna(gm).astype(float).values
    return pd.DataFrame(out, index=df.index)


def block_feats(X):
    n = X[num_used]
    return pd.DataFrame(
        {
            "num_mean": n.mean(axis=1).values,
            "num_std": n.std(axis=1).values,
            "num_max": n.max(axis=1).values,
            "num_min": n.min(axis=1).values,
            "num_range": (n.max(axis=1) - n.min(axis=1)).values,
            "num_nzero": (n == 0).sum(axis=1).astype(float).values,
            "num_nmiss": n.isna().sum(axis=1).astype(float).values,
        },
        index=X.index,
    )


def extra_feats(X, df):
    out = {
        c
        + "__miss": np.isnan(pd.to_numeric(df[c], errors="coerce").values).astype(float)
        for c in miss_cols
    }
    tk = X[topk]
    out["topk_mean"], out["topk_max"] = tk.mean(axis=1).values, tk.max(axis=1).values
    for i in range(len(topk)):
        for j in range(i + 1, len(topk)):
            a, b = topk[i], topk[j]
            out[f"{a}__x__{b}"] = X[a].values * X[b].values
    return pd.DataFrame(out, index=df.index)


def process(df):
    X = clip_num(df, num_cols)[num_used]
    X = apply_log(X)
    feats = pd.concat([X, block_feats(X), extra_feats(X, df), apply_cats(df)], axis=1)
    num_like = [
        c
        for c in feats.columns
        if not isinstance(feats[c].dtype, pd.CategoricalDtype)
    ]
    feats[num_like] = feats[num_like].astype(np.float32)
    return feats


F_tr, F_va, F_te = process(tr), process(va), process(test)
features = F_tr.columns.tolist()

d_tr, d_va = F_tr.copy(), F_va.copy()
d_tr[TARGET], d_va[TARGET] = y_tr, y_va
d_tr.to_parquet(WORK / "train.parquet")
d_va.to_parquet(WORK / "val.parquet")
F_te.to_parquet(WORK / "test.parquet")
with open(WORK / "fe_meta.json", "w") as f:
    json.dump(
        {
            "features": features,
            "num_cols": num_used,
            "cat_cols": cat_cols,
            "topk": topk,
            "miss_cols": miss_cols,
            "seed": SEED,
        },
        f,
    )

# ------------------------------------------------------------------
# Step 2: load processed frames and train the DeepFM model
# ------------------------------------------------------------------
tr = pd.read_parquet(WORK / "train.parquet")
va = pd.read_parquet(WORK / "val.parquet")
te = pd.read_parquet(WORK / "test.parquet")

try:
    meta = json.loads((WORK / "fe_meta.json").read_text())
except Exception:
    meta = {}

cat_cols = [c for c in meta.get("cat_cols", []) if c in tr.columns]
if not cat_cols:
    cat_cols = [
        c
        for c in tr.columns
        if c != TARGET
        and (isinstance(tr[c].dtype, pd.CategoricalDtype) or tr[c].dtype.kind in "OUS")
    ]
num_cols = [c for c in tr.columns if c != TARGET and c not in set(cat_cols)]

y_tr = (tr[TARGET].astype(float).values > 0).astype(float)
y_va = (va[TARGET].astype(float).values > 0).astype(float)
assert len(np.unique(y_tr)) == 2 and len(np.unique(y_va)) == 2

# Input adaptation fitted only on the train split.
if len(num_cols) > 0:
    X_tr_raw = (
        tr[num_cols].apply(pd.to_numeric, errors="coerce").astype(float).values
    )
    finite = np.isfinite(X_tr_raw)
    count = finite.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        safe = np.where(finite, X_tr_raw, 0.0)
        sumv = safe.sum(axis=0)
        mean = np.divide(
            sumv, count, out=np.zeros_like(sumv, dtype=float), where=count > 0
        )
        sumsq = (safe * safe).sum(axis=0)
        var = np.divide(
            sumsq, count, out=np.zeros_like(sumsq, dtype=float), where=count > 0
        ) - mean * mean
        var = np.where(count > 1, np.maximum(var, 0.0), 0.0)
        std = np.sqrt(var)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
else:
    mean = np.array([], dtype=float)
    std = np.array([], dtype=float)


def standardize(df):
    if len(num_cols) == 0:
        return np.zeros((len(df), 0), dtype=np.float32)
    X = df[num_cols].apply(pd.to_numeric, errors="coerce").astype(float).values
    X = (X - mean) / std
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


X_tr = standardize(tr)
X_va = standardize(va)
X_te = standardize(te)

cat_maps = {}
cat_cardinalities = []
for c in cat_cols:
    vals = tr[c].astype(str).unique()
    mapping = {v: i + 1 for i, v in enumerate(sorted(vals))}
    cat_maps[c] = mapping
    cat_cardinalities.append(len(mapping) + 1)


def encode_cats(df):
    if not cat_cols:
        return np.zeros((len(df), 0), dtype=np.int64)
    arrs = []
    for c in cat_cols:
        s = df[c].astype(str)
        arrs.append(s.map(cat_maps[c]).fillna(0).astype(np.int64).values)
    return np.column_stack(arrs)


C_tr = encode_cats(tr)
C_va = encode_cats(va)
C_te = encode_cats(te)


class DeepFM(nn.Module):
    """
    DeepFM-style tabular model:
    - numeric features become embeddings via learned per-feature vectors
    - categorical features become embeddings via learned category vectors
    - explicit FM branch captures second-order interactions
    - MLP branch captures higher-order interactions
    - direct numeric linear tower preserves strong first-order numeric signal
    """

    def __init__(
        self, num_numeric, cat_cardinalities, emb_dim=4, hidden=(128, 64), dropout=0.35
    ):
        super().__init__()
        self.emb_dim = emb_dim

        self.num_weights = nn.Parameter(torch.randn(num_numeric, emb_dim) * 0.1)
        self.num_linear = nn.Linear(num_numeric, 1)

        self.cat_embs = nn.ModuleList(
            [nn.Embedding(card, emb_dim, padding_idx=0) for card in cat_cardinalities]
        )

        n_feat = num_numeric + len(cat_cardinalities)
        self.mlp = nn.Sequential(
            nn.Linear(n_feat * emb_dim, hidden[0]),
            nn.BatchNorm1d(hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.BatchNorm1d(hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.linear = nn.Linear(emb_dim, 1)
        self.head = nn.Linear(hidden[1] + emb_dim + 2, 1)

    def forward(self, x_num, x_cat):
        b = x_num.size(0)

        num_emb = x_num.unsqueeze(-1) * self.num_weights.unsqueeze(0)
        cat_embs = [emb(x_cat[:, i]).unsqueeze(1) for i, emb in enumerate(self.cat_embs)]
        all_emb = torch.cat([num_emb] + cat_embs, dim=1)

        sum_emb = all_emb.sum(dim=1)
        sum_emb_sq = (all_emb**2).sum(dim=1)
        fm = 0.5 * (sum_emb**2 - sum_emb_sq)

        mlp = self.mlp(all_emb.view(b, -1))

        lin = self.linear(all_emb.mean(dim=1))
        num_lin = self.num_linear(x_num)

        return self.head(torch.cat([mlp, fm, lin, num_lin], dim=1)).squeeze(-1)


class FocalBCELoss(nn.Module):
    """Class-aware focal binary cross-entropy for imbalanced tabular classification."""

    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * targets + (1.0 - p) * (1.0 - targets)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        return (alpha_t * (1.0 - pt) ** self.gamma * bce).mean()


pos_rate = float(y_tr.mean())
pos_weight = min((1.0 - pos_rate) / pos_rate, 5.0) if pos_rate > 0 else 1.0
alpha = pos_weight / (1.0 + pos_weight)

model = DeepFM(
    num_numeric=len(num_cols),
    cat_cardinalities=cat_cardinalities,
    emb_dim=4,
    hidden=(128, 64),
    dropout=0.35,
).to(device)

criterion = FocalBCELoss(alpha=alpha, gamma=2.0).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

train_ds = TensorDataset(
    torch.tensor(X_tr, dtype=torch.float32),
    torch.tensor(C_tr, dtype=torch.long),
    torch.tensor(y_tr, dtype=torch.float32),
)
train_loader = DataLoader(
    train_ds,
    batch_size=4096,
    shuffle=True,
    num_workers=2,
    pin_memory=(device.type == "cuda"),
)

val_X = torch.tensor(X_va, dtype=torch.float32).to(device)
val_C = torch.tensor(C_va, dtype=torch.long).to(device)
test_X = torch.tensor(X_te, dtype=torch.float32).to(device)
test_C = torch.tensor(C_te, dtype=torch.long).to(device)


def predict_proba(model, X, C, batch_size=8192):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, X.size(0), batch_size):
            logits = model(X[i : i + batch_size], C[i : i + batch_size])
            outs.append(torch.sigmoid(logits).cpu().numpy())
    if not outs:
        return np.array([], dtype=float)
    return np.nan_to_num(np.concatenate(outs), nan=0.5, posinf=1.0, neginf=0.0)


max_epochs = 50
patience = 8
best_auc = -1.0
best_state = None
no_improve = 0
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max_epochs, eta_min=1e-5
)

for epoch in range(1, max_epochs + 1):
    model.train()
    epoch_loss = 0.0
    n_batches = 0

    for x_num, x_cat, y in train_loader:
        x_num = x_num.to(device, non_blocking=True)
        x_cat = x_cat.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_num, x_cat)
        loss = criterion(logits, y)

        if not torch.isfinite(loss).item():
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        epoch_loss += float(loss.item())
        n_batches += 1

    val_proba = predict_proba(model, val_X, val_C)
    val_auc = float(roc_auc_score(y_va, val_proba))

    print(
        f"Epoch {epoch:02d} | train_loss {epoch_loss / max(n_batches, 1):.4f} | val_auc {val_auc:.4f}"
    )

    if np.isfinite(val_auc) and val_auc > best_auc:
        best_auc = val_auc
        best_state = model.state_dict().copy()
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            break

    scheduler.step()

if best_state is not None:
    model.load_state_dict(best_state)

test_proba = predict_proba(model, test_X, test_C)
test_proba = np.clip(np.nan_to_num(test_proba, nan=0.5), 1e-7, 1.0 - 1e-7)
pd.DataFrame({"id": np.arange(len(test_proba)), TARGET: test_proba}).to_csv(
    SUB / "submission.csv", index=False
)

if best_state is not None and np.isfinite(best_auc) and best_auc >= 0.0:
    score = best_auc
else:
    score = float(roc_auc_score(y_va, predict_proba(model, val_X, val_C)))

if not np.isfinite(score):
    score = 0.5

print(f"Final Validation Score: {score}")
