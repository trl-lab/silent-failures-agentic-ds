import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

TARGET = "Estimated Percentage of Patients with Controlled Blood Pressure"
INPUT_DIR = Path("./input")
PROC = Path("./working/processed")
PROC.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Data processing and feature engineering
# ---------------------------------------------------------------------------
train = pd.read_parquet(INPUT_DIR / "train.parquet")
test = pd.read_parquet(INPUT_DIR / "test.parquet")

train = train.copy()
test = test.copy()

train["id"] = np.arange(len(train), dtype=np.int64)
test["id"] = np.arange(len(test), dtype=np.int64)

train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
train = train.dropna(subset=[TARGET]).reset_index(drop=True)

feature_candidates = [
    c for c in train.columns if c not in ("id", TARGET) and c in test.columns
]

features = []
for f in feature_candidates:
    s = train[f]
    if s.nunique(dropna=True) <= 1:
        continue

    if (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_categorical_dtype(s)
        or pd.api.types.is_string_dtype(s)
    ):
        train[f] = train[f].astype("string").str.strip()
        test[f] = test[f].astype("string").str.strip()

    features.append(f)

if not features:
    train["__constant__"] = 1.0
    test["__constant__"] = 1.0
    features = ["__constant__"]

if len(train) > 1:
    stratify = None
    y_tmp = train[TARGET]
    if y_tmp.nunique(dropna=True) == 2:
        stratify = y_tmp

    try:
        train_part, valid_part = train_test_split(
            train, test_size=0.2, random_state=SEED, stratify=stratify
        )
    except Exception:
        train_part, valid_part = train_test_split(
            train, test_size=0.2, random_state=SEED
        )
else:
    train_part = train.copy()
    valid_part = train.iloc[:0].copy()

train_part = train_part.reset_index(drop=True)
valid_part = valid_part.reset_index(drop=True)

cols = ["id"] + features + [TARGET]
train_out = train_part[cols].copy()
valid_out = valid_part[cols].copy()
test_out = test[["id"] + features].copy()

train_out.to_parquet(PROC / "train.parquet", index=False)
valid_out.to_parquet(PROC / "valid.parquet", index=False)
test_out.to_parquet(PROC / "test.parquet", index=False)

metadata = {
    "features": features,
    "target": TARGET,
    "id": "id",
    "train_shape": list(train_out.shape),
    "valid_shape": list(valid_out.shape),
    "test_shape": list(test_out.shape),
}
with open(PROC / "metadata.json", "w") as fh:
    json.dump(metadata, fh, indent=2)

# ---------------------------------------------------------------------------
# 2. Model design
# ---------------------------------------------------------------------------
with open(PROC / "metadata.json", "r") as fh:
    metadata = json.load(fh)

TARGET = metadata.get("target", TARGET)
features = metadata.get("features", features)
train_df = pd.read_parquet(PROC / "train.parquet")


def _is_categorical(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return 1 < series.nunique(dropna=True) <= 20
    return True


cat_features = []
num_features = []
for f in features:
    if f not in train_df.columns:
        continue
    if _is_categorical(train_df[f]):
        cat_features.append(f)
    else:
        num_features.append(f)


def _canonical(x):
    try:
        if pd.isna(x):
            return "nan"
    except Exception:
        pass

    if isinstance(x, (float, np.floating)):
        x = float(x)
        if np.isfinite(x) and x.is_integer():
            return str(int(x))
        return repr(x)

    if isinstance(x, (int, np.integer)):
        return str(int(x))

    return str(x)


vocabs = {}
for f in cat_features:
    unique_vals = pd.unique(train_df[f].dropna())
    vocabs[f] = {_canonical(v): i + 1 for i, v in enumerate(unique_vals)}


class TabularFeatureEncoder:
    def __init__(self, cat_features, num_features, vocabs, num_mean=None, num_std=None):
        self.cat_features = list(cat_features)
        self.num_features = list(num_features)
        self.vocabs = vocabs
        self.num_mean = num_mean
        self.num_std = num_std

    def transform(self, df):
        n = len(df)

        if self.cat_features:
            x_cat = torch.zeros(n, len(self.cat_features), dtype=torch.long)
            for j, f in enumerate(self.cat_features):
                vocab = self.vocabs[f]
                mapped = df[f].map(
                    lambda x: 0 if pd.isna(x) else vocab.get(_canonical(x), 0)
                )
                x_cat[:, j] = torch.tensor(
                    mapped.to_numpy(dtype=np.int64), dtype=torch.long
                )
        else:
            x_cat = torch.zeros(n, 0, dtype=torch.long)

        if self.num_features:
            x_num = (
                df[self.num_features]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=np.float32)
            )
        else:
            x_num = np.zeros((n, 0), dtype=np.float32)

        x_num = np.nan_to_num(x_num, nan=0.0, posinf=0.0, neginf=0.0)
        if self.num_mean is not None and self.num_std is not None:
            x_num = (x_num - self.num_mean) / (self.num_std + 1e-8)
        return x_cat, torch.from_numpy(x_num)


class TabularFMNet(nn.Module):
    def __init__(
        self,
        n_cat,
        cat_vocab_sizes,
        n_num,
        field_dim=8,
        hidden=64,
        dropout=0.15,
    ):
        super().__init__()
        self.n_cat = n_cat
        self.n_num = n_num
        self.field_dim = field_dim
        self.hidden = hidden
        self.n_fields = n_cat + n_num

        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(v + 1, field_dim, padding_idx=0) for v in cat_vocab_sizes]
        )

        self.num_proj = nn.Linear(n_num, n_num * field_dim) if n_num > 0 else None

        self.field_embedding = (
            nn.Embedding(self.n_fields, field_dim) if self.n_fields > 0 else None
        )

        self.dropout = nn.Dropout(dropout)

        if self.n_fields > 0:
            self.head = nn.Sequential(
                nn.Linear(field_dim * 2, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
        else:
            self.head = nn.Linear(1, 1)

    def forward(self, x_cat, x_num):
        if x_cat is None:
            x_cat = torch.empty(
                x_num.shape[0], 0, dtype=torch.long, device=x_num.device
            )
        if x_num is None:
            x_num = torch.empty(
                x_cat.shape[0], 0, dtype=torch.float32, device=x_cat.device
            )

        if self.n_fields == 0:
            return torch.zeros(x_cat.shape[0], device=x_cat.device)

        fields = []
        for j, emb in enumerate(self.cat_embeddings):
            fields.append(emb(x_cat[:, j]).unsqueeze(1))

        if self.num_proj is not None:
            fields.append(
                self.num_proj(x_num).view(x_num.shape[0], self.n_num, self.field_dim)
            )

        x = torch.cat(fields, dim=1)
        x = x + self.field_embedding(
            torch.arange(self.n_fields, device=x.device)
        ).unsqueeze(0)

        main_effect = x.mean(dim=1)
        interaction = 0.5 * (x.sum(dim=1).pow(2) - x.pow(2).sum(dim=1))

        z = torch.cat([main_effect, interaction], dim=1)
        z = self.dropout(z)
        return self.head(z).squeeze(-1)


class AUCRankingLoss(nn.Module):
    def __init__(self, bce_weight=0.25, max_pairs=65536, min_pair_diff=1e-8):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.max_pairs = int(max_pairs)
        self.min_pair_diff = float(min_pair_diff)

    def _pairwise_loss(self, logits, y):
        n = y.numel()
        if n < 2:
            return torch.zeros((), device=logits.device)

        if n <= 256:
            dy = y.unsqueeze(1) - y.unsqueeze(0)
            mask = dy.abs() > self.min_pair_diff
            if not mask.any():
                return torch.zeros((), device=logits.device)

            ds = logits.unsqueeze(1) - logits.unsqueeze(0)
            delta = torch.sign(dy) * ds
            return F.softplus(-delta)[mask].mean()

        i = torch.randint(0, n, (self.max_pairs,), device=logits.device)
        j = torch.randint(0, n, (self.max_pairs,), device=logits.device)

        yi, yj = y[i], y[j]
        dy = yi - yj
        mask = dy.abs() > self.min_pair_diff
        if not mask.any():
            return torch.zeros((), device=logits.device)

        ds = logits[i] - logits[j]
        delta = torch.sign(dy) * ds
        return F.softplus(-delta)[mask].mean()

    def forward(self, logits, y):
        y = y.float()
        valid = torch.isfinite(y)
        if not valid.all():
            logits = logits[valid]
            y = y[valid]

        if y.numel() == 0:
            return torch.zeros((), device=logits.device)

        uniq = y.unique()
        if uniq.numel() <= 2:
            y_bin = (y == uniq.max()).float()
            bce = F.binary_cross_entropy_with_logits(logits, y_bin)
            pair = self._pairwise_loss(logits, y_bin)
            return self.bce_weight * bce + (1.0 - self.bce_weight) * pair

        return self._pairwise_loss(logits, y)


def _param_groups(model):
    decay = []
    no_decay = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)

    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": 1e-2})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    m = TabularFMNet(
        n_cat=len(cat_features),
        cat_vocab_sizes=[len(vocabs[f]) for f in cat_features],
        n_num=len(num_features),
        field_dim=8,
        hidden=64,
        dropout=0.15,
    ).to(device)

    opt = AdamW(
        _param_groups(m),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    return m, opt


model, optimizer = build_model()
criterion = AUCRankingLoss()
if num_features:
    _num_df = train_df[num_features].apply(pd.to_numeric, errors="coerce")
    num_mean = _num_df.mean().fillna(0.0).to_numpy(dtype=np.float32)
    num_std = _num_df.std().fillna(1.0).to_numpy(dtype=np.float32)
else:
    num_mean = None
    num_std = None
encoder = TabularFeatureEncoder(cat_features, num_features, vocabs, num_mean, num_std)

out_dir = Path("./working/model_design")
out_dir.mkdir(parents=True, exist_ok=True)

config = {
    "model": "TabularFMNet",
    "cat_features": cat_features,
    "num_features": num_features,
    "field_dim": model.field_dim,
    "hidden": model.hidden,
    "n_fields": model.n_fields,
    "loss": "AUCRankingLoss",
    "optimizer": "AdamW",
    "lr": 3e-3,
    "weight_decay": 1e-2,
    "device": str(device),
}
with open(out_dir / "config.json", "w") as fh:
    json.dump(config, fh, indent=2)

with open(out_dir / "vocabs.json", "w") as fh:
    json.dump(vocabs, fh, indent=2)


# ---------------------------------------------------------------------------
# 3. Training and evaluation
# ---------------------------------------------------------------------------
def _finite_target_mask(df):
    if TARGET not in df.columns:
        return np.ones(len(df), dtype=bool)
    y = pd.to_numeric(df[TARGET], errors="coerce")
    return np.isfinite(y.to_numpy(dtype=float))


train_df = train_out[_finite_target_mask(train_out)].reset_index(drop=True)
valid_df = valid_out[_finite_target_mask(valid_out)].reset_index(drop=True)
test_df = test_out.copy()

if len(valid_df) == 0 and len(train_df) > 1:
    stratify = None
    y_tmp = train_df[TARGET]
    if y_tmp.nunique(dropna=True) == 2:
        stratify = y_tmp

    try:
        train_df, valid_df = train_test_split(
            train_df, test_size=0.2, random_state=SEED, stratify=stratify
        )
    except Exception:
        train_df, valid_df = train_test_split(
            train_df, test_size=0.2, random_state=SEED
        )

train_x_cat, train_x_num = encoder.transform(train_df)
valid_x_cat, valid_x_num = encoder.transform(valid_df)
test_x_cat, test_x_num = encoder.transform(test_df)

train_y = pd.to_numeric(train_df[TARGET], errors="coerce").to_numpy(dtype=np.float32)
valid_y = pd.to_numeric(valid_df[TARGET], errors="coerce").to_numpy(dtype=np.float32)

finite_train = np.isfinite(train_y)
train_x_cat = train_x_cat[torch.from_numpy(finite_train)]
train_x_num = train_x_num[torch.from_numpy(finite_train)]
train_y = train_y[finite_train]

finite_valid = np.isfinite(valid_y)
valid_x_cat = valid_x_cat[torch.from_numpy(finite_valid)]
valid_x_num = valid_x_num[torch.from_numpy(finite_valid)]
valid_y = valid_y[finite_valid]

train_target_scale = 1.0
if train_y.size > 0:
    train_target_scale = 100.0 if float(np.max(train_y)) > 1.5 else 1.0


def _postprocess(logits):
    vals = np.asarray(logits, dtype=np.float64)
    vals = np.nan_to_num(vals, nan=0.0, posinf=500.0, neginf=-500.0)
    vals = 1.0 / (1.0 + np.exp(-vals))
    if train_target_scale > 1.5:
        vals = vals * 100.0
    return vals.astype(np.float32)


def _fenwick_add(bit, i, n):
    i += 1
    while i <= n:
        bit[i] += 1
        i += i & -i


def _fenwick_sum(bit, i):
    s = 0
    while i > 0:
        s += bit[i]
        i -= i & -i
    return s


def _concordance_auc(y_true, y_score):
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(y_score, dtype=np.float64)

    mask = np.isfinite(y) & np.isfinite(s)
    y = y[mask]
    s = s[mask]

    if y.size < 2:
        return 0.5

    order = np.argsort(y, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]

    unique_scores, inverse = np.unique(s_sorted, return_inverse=True)
    m = len(unique_scores)

    bit = np.zeros(m + 2, dtype=np.int64)
    freq = np.zeros(m + 2, dtype=np.int64)

    concordant = 0.0
    denominator = 0
    i = 0
    n = y_sorted.size

    while i < n:
        j = i + 1
        while j < n and y_sorted[j] == y_sorted[i]:
            j += 1

        for k in range(i, j):
            r = int(inverse[k]) + 1
            less = _fenwick_sum(bit, r - 1)
            equal = int(freq[r])
            concordant += less + 0.5 * equal

        denominator += (j - i) * i

        for k in range(i, j):
            r = int(inverse[k]) + 1
            freq[r] += 1
            _fenwick_add(bit, r, m + 1)

        i = j

    if denominator == 0:
        return 0.5

    return float(concordant / denominator)


def official_auc(y_true, y_score):
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(y_score, dtype=np.float64)

    mask = np.isfinite(y) & np.isfinite(s)
    y = y[mask]
    s = s[mask]

    if y.size < 2:
        return 0.5

    uniq = np.unique(np.round(y, 12))

    if uniq.size == 1:
        return 0.5

    if uniq.size == 2:
        y_bin = (y == uniq[1]).astype(np.int8)
        return float(roc_auc_score(y_bin, s))

    return _concordance_auc(y, s)


def predict_from_tensors(model, x_cat, x_num, batch_size=1024):
    if len(x_cat) == 0:
        return np.empty((0,), dtype=np.float32)

    model.eval()
    ds = TensorDataset(x_cat, x_num)

    try:
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    except Exception:
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    preds = []
    with torch.no_grad():
        for xc, xn in dl:
            xc = xc.to(device, non_blocking=True)
            xn = xn.to(device, non_blocking=True)
            out = model(xc, xn)
            preds.append(out.detach().cpu().numpy())

    return np.concatenate(preds, axis=0) if preds else np.empty((0,), dtype=np.float32)


def _make_loader(x_cat, x_num, y, shuffle=True, batch_size=256):
    ds = TensorDataset(x_cat, x_num, torch.from_numpy(y.astype(np.float32)))
    try:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            drop_last=False,
        )
    except Exception:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
        )


def _validate(model):
    if valid_y.size == 0:
        return 0.5

    val_logits = predict_from_tensors(model, valid_x_cat, valid_x_num)
    val_preds = _postprocess(val_logits)
    score = official_auc(valid_y, val_preds)

    if not np.isfinite(score):
        return 0.5

    return float(score)


def train_with_validation(
    model,
    criterion,
    optimizer,
    x_cat,
    x_num,
    y,
    max_epochs=200,
    patience=20,
    min_delta=1e-4,
    batch_size=256,
):
    if y.size < 2:
        return None, 0.5, 1

    model.to(device)
    loader = _make_loader(x_cat, x_num, y, shuffle=True, batch_size=batch_size)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.6,
        patience=5,
        min_lr=1e-5,
    )

    best_score = -np.inf
    best_state = None
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for xc, xn, yy in loader:
            xc = xc.to(device, non_blocking=True)
            xn = xn.to(device, non_blocking=True)
            yy = yy.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(xc, xn)
            loss = criterion(logits, yy)

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += float(loss.item())
                n_batches += 1

        train_loss = total_loss / max(1, n_batches)
        val_score = _validate(model)

        print(
            f"Epoch {epoch:03d} | train_loss {train_loss:.6f} | val_auc {val_score:.6f}"
        )

        scheduler.step(val_score)

        if val_score > best_score + min_delta:
            best_score = val_score
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    return best_state, best_score, best_epoch


def train_fixed(
    model,
    criterion,
    optimizer,
    x_cat,
    x_num,
    y,
    epochs,
    batch_size=256,
):
    if epochs <= 0 or y.size < 2:
        return

    model.to(device)
    loader = _make_loader(x_cat, x_num, y, shuffle=True, batch_size=batch_size)

    for _ in range(1, int(epochs) + 1):
        model.train()
        for xc, xn, yy in loader:
            xc = xc.to(device, non_blocking=True)
            xn = xn.to(device, non_blocking=True)
            yy = yy.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(xc, xn)
            loss = criterion(logits, yy)

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()


best_state, best_score, best_epoch = train_with_validation(
    model,
    criterion,
    optimizer,
    train_x_cat,
    train_x_num,
    train_y,
    max_epochs=200,
    patience=20,
)

if best_state is None:
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_score = _validate(model)
    best_epoch = 1

model.load_state_dict(best_state)
final_val_score = _validate(model)
if not np.isfinite(final_val_score):
    final_val_score = 0.5

full_df = pd.concat([train_df, valid_df], axis=0, ignore_index=True)
full_df = full_df[_finite_target_mask(full_df)].reset_index(drop=True)

full_x_cat, full_x_num = encoder.transform(full_df)
full_y = pd.to_numeric(full_df[TARGET], errors="coerce").to_numpy(dtype=np.float32)

finite_full = np.isfinite(full_y)
full_x_cat = full_x_cat[torch.from_numpy(finite_full)]
full_x_num = full_x_num[torch.from_numpy(finite_full)]
full_y = full_y[finite_full]

torch.manual_seed(SEED)
np.random.seed(SEED)

final_model, final_optimizer = build_model()
train_fixed(
    final_model,
    criterion,
    final_optimizer,
    full_x_cat,
    full_x_num,
    full_y,
    epochs=max(1, int(best_epoch)),
    batch_size=256,
)

model = final_model

test_logits = predict_from_tensors(model, test_x_cat, test_x_num)
test_preds = _postprocess(test_logits)

submission_dir = Path("./submission")
submission_dir.mkdir(parents=True, exist_ok=True)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test_df), dtype=np.int64),
        TARGET: test_preds,
    }
)
submission.to_csv(submission_dir / "submission.csv", index=False)

print(f"Final Validation Score: {final_val_score}")
