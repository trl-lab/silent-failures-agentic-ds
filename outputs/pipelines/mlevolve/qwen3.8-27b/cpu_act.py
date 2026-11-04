import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

SEED = 42
EPS = 1e-9
TARGET = "usr"

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Official CompAct features: attributes 1-21. Explicitly exclude target and
# CPU-state columns (sys/wio/idle) if present, to avoid trivial leakage.
FEATURE_ORDER = [
    "lread",
    "lwrite",
    "scall",
    "sread",
    "swrite",
    "fork",
    "exec",
    "rchar",
    "wchar",
    "pgout",
    "ppgout",
    "pgfree",
    "pgscan",
    "atch",
    "pgin",
    "ppgin",
    "pflt",
    "vflt",
    "runqsz",
    "freemem",
    "freeswap",
]
EXCLUDE_COLS = {"sys", "wio", "idle"}

train_raw = pd.read_parquet("./input/train.parquet")
test_raw = pd.read_parquet("./input/test.parquet")

train_raw[TARGET] = pd.to_numeric(train_raw[TARGET], errors="coerce")
train_raw = train_raw.dropna(subset=[TARGET])

id_cols = [c for c in train_raw.columns if str(c).lower() in ("id", "index")]
candidate_cols = [
    c
    for c in train_raw.columns
    if c != TARGET
    and c not in id_cols
    and c in test_raw.columns
    and pd.api.types.is_numeric_dtype(train_raw[c])
]

base_cols = [c for c in FEATURE_ORDER if c in candidate_cols]
if not base_cols:
    base_cols = [c for c in candidate_cols if c not in EXCLUDE_COLS]

print(f"Features: {len(base_cols)} | split: 90/10 seed={SEED} | metric: RMSE")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df[base_cols].astype(float).copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    for c in out.columns:
        if (out[c] < 0).any():
            out[c] = out[c].clip(lower=0.0)
    return out


train_clean = clean(train_raw)
test_clean = clean(test_raw)

# Split first, then fit imputation statistics only on the training split.
train_split_df = train_clean.copy()
train_split_df[TARGET] = train_raw[TARGET].to_numpy()
train_part, val_part = train_test_split(
    train_split_df,
    test_size=0.10,
    random_state=SEED,
)

tr = train_part[base_cols].copy()
va = val_part[base_cols].copy()
te = test_clean.copy()

y_tr = train_part[TARGET].to_numpy(dtype=np.float64)
y_va = val_part[TARGET].to_numpy(dtype=np.float64)

fill = tr.median(numeric_only=True).fillna(0.0)
tr = tr.fillna(fill)
va = va.fillna(fill)
te = te.fillna(fill)


def engineer(X: pd.DataFrame) -> pd.DataFrame:
    f = X[base_cols].copy()
    has = lambda c: c in X.columns

    for c in base_cols:
        f[f"{c}_log"] = np.log1p(X[c])

    for n, d in [
        ("sread", "scall"),
        ("swrite", "scall"),
        ("wchar", "rchar"),
        ("ppgout", "pgout"),
        ("ppgin", "pgin"),
        ("pgfree", "pgscan"),
        ("lread", "lwrite"),
        ("fork", "exec"),
        ("exec", "fork"),
    ]:
        if has(n) and has(d):
            f[f"{n}_over_{d}"] = X[n] / (X[d] + EPS)

    if has("lread") and has("lwrite"):
        f["mem_xfer"] = X["lread"] + X["lwrite"]
        f["rw_diff"] = X["lread"] - X["lwrite"]

    if has("sread") and has("swrite"):
        f["sys_io"] = X["sread"] + X["swrite"]

    if has("rchar") and has("wchar"):
        f["chars"] = X["rchar"] + X["wchar"]
        f["chars_diff"] = X["rchar"] - X["wchar"]

    if has("pflt") and has("vflt"):
        f["faults"] = X["pflt"] + X["vflt"]
        f["fault_mix"] = X["pflt"] / (X["pflt"] + X["vflt"] + EPS)

    if has("pgout") and has("pgin"):
        f["paging"] = X["pgout"] + X["pgin"]
        f["pg_diff"] = X["pgout"] - X["pgin"]

    if has("ppgout") and has("ppgin"):
        f["pages_moved"] = X["ppgout"] + X["ppgin"]
        f["ppg_diff"] = X["ppgout"] - X["ppgin"]

    if has("pgout") and has("pgfree") and has("pgscan"):
        f["pg_work"] = X["pgout"] + X["pgfree"] + X["pgscan"]

    for name in [
        "mem_xfer",
        "sys_io",
        "chars",
        "faults",
        "paging",
        "pages_moved",
        "pg_work",
    ]:
        if name in f:
            f[f"{name}_log"] = np.log1p(f[name])

    if has("runqsz") and has("scall"):
        f["runqsz_x_logscall"] = X["runqsz"] * np.log1p(X["scall"])

    if has("freemem") and has("pgfree"):
        f["freemem_x_logpgfree"] = X["freemem"] * np.log1p(X["pgfree"])

    if has("freeswap") and has("pgout"):
        f["freeswap_x_logpgout"] = X["freeswap"] * np.log1p(X["pgout"])

    if has("runqsz") and has("vflt"):
        f["runqsz_x_logvflt"] = X["runqsz"] * np.log1p(X["vflt"])

    if has("freemem") and has("pgout") and has("pgin"):
        f["freemem_x_logpaging"] = X["freemem"] * np.log1p(X["pgout"] + X["pgin"])

    for name in ["scall", "rchar", "vflt"]:
        if has(name):
            f[f"{name}_log_sq"] = np.log1p(X[name]) ** 2

    out = pd.DataFrame(f, index=X.index)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


tr_f = engineer(tr)
va_f = engineer(va)
te_f = engineer(te)
feat_cols = list(tr_f.columns)

scaler = StandardScaler().fit(tr_f.values)
tr_out = pd.DataFrame(scaler.transform(tr_f.values), columns=feat_cols)
va_out = pd.DataFrame(scaler.transform(va_f.values), columns=feat_cols)
te_out = pd.DataFrame(scaler.transform(te_f.values), columns=feat_cols)

torch.manual_seed(SEED)
np.random.seed(SEED)

X_train = torch.tensor(tr_out.values, dtype=torch.float32).cpu()
y_train = torch.tensor(y_tr, dtype=torch.float32).cpu()
X_val = torch.tensor(va_out.values, dtype=torch.float32).cpu()
y_val = torch.tensor(y_va, dtype=torch.float32).cpu()
X_test = torch.tensor(te_out.values, dtype=torch.float32).cpu()


class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


model = MLP(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience=3,
    factor=0.5,
)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=2,
    pin_memory=False,
    drop_last=False,
)


def predict_with_model(
    model: nn.Module, X: torch.Tensor, batch_size: int = 4096
) -> np.ndarray:
    model.eval()
    predictions = []

    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            xb = X[start : start + batch_size]
            if xb.ndim == 1:
                xb = xb.unsqueeze(0)
            preds = model(xb).detach().cpu().numpy().reshape(-1)
            predictions.append(preds)

    if len(predictions) == 0:
        return np.empty(0, dtype=np.float32)

    pred = np.concatenate(predictions, axis=0)
    return np.clip(pred, 0.0, 100.0).astype(np.float32, copy=False)


def official_rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if y_true.size == 0:
        return float("nan")

    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


current_val_pred = predict_with_model(model, X_val)
best_val_rmse = official_rmse(y_val, current_val_pred)
best_state = copy.deepcopy(model.state_dict())

max_epochs = 300
patience = 20
min_delta = 1e-4
epochs_without_improvement = 0

model.train()

for epoch in range(1, max_epochs + 1):
    epoch_loss = 0.0
    n_seen = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()

        yb = yb.reshape(-1)
        pred = model(xb).reshape(-1)
        loss = criterion(pred, yb)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item() * xb.shape[0]
        n_seen += xb.shape[0]

    train_loss = epoch_loss / max(n_seen, 1)
    val_pred = predict_with_model(model, X_val)
    val_rmse = official_rmse(y_val, val_pred)
    scheduler.step(val_rmse)

    print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_rmse={val_rmse:.6f}")

    if val_rmse < best_val_rmse - min_delta:
        best_val_rmse = val_rmse
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

torch.save(best_state, "./working/best_model.pth")

model.load_state_dict(best_state)
final_val_pred = predict_with_model(model, X_val)
final_test_pred = predict_with_model(model, X_test)

cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 25,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "seed": SEED,
    "n_jobs": -1,
    "verbose": -1,
}

cv_best_iters = []
for tr_idx, va_idx in cv.split(tr_f):
    dtrain = lgb.Dataset(tr_f.iloc[tr_idx], label=y_tr[tr_idx])
    dvalid = lgb.Dataset(tr_f.iloc[va_idx], label=y_tr[va_idx])

    model_cv = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )
    cv_best_iters.append(max(1, int(model_cv.best_iteration)))

n_estimators_lgb = max(100, int(round(np.mean(cv_best_iters))))

lgb_val_preds = []
lgb_test_preds = []

for lgb_seed in [42, 123, 2024]:
    lgb_model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=25,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        n_estimators=n_estimators_lgb,
        random_state=lgb_seed,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(tr_f, y_tr)

    lgb_val_preds.append(np.asarray(lgb_model.predict(va_f), dtype=np.float64))
    lgb_test_preds.append(np.asarray(lgb_model.predict(te_f), dtype=np.float64))

lgb_val_pred = np.clip(np.mean(lgb_val_preds, axis=0), 0.0, 100.0)
lgb_test_pred = np.clip(np.mean(lgb_test_preds, axis=0), 0.0, 100.0)

best_blend_w = 0.0
best_blend_val_rmse = float("inf")

for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
    val_blend = np.clip(
        (1.0 - w) * final_val_pred + w * lgb_val_pred,
        0.0,
        100.0,
    )
    val_blend_rmse = official_rmse(y_val, val_blend)

    if val_blend_rmse < best_blend_val_rmse:
        best_blend_val_rmse = val_blend_rmse
        best_blend_w = w

final_blend_test = np.clip(
    (1.0 - best_blend_w) * final_test_pred + best_blend_w * lgb_test_pred,
    0.0,
    100.0,
)

score = official_rmse(
    y_val,
    np.clip(
        (1.0 - best_blend_w) * final_val_pred + best_blend_w * lgb_val_pred,
        0.0,
        100.0,
    ),
)

submission = pd.DataFrame(
    {
        "id": np.arange(final_blend_test.shape[0], dtype=int),
        "usr": final_blend_test.astype(np.float64, copy=False),
    }
)
submission.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {score}")
