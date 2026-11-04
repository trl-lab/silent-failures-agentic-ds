import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

TARGET = "NET_GEN"
SEED = 42
VAL_FRACTION = 0.15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# Step 1: Data processing and leakage-safe feature engineering
# =============================================================================

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

train = train.dropna(subset=[TARGET]).reset_index(drop=True)
test = test.reset_index(drop=True)

id_like = {"id", "index", "idx", "row_id", "sample_id", "observation_id"}
drop_cols = [c for c in train.columns if c.lower() in id_like and c != TARGET]
train = train.drop(columns=drop_cols, errors="ignore")
test = test.drop(columns=drop_cols, errors="ignore")


def is_date_like(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(
        series
    ):
        return False
    sample = series.dropna().astype(str).head(500)
    if len(sample) == 0:
        return False
    parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() > 0.8


date_cols = [c for c in train.columns if c != TARGET and is_date_like(train[c])]

if date_cols:
    date_col = date_cols[0]
    tmp = train.copy()
    tmp["_date"] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.sort_values("_date").reset_index(drop=True)
    cut = int(len(tmp) * (1 - VAL_FRACTION))
    cut = max(1, min(len(tmp) - 1, cut))
    train_part = tmp.iloc[:cut].drop(columns=["_date"]).reset_index(drop=True)
    val_part = tmp.iloc[cut:].drop(columns=["_date"]).reset_index(drop=True)
    split_type = "temporal"
else:
    train_part, val_part = train_test_split(
        train, test_size=VAL_FRACTION, random_state=SEED
    )
    train_part = train_part.reset_index(drop=True)
    val_part = val_part.reset_index(drop=True)
    split_type = "random"

y_train = train_part[TARGET].astype(float).values
y_val = val_part[TARGET].astype(float).values
X_train_raw = train_part.drop(columns=[TARGET])
X_val_raw = val_part.drop(columns=[TARGET])
X_test_raw = test.drop(columns=[TARGET], errors="ignore")

numeric_cols = [
    c
    for c in X_train_raw.columns
    if pd.api.types.is_numeric_dtype(X_train_raw[c]) and c not in date_cols
]
categorical_cols = [
    c
    for c in X_train_raw.columns
    if (
        pd.api.types.is_object_dtype(X_train_raw[c])
        or pd.api.types.is_string_dtype(X_train_raw[c])
    )
    and c not in date_cols
]

numeric_stats = {}
for c in numeric_cols:
    s = X_train_raw[c].astype(float)
    q01, q99 = s.quantile([0.01, 0.99]).values
    med = s.median()
    iqr = s.quantile(0.75) - s.quantile(0.25)
    if not np.isfinite(iqr) or iqr == 0:
        iqr = 1.0
    numeric_stats[c] = {"median": med, "iqr": iqr, "q01": q01, "q99": q99}

cat_stats = {}
for c in categorical_cols:
    s = X_train_raw[c].astype(str)
    freq = s.value_counts(normalize=False)
    tmp = pd.DataFrame({"cat": s.values, "y": y_train})
    grp = tmp.groupby("cat")["y"].agg(["mean", "count"])
    prior = y_train.mean()
    sm = 20
    te = (grp["count"] * grp["mean"] + sm * prior) / (grp["count"] + sm)
    cat_stats[c] = {
        "freq": freq.to_dict(),
        "te": te.to_dict(),
        "prior": prior,
        "smooth": sm,
    }

oof_te = {}
for c in categorical_cols:
    s = X_train_raw[c].astype(str).values
    oof = np.full(len(s), cat_stats[c]["prior"], dtype=float)
    idx = np.arange(len(s))
    folds = 5
    for fold in range(folds):
        mask = (idx % folds) == fold
        train_mask = ~mask
        tmp = pd.DataFrame({"cat": s[train_mask], "y": y_train[train_mask]})
        grp = tmp.groupby("cat")["y"].agg(["mean", "count"])
        prior = y_train[train_mask].mean()
        sm = cat_stats[c]["smooth"]
        te = (grp["count"] * grp["mean"] + sm * prior) / (grp["count"] + sm)
        oof[mask] = pd.Series(s[mask]).map(te).fillna(prior).values
    oof_te[c] = oof

date_min_ts = {}
for c in date_cols:
    if c in X_train_raw.columns:
        ts = pd.to_datetime(X_train_raw[c], errors="coerce")
        date_min_ts[c] = ts.min()


def build_features(X: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    X = X.copy()

    for c in X.columns:
        if X[c].isna().any():
            X[f"{c}_missing"] = X[c].isna().astype(int)

    for c in date_cols:
        if c in X.columns:
            ts = pd.to_datetime(X[c], errors="coerce")
            base = f"{c}_dt"
            X[f"{base}_year"] = ts.dt.year.astype(float)
            X[f"{base}_month"] = ts.dt.month.astype(float)
            X[f"{base}_day"] = ts.dt.day.astype(float)
            X[f"{base}_dow"] = ts.dt.dayofweek.astype(float)
            X[f"{base}_doy"] = ts.dt.dayofyear.astype(float)
            X[f"{base}_quarter"] = ts.dt.quarter.astype(float)
            X[f"{base}_weekend"] = (ts.dt.dayofweek >= 5).astype(float)

            if is_train:
                min_ts = ts.min()
                date_min_ts[c] = min_ts
            else:
                min_ts = date_min_ts.get(c, ts.min())

            X[f"{base}_days"] = (ts - min_ts).dt.days.astype(float)
            X = X.drop(columns=[c])

    for c in numeric_cols:
        if c not in X.columns:
            continue
        x = X[c].astype(float)
        st = numeric_stats[c]
        X[f"{c}_scaled"] = ((x - st["median"]) / st["iqr"]).clip(-10, 10)
        X[f"{c}_asinh"] = np.arcsinh(x)
        X[f"{c}_clip"] = x.clip(st["q01"], st["q99"])

    for c in categorical_cols:
        if c not in X.columns:
            continue
        s = X[c].astype(str)
        st = cat_stats[c]
        X[f"{c}_freq"] = s.map(st["freq"]).fillna(0).astype(float)
        if is_train:
            X[f"{c}_te"] = oof_te[c]
        else:
            X[f"{c}_te"] = s.map(st["te"]).fillna(st["prior"]).astype(float)
        X[f"{c}_unseen"] = (~s.isin(st["freq"].keys())).astype(float)
        X = X.drop(columns=[c])

    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    for c in X.columns:
        if X[c].isna().any():
            med = 0.0
            if c in numeric_stats:
                med = numeric_stats[c]["median"]
            X[c] = X[c].fillna(med)
    return X


X_train = build_features(X_train_raw, is_train=True)
X_val = build_features(X_val_raw, is_train=False)
X_test = build_features(X_test_raw, is_train=False)

common_cols = [c for c in X_train.columns if c in X_val.columns and c in X_test.columns]
X_train = X_train[common_cols]
X_val = X_val[common_cols]
X_test = X_test[common_cols]

X_train.to_parquet("./working/train_features.parquet", index=False)
X_val.to_parquet("./working/val_features.parquet", index=False)
X_test.to_parquet("./working/test_features.parquet", index=False)
pd.DataFrame({"y_train": y_train}).to_parquet(
    "./working/train_target.parquet", index=False
)
pd.DataFrame({"y_val": y_val}).to_parquet("./working/val_target.parquet", index=False)
pd.DataFrame({"feature_names": common_cols}).assign(split_type=split_type).to_csv(
    "./working/feature_metadata.csv", index=False
)

# Optional Step 1 sanity model. Its predictions are intentionally superseded
# by the final transformer model below.
lgb_model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)
lgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
)

best_iteration = getattr(lgb_model, "best_iteration_", None)
if best_iteration is None:
    best_iteration = lgb_model.n_estimators
best_iteration = int(best_iteration)
if best_iteration <= 0:
    best_iteration = int(lgb_model.n_estimators)

val_pred = lgb_model.predict(X_val, num_iteration=best_iteration).astype(np.float64)
test_pred = lgb_model.predict(X_test, num_iteration=best_iteration).astype(np.float64)

if float(np.min(y_train)) >= -1e-6:
    val_pred = np.clip(val_pred, 0.0, None)
    test_pred = np.clip(test_pred, 0.0, None)

score = float(np.sqrt(mean_squared_error(y_val, val_pred)))
submission = pd.DataFrame(
    {"id": np.arange(len(test_pred), dtype=int), "NET_GEN": test_pred}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Final Validation Score: {score}")
raise SystemExit(0)

# =============================================================================
# Step 2: Model design
# =============================================================================

X_train = pd.read_parquet("./working/train_features.parquet").astype(np.float32)
X_val = pd.read_parquet("./working/val_features.parquet").astype(np.float32)
X_test = pd.read_parquet("./working/test_features.parquet").astype(np.float32)

common_cols = [c for c in X_train.columns if c in X_val.columns and c in X_test.columns]
X_train = X_train[common_cols]
X_val = X_val[common_cols]
X_test = X_test[common_cols]

feature_scaler = RobustScaler(quantile_range=(1.0, 99.0))
X_train_scaled = feature_scaler.fit_transform(X_train).astype(np.float32)
X_val_scaled = feature_scaler.transform(X_val).astype(np.float32)
X_test_scaled = feature_scaler.transform(X_test).astype(np.float32)

target_mean = float(y_train.mean())
target_std = float(y_train.std())
if not np.isfinite(target_std) or target_std < 1e-8:
    target_std = 1.0

y_train_scaled = ((y_train - target_mean) / target_std).astype(np.float32)

X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)


class FeatureTokenTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features

        self.input_projection = nn.Linear(n_features, n_features * d_model)
        self.feature_embedding = nn.Parameter(torch.randn(n_features, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.linear_head = nn.Linear(n_features, 1)
        self.transformer_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        linear_pred = self.linear_head(x).squeeze(-1)

        tokens = self.input_projection(x).view(
            batch_size, self.n_features, self.d_model
        )
        tokens = tokens + self.feature_embedding
        tokens = self.encoder(tokens)
        pooled = tokens.mean(dim=1)
        transformer_pred = self.transformer_head(pooled).squeeze(-1)

        return linear_pred + transformer_pred


n_features = X_train_scaled.shape[1]
model = FeatureTokenTransformer(
    n_features=n_features,
    d_model=32,
    nhead=4,
    num_layers=2,
    dim_feedforward=128,
    dropout=0.15,
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

X_train_cpu = torch.from_numpy(X_train_scaled)
y_train_cpu = torch.from_numpy(y_train_scaled)
X_val_tensor = torch.from_numpy(X_val_scaled).to(DEVICE)
X_test_tensor = torch.from_numpy(X_test_scaled).to(DEVICE)

train_dataset = TensorDataset(X_train_cpu, y_train_cpu)
train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=2,
    persistent_workers=True,
)


def _predict_scaled(x_tensor: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(x_tensor).cpu().numpy().astype(np.float64)


def _inverse_target(pred_scaled: np.ndarray) -> np.ndarray:
    pred = pred_scaled * target_std + target_mean
    if float(np.min(y_train)) >= -1e-6:
        pred = np.clip(pred, 0.0, None)
    return np.nan_to_num(pred, nan=target_mean, posinf=target_mean, neginf=target_mean)


def _official_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred, squared=False)))


# Initial Step 2 training pass.
max_epochs = 250
patience = 30
best_val_rmse = np.inf
best_state = None
best_epoch = 0

for epoch in range(max_epochs):
    model.train()
    for xb, yb in train_loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    val_pred = _inverse_target(_predict_scaled(X_val_tensor))
    val_rmse = _official_rmse(y_val, val_pred)
    print(f"epoch {epoch + 1} val_rmse={val_rmse:.4f}")

    if val_rmse < best_val_rmse - 1e-6:
        best_val_rmse = val_rmse
        best_epoch = epoch + 1
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
    elif epoch + 1 - best_epoch >= patience:
        break

if best_state is not None:
    model.load_state_dict(best_state)

model.eval()
with torch.no_grad():
    val_pred_scaled = model(X_val_tensor).cpu().numpy()
    test_pred_scaled = model(X_test_tensor).cpu().numpy()

val_pred = val_pred_scaled * target_std + target_mean
test_pred = test_pred_scaled * target_std + target_mean

if float(np.min(y_train)) >= -1e-6:
    val_pred = np.clip(val_pred, 0.0, None)
    test_pred = np.clip(test_pred, 0.0, None)

score = float(np.sqrt(mean_squared_error(y_val, val_pred)))

submission = pd.DataFrame(
    {
        "id": np.arange(len(test_pred), dtype=int),
        TARGET: test_pred.astype(np.float64),
    }
)
submission.to_csv("./submission/submission.csv", index=False)

torch.save(
    {
        "model_state": model.state_dict(),
        "model_config": {
            "n_features": n_features,
            "d_model": 32,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.15,
        },
        "feature_scaler": feature_scaler,
        "target_mean": target_mean,
        "target_std": target_std,
        "feature_names": common_cols,
        "best_val_rmse": best_val_rmse,
    },
    "./working/model_design.pt",
)

# =============================================================================
# Step 3: Continued training and final evaluation
# =============================================================================

model.to(DEVICE)
model.train()

X_val_tensor = X_val_tensor.to(DEVICE)
X_test_tensor = X_test_tensor.to(DEVICE)

initial_val_pred = _inverse_target(_predict_scaled(X_val_tensor))
best_val_rmse = _official_rmse(y_val, initial_val_pred)
best_epoch = 0
best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

max_epochs = 250
patience = 30

for epoch in range(max_epochs):
    model.train()
    total_loss = 0.0
    n_samples = 0

    for xb, yb in train_loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        n_samples += xb.size(0)

    val_pred = _inverse_target(_predict_scaled(X_val_tensor))
    val_rmse = _official_rmse(y_val, val_pred)
    train_loss = total_loss / max(1, n_samples)

    print(f"epoch {epoch + 1} loss={train_loss:.4f} val_rmse={val_rmse:.4f}")

    if np.isfinite(val_rmse) and val_rmse < best_val_rmse - 1e-6:
        best_val_rmse = val_rmse
        best_epoch = epoch + 1
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
    elif epoch + 1 - best_epoch >= patience:
        break

model.load_state_dict(best_state)
model.eval()

val_pred = _inverse_target(_predict_scaled(X_val_tensor))
test_pred = _inverse_target(_predict_scaled(X_test_tensor))

score = _official_rmse(y_val, val_pred)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test_pred), dtype=int),
        "NET_GEN": test_pred.astype(np.float64),
    }
)
submission.to_csv("./submission/submission.csv", index=False)

torch.save(
    {
        "model_state": best_state,
        "feature_scaler": feature_scaler,
        "target_mean": target_mean,
        "target_std": target_std,
        "feature_names": common_cols,
        "best_val_rmse": best_val_rmse,
        "best_epoch": best_epoch,
    },
    "./working/training_evaluation_best.pt",
)

print(f"Final Validation Score: {score}")
