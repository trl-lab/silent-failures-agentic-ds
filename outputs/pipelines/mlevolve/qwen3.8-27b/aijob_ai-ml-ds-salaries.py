import os
import re
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import root_mean_squared_error

# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

TARGET = "salary_in_usd"
VAL_FRACTION = 0.2
SMOOTH_M = 20.0
ONEHOT_MAX_CARD = 20

# Domain vocabulary for AI/ML/DS salary signal (compact text features)
KEYWORDS = [
    "senior",
    "lead",
    "principal",
    "staff",
    "junior",
    "intern",
    "manager",
    "director",
    "head",
    "architect",
    "consultant",
    "deep learning",
    "machine learning",
    "data science",
    "data engineering",
    "nlp",
    "computer vision",
    "generative",
    "genai",
    "llm",
    "gpt",
    "transformer",
    "bert",
    "pytorch",
    "tensorflow",
    "keras",
    "scikit",
    "python",
    "r ",
    "sql",
    "java",
    "scala",
    "spark",
    "hadoop",
    "airflow",
    "aws",
    "gcp",
    "azure",
    "docker",
    "kubernetes",
    "mlflow",
    "etl",
    "product",
    "analytics",
    "data warehouse",
    "tableau",
    "power bi",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _is_listy(series):
    """True if the column holds list/tuple/set/ndarray values."""
    sample = series.dropna().head(50)
    if sample.empty:
        return False
    return all(isinstance(v, (list, tuple, set, np.ndarray)) for v in sample)


def _normalize_text(series):
    """Return a lower-cased string Series; join list-like values; NaN -> ''."""
    if _is_listy(series):
        s = series.map(
            lambda v: (
                " ".join(str(x) for x in v)
                if isinstance(v, (list, tuple, set, np.ndarray)) and len(v)
                else ""
            )
        )
    else:
        s = series.astype(object)
        s = s.map(lambda v: str(v) if pd.notna(v) else "")
    return s.fillna("").str.lower().str.strip()


def _safe_name(col):
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(col)).strip("_") or "col"


def _get_ids(df, length):
    if "id" in df.columns:
        return pd.to_numeric(df["id"], errors="coerce").fillna(0).astype(int).to_numpy()
    return np.arange(length, dtype=int)


# --------------------------------------------------------------------------- #
# Feature pipeline (fit on train fold ONLY, then transform val/test)
# --------------------------------------------------------------------------- #
class SalaryFeaturePipeline:
    def __init__(self, target=TARGET, keywords=KEYWORDS):
        self.target = target
        self.keywords = keywords
        self.numeric_cols = []
        self.categorical_cols = []
        self.medians = {}
        self.freq_maps = {}
        self.target_maps = {}
        self.onehot_cols = {}
        self.global_mean = 0.0
        self.feature_cols = []
        self.scaler = None
        self._fitted = False

    def _detect(self, df):
        for col in df.columns:
            if col == self.target or col == "id":
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                self.numeric_cols.append(col)
            else:
                self.categorical_cols.append(col)

    def fit(self, df):
        self._detect(df)
        self.global_mean = float(df[self.target].mean())

        # Numeric: median for imputation
        for col in self.numeric_cols:
            med = df[col].median()
            self.medians[col] = (
                float(med) if pd.notna(med) and np.isfinite(med) else 0.0
            )

        # Categorical: frequency + smoothed target encoding + one-hot vocab
        for col in self.categorical_cols:
            s = _normalize_text(df[col])
            counts = s.value_counts()
            self.freq_maps[col] = (counts / len(s)).to_dict()

            tmp = pd.DataFrame({"cat": s.values, "y": df[self.target].values})
            stats = tmp.groupby("cat")["y"].agg(["mean", "count"])
            smoothed = (
                stats["count"] * stats["mean"] + SMOOTH_M * self.global_mean
            ) / (stats["count"] + SMOOTH_M)
            self.target_maps[col] = smoothed.to_dict()

            # one-hot only for low-cardinality
            if s.nunique() <= ONEHOT_MAX_CARD:
                self.onehot_cols[col] = [str(c) for c in s.unique()]

        self._fitted = True
        return self

    def transform(self, df):
        if not self._fitted:
            raise RuntimeError("Pipeline must be fitted before transform.")
        out = pd.DataFrame(index=df.index)

        # ---- numeric features ------------------------------------------- #
        for col in self.numeric_cols:
            s = pd.to_numeric(df[col], errors="coerce")
            miss = s.isna().astype(int)
            s = s.fillna(self.medians[col])
            out[_safe_name(col)] = s
            out[_safe_name(col) + "_log"] = np.log1p(s.clip(lower=0))
            out[_safe_name(col) + "_miss"] = miss

        # ---- categorical features --------------------------------------- #
        for col in self.categorical_cols:
            s = _normalize_text(df[col])
            out[_safe_name(col) + "_freq"] = (
                s.map(self.freq_maps[col]).fillna(0.0).astype(float)
            )
            out[_safe_name(col) + "_tgt"] = (
                s.map(self.target_maps[col]).fillna(self.global_mean).astype(float)
            )
            if col in self.onehot_cols:
                for i, cat in enumerate(self.onehot_cols[col]):
                    out[_safe_name(col) + "_oh" + str(i)] = (s == cat).astype(int)

        # ---- text features (compact keyword + length block) ------------- #
        text_cols = list(self.categorical_cols)
        if text_cols:
            blob = pd.concat(
                [_normalize_text(df[c]) for c in text_cols], axis=1, ignore_index=True
            )
            blob = blob.fillna("").apply(lambda row: " ".join(row), axis=1)
        else:
            blob = pd.Series([""] * len(df), index=df.index)

        for kw in self.keywords:
            out["kw_" + _safe_name(kw)] = blob.str.contains(
                re.escape(kw), regex=True
            ).astype(int)

        out["text_n_words"] = blob.str.split().str.len().fillna(0).astype(float)
        out["text_len"] = blob.str.len().fillna(0).astype(float)

        # ---- robust scaling of all engineered numeric features ---------- #
        X = out.astype(float)
        if self.scaler is None:
            self.scaler = RobustScaler()
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        X = pd.DataFrame(X, index=out.index, columns=out.columns)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        self.feature_cols = list(X.columns)
        return X


# --------------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------------- #
train = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
test = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

# Ensure a stable integer id on test (0-based row index) for the submission.
if "id" not in test.columns:
    test = test.reset_index(drop=True)
    test["id"] = np.arange(len(test), dtype=int)
else:
    test["id"] = pd.to_numeric(test["id"], errors="coerce").fillna(0).astype(int)

print(f"Loaded train={train.shape}, test={test.shape}")
print(f"Train columns: {list(train.columns)}")

# Target sanity: coerce to float, keep finite rows only (train only)
train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
n_before = len(train)
train = train[np.isfinite(train[TARGET])].reset_index(drop=True)
print(f"Train rows with valid target: {len(train)} (dropped {n_before - len(train)})")

# --------------------------------------------------------------------------- #
# SPLIT FIRST, then fit/transform (strict leakage isolation)
# Stratify by salary decile so the validation fold spans the skewed range.
# --------------------------------------------------------------------------- #
try:
    strata = pd.qcut(train[TARGET], q=10, labels=False, duplicates="drop")
    train_part, val_part = train_test_split(
        train, test_size=VAL_FRACTION, random_state=SEED, stratify=strata
    )
except Exception:
    train_part, val_part = train_test_split(
        train, test_size=VAL_FRACTION, random_state=SEED
    )

print(f"Split -> train_part={train_part.shape}, val_part={val_part.shape}")

# --------------------------------------------------------------------------- #
# Fit pipeline on train_part ONLY, then transform all three folds
# --------------------------------------------------------------------------- #
pipe = SalaryFeaturePipeline(target=TARGET, keywords=KEYWORDS)
pipe.fit(train_part)

X_train = pipe.transform(train_part)
X_val = pipe.transform(val_part)
X_test = pipe.transform(test)

y_train = train_part[TARGET].values.astype(float)
y_val = val_part[TARGET].values.astype(float)

# Persist processed artifacts for the modeling step
X_train.to_parquet(os.path.join(WORKING_DIR, "X_train.parquet"))
X_val.to_parquet(os.path.join(WORKING_DIR, "X_val.parquet"))
X_test.to_parquet(os.path.join(WORKING_DIR, "X_test.parquet"))
pd.DataFrame(
    {
        "id": _get_ids(val_part, len(val_part)),
        TARGET: y_val,
    }
).to_parquet(os.path.join(WORKING_DIR, "y_val.parquet"))
pd.DataFrame(
    {
        "id": _get_ids(train_part, len(train_part)),
        TARGET: y_train,
    }
).to_parquet(os.path.join(WORKING_DIR, "y_train.parquet"))
pd.DataFrame({"id": test["id"].values}).to_parquet(
    os.path.join(WORKING_DIR, "test_ids.parquet")
)

print(f"Engineered feature matrix: {X_train.shape[1]} features")

# --------------------------------------------------------------------------- #
# Data-quality baseline (no-skill) for auditability only.
# The official local validation score is produced later by the trained model.
# --------------------------------------------------------------------------- #
train_mean = float(np.mean(y_train))
baseline_pred = np.full(len(y_val), train_mean)
baseline_rmse = float(root_mean_squared_error(y_val, baseline_pred))
print(f"[data-step] no-skill baseline RMSE (train-mean on val): {baseline_rmse:.4f}")


# --------------------------------------------------------------------------- #
# Model design: Heteroscedastic Mixture-of-Experts tabular network
# --------------------------------------------------------------------------- #
class _Expert(nn.Module):
    """Small residual expert used by the mixture-of-experts head."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class HeteroscedasticMOESalaryNet(nn.Module):
    """
    Architecture hypothesis:
    Salary data contains heterogeneous regimes (junior/mid/senior, industry,
    location, role family). A single MLP may average across these regimes.
    This model uses:
      - a shared encoder to learn a stable tabular representation,
      - multiple experts to model regime-specific mean salary,
      - a gating network to soft-route samples to experts,
      - a separate head to predict heteroscedastic uncertainty.

    The model outputs:
      mu_std: predicted salary in standardized training-target space
      log_sigma: log standard deviation in standardized target space
    """

    def __init__(
        self,
        input_dim: int,
        trunk_dims=(256, 128),
        expert_hidden_dim=96,
        n_experts=4,
        dropout=0.18,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")

        layers = []
        prev_dim = input_dim
        for dim in trunk_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.LayerNorm(dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = dim

        self.encoder = nn.Sequential(*layers)
        self.encoder_out_dim = trunk_dims[-1]

        self.experts = nn.ModuleList(
            [
                _Expert(
                    in_dim=self.encoder_out_dim,
                    hidden_dim=expert_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(n_experts)
            ]
        )

        self.gate = nn.Sequential(
            nn.Linear(self.encoder_out_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, n_experts),
        )

        self.sigma_head = nn.Sequential(
            nn.Linear(self.encoder_out_dim, 48),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(48, 1),
        )

        # Conservative initialization to avoid unstable early predictions.
        self.apply(self._init_weights)
        # Start with near-certain standardized predictions.
        with torch.no_grad():
            self.sigma_head[-1].weight.zero_()
            self.sigma_head[-1].bias.fill_(-2.0)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        h = self.encoder(x)

        # Expert outputs: (batch, n_experts)
        expert_out = torch.stack([expert(h) for expert in self.experts], dim=1)

        # Gate: (batch, n_experts)
        gate_logits = self.gate(h)
        gate_probs = F.softmax(gate_logits, dim=1)

        # Weighted mean expert: (batch,)
        mu_std = (gate_probs * expert_out).sum(dim=1)

        # Log-sigma head: (batch,)
        log_sigma = self.sigma_head(h).squeeze(-1)

        return mu_std, log_sigma


# --------------------------------------------------------------------------- #
# Loss design: heteroscedastic Gaussian NLL in standardized salary space
# --------------------------------------------------------------------------- #
class HeteroscedasticGaussianNLLLoss(nn.Module):
    """
    Loss aligned with RMSE semantics:
      - standardizes the raw USD target using training-fold statistics only,
      - predicts standardized salary mean and standard deviation,
      - minimizes negative log-likelihood of a Gaussian, whose core term is
        squared error weighted by predicted uncertainty.

    The official local validation metric remains raw-USD RMSE, computed
    outside this loss in the training step.
    """

    def __init__(self, target_mean: float, target_std: float, min_sigma: float = 0.05):
        super().__init__()
        if target_std <= 0:
            raise ValueError("target_std must be positive")

        self.register_buffer(
            "target_mean", torch.tensor(float(target_mean), dtype=torch.float32)
        )
        self.register_buffer(
            "target_std", torch.tensor(float(target_std), dtype=torch.float32)
        )
        self.min_sigma = float(min_sigma)

    def forward(self, mu_std, log_sigma, y_raw):
        y_std = (y_raw - self.target_mean) / self.target_std

        sigma = F.softplus(log_sigma) + self.min_sigma
        sigma2 = sigma * sigma

        # Gaussian NLL: 0.5 * (residual^2 / sigma^2 + log(sigma^2))
        nll = 0.5 * ((y_std - mu_std) ** 2 / sigma2 + torch.log(sigma2))
        return nll.mean()


# --------------------------------------------------------------------------- #
# Optimizer design
# --------------------------------------------------------------------------- #
def build_optimizer(model, lr: float = 3e-3, weight_decay: float = 1e-4):
    """
    AdamW with parameter-group-aware weight decay:
      - no decay for biases and LayerNorm affine parameters,
      - decay for linear weights.
    """
    no_decay_ids = set()
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            for _, param in module.named_parameters(recurse=False):
                no_decay_ids.add(id(param))

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in no_decay_ids or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [{"params": decay_params, "weight_decay": weight_decay}]
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

    return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)


# --------------------------------------------------------------------------- #
# Instantiate model, criterion, and optimizer from previous-step artifacts
# --------------------------------------------------------------------------- #
def _load_previous_step_artifacts():
    x_train_path = os.path.join(WORKING_DIR, "X_train.parquet")
    y_train_path = os.path.join(WORKING_DIR, "y_train.parquet")
    x_val_path = os.path.join(WORKING_DIR, "X_val.parquet")
    y_val_path = os.path.join(WORKING_DIR, "y_val.parquet")

    if not (
        os.path.exists(x_train_path)
        and os.path.exists(y_train_path)
        and os.path.exists(x_val_path)
        and os.path.exists(y_val_path)
    ):
        raise FileNotFoundError(
            "Expected previous-step artifacts in ./working: "
            "X_train.parquet, y_train.parquet, X_val.parquet, y_val.parquet"
        )

    X_train = pd.read_parquet(x_train_path)
    y_train_df = pd.read_parquet(y_train_path)
    X_val = pd.read_parquet(x_val_path)
    y_val_df = pd.read_parquet(y_val_path)

    y_train = y_train_df[TARGET].to_numpy(dtype=np.float32)
    y_val = y_val_df[TARGET].to_numpy(dtype=np.float32)

    return X_train, y_train, X_val, y_val


def _to_float_tensor(df):
    arr = df.to_numpy(dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(arr)


X_train, y_train, X_val, y_val = _load_previous_step_artifacts()

input_dim = int(X_train.shape[1])
target_mean = float(np.mean(y_train))
if not np.isfinite(target_mean):
    target_mean = 0.0
target_std = float(np.std(y_train))
if target_std <= 0 or not np.isfinite(target_std):
    target_std = 1.0

model = HeteroscedasticMOESalaryNet(
    input_dim=input_dim,
    trunk_dims=(256, 128),
    expert_hidden_dim=96,
    n_experts=4,
    dropout=0.18,
)

criterion = HeteroscedasticGaussianNLLLoss(
    target_mean=target_mean,
    target_std=target_std,
    min_sigma=0.05,
)

optimizer = build_optimizer(model, lr=3e-3, weight_decay=1e-4)

# --------------------------------------------------------------------------- #
# Smoke validation for model-design step only.
# This is NOT the final validation score; it only verifies forward-pass sanity.
# --------------------------------------------------------------------------- #
model.eval()
with torch.no_grad():
    X_val_tensor = _to_float_tensor(X_val)
    mu_std, log_sigma = model(X_val_tensor)
    val_pred_raw = (mu_std * target_std + target_mean).numpy()

val_pred_raw = np.nan_to_num(
    val_pred_raw, nan=target_mean, posinf=target_mean, neginf=target_mean
)
smoke_rmse = float(root_mean_squared_error(y_val, val_pred_raw))

print(
    f"Model design smoke check: input_dim={input_dim}, "
    f"target_mean={target_mean:.4f}, target_std={target_std:.4f}, "
    f"untrained_forward_rmse={smoke_rmse:.6f}"
)


# --------------------------------------------------------------------------- #
# Training / evaluation configuration
# --------------------------------------------------------------------------- #
BATCH_SIZE = 512
MAX_EPOCHS = 300
PATIENCE = 20
WARMUP_EPOCHS = 10
MIN_LR_FACTOR = 0.01
GRAD_CLIP_NORM = 1.0
VAL_BATCH_SIZE = 8192

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Small utilities for robust tensor conversion and identical inference
# --------------------------------------------------------------------------- #
def _as_float_array(obj):
    """Convert DataFrame / Tensor / ndarray to a clean float32 numpy array."""
    if isinstance(obj, pd.DataFrame):
        arr = obj.to_numpy(dtype=np.float32)
    elif isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu().numpy().astype(np.float32)
    else:
        arr = np.asarray(obj, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _as_float_vector(obj):
    """Convert Series / Tensor / ndarray to a clean 1D float32 numpy array."""
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    if isinstance(obj, pd.Series):
        obj = obj.to_numpy()
    arr = np.asarray(obj, dtype=np.float32).reshape(-1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _predict_raw_usd(features, model, target_mean, target_std, device):
    """
    Identical inference path for validation and test:
      forward pass -> mu in standardized salary space -> raw USD -> sanitize -> clip >= 0.
    """
    model.eval()
    x_arr = _as_float_array(features)
    if x_arr.size == 0:
        return np.array([], dtype=np.float64)

    preds = []
    with torch.no_grad():
        for start in range(0, x_arr.shape[0], VAL_BATCH_SIZE):
            xb = torch.from_numpy(x_arr[start : start + VAL_BATCH_SIZE]).to(device)
            mu_std, _ = model(xb)
            pred_raw = mu_std.cpu().numpy() * target_std + target_mean
            preds.append(pred_raw)

    pred_raw = (
        np.concatenate(preds, axis=0) if preds else np.array([], dtype=np.float64)
    )
    pred_raw = np.nan_to_num(
        pred_raw,
        nan=target_mean,
        posinf=target_mean,
        neginf=target_mean,
    )
    # Salary is non-negative; this post-processing is applied identically to val and test.
    pred_raw = np.clip(pred_raw, 0.0, None)
    return pred_raw.astype(np.float64)


# --------------------------------------------------------------------------- #
# Prepare arrays
# --------------------------------------------------------------------------- #
X_train_arr = _as_float_array(X_train)
y_train_arr = _as_float_vector(y_train)
X_val_arr = _as_float_array(X_val)
y_val_arr = _as_float_vector(y_val)

if "X_test" not in globals():
    X_test = pd.read_parquet(os.path.join(WORKING_DIR, "X_test.parquet"))

X_test_arr = _as_float_array(X_test)

# --------------------------------------------------------------------------- #
# Training data loader
# --------------------------------------------------------------------------- #
train_dataset = TensorDataset(
    torch.from_numpy(X_train_arr),
    torch.from_numpy(y_train_arr),
)

generator = torch.Generator()
generator.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=(device.type == "cuda"),
    generator=generator,
    drop_last=False,
)

# --------------------------------------------------------------------------- #
# Move existing model/criterion to device
# --------------------------------------------------------------------------- #
model.to(device)
criterion.to(device)


# --------------------------------------------------------------------------- #
# Scheduler: warmup + cosine decay
# --------------------------------------------------------------------------- #
def _lr_lambda(epoch: int) -> float:
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / max(1, WARMUP_EPOCHS)
    progress = (epoch - WARMUP_EPOCHS) / max(1, MAX_EPOCHS - WARMUP_EPOCHS)
    progress = min(1.0, max(0.0, progress))
    return MIN_LR_FACTOR + 0.5 * (1.0 - MIN_LR_FACTOR) * (
        1.0 + math.cos(math.pi * progress)
    )


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


# --------------------------------------------------------------------------- #
# Validation: official metric is raw-USD RMSE
# --------------------------------------------------------------------------- #
def _validate(model):
    val_pred = _predict_raw_usd(X_val_arr, model, target_mean, target_std, device)
    if val_pred.shape[0] != y_val_arr.shape[0]:
        raise RuntimeError("Validation prediction length mismatch.")
    return float(root_mean_squared_error(y_val_arr, val_pred))


# --------------------------------------------------------------------------- #
# Training loop with early stopping on official validation RMSE
# --------------------------------------------------------------------------- #
best_val_rmse = float("inf")
best_state = None
patience_counter = 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        mu_std, log_sigma = model(xb)
        loss = criterion(mu_std, log_sigma, yb)

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    scheduler.step()

    val_rmse = _validate(model)
    current_lr = optimizer.param_groups[0]["lr"]
    avg_loss = total_loss / max(1, n_batches)

    print(
        f"Epoch {epoch:03d} | train_loss {avg_loss:.6f} | "
        f"val_rmse {val_rmse:.6f} | lr {current_lr:.6f}"
    )

    if np.isfinite(val_rmse) and val_rmse < best_val_rmse - 1e-7:
        best_val_rmse = val_rmse
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch:03d}.")
            break

# --------------------------------------------------------------------------- #
# Restore best model and save checkpoint
# --------------------------------------------------------------------------- #
if best_state is not None:
    model.load_state_dict(best_state)
    torch.save(
        {
            "model_state_dict": best_state,
            "target_mean": float(target_mean),
            "target_std": float(target_std),
            "best_val_rmse": float(best_val_rmse),
        },
        os.path.join(WORKING_DIR, "best_model.pt"),
    )
else:
    current_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(
        {
            "model_state_dict": current_state,
            "target_mean": float(target_mean),
            "target_std": float(target_std),
            "best_val_rmse": float("inf"),
        },
        os.path.join(WORKING_DIR, "best_model.pt"),
    )

# --------------------------------------------------------------------------- #
# Final validation score using the same inference path as test
# --------------------------------------------------------------------------- #
model.eval()
final_val_pred = _predict_raw_usd(X_val_arr, model, target_mean, target_std, device)
score = float(root_mean_squared_error(y_val_arr, final_val_pred))

# --------------------------------------------------------------------------- #
# Test inference and submission
# --------------------------------------------------------------------------- #
test_pred = _predict_raw_usd(X_test_arr, model, target_mean, target_std, device)

# Official submission format: id is 0-based integer row index.
test_ids = np.arange(len(test_pred), dtype=int)

submission = pd.DataFrame(
    {
        "id": test_ids,
        TARGET: test_pred,
    }
)

submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)

print(f"Submission saved to {submission_path} with {len(submission)} rows.")
print(f"Final Validation Score: {score}")
