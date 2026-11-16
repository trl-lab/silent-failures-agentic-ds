import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
import os
import copy
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# ============================================================
# 2. IDENTIFY COLUMN TYPES
# ============================================================
target_col = "projectPayment"

y_full = train[target_col].copy()
train = train.drop(columns=[target_col])

cat_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = train.select_dtypes(include=["number"]).columns.tolist()

for col in num_cols[:]:
    if train[col].nunique() < 20:
        cat_cols.append(col)
        num_cols.remove(col)

print(f"Categorical columns: {cat_cols}")
print(f"Numeric columns: {num_cols}")

# ============================================================
# 3. CREATE TRAIN/VAL SPLIT
# ============================================================
train_df = train.copy()
train_df[target_col] = y_full

n_bins = min(10, len(train_df) // 100)
if n_bins < 2:
    n_bins = 2
train_df["_strat"] = pd.qcut(train_df[target_col], q=n_bins, duplicates="drop")

train_part, val_part = train_test_split(
    train_df, test_size=0.15, random_state=42, stratify=train_df["_strat"]
)

train_part = train_part.drop(columns=["_strat"])
val_part = val_part.drop(columns=["_strat"])

X_train = train_part.drop(columns=[target_col])
y_train = train_part[target_col]
X_val = val_part.drop(columns=[target_col])
y_val = val_part[target_col]
X_test = test.copy()

print(f"Split: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
def engineer_features(X_train, X_val, X_test, y_train, cat_cols, num_cols):
    X_tr = X_train.copy()
    X_va = X_val.copy()
    X_te = X_test.copy()

    # --- 4a: Handle missing values ---
    for col in X_tr.columns:
        if X_tr[col].isnull().any():
            if col in cat_cols:
                train_mode = X_tr[col].mode()
                fill_val = train_mode.iloc[0] if len(train_mode) > 0 else "MISSING"
                X_tr[col] = X_tr[col].fillna(fill_val)
                X_va[col] = X_va[col].fillna(fill_val)
                X_te[col] = X_te[col].fillna(fill_val)
            else:
                train_median = X_tr[col].median()
                X_tr[col] = X_tr[col].fillna(train_median)
                X_va[col] = X_va[col].fillna(train_median)
                X_te[col] = X_te[col].fillna(train_median)

    # Add missing indicators
    for col in X_train.columns:
        if X_train[col].isnull().any():
            miss_col = f"{col}_missing"
            X_tr[miss_col] = X_train[col].isnull().astype(int)
            X_va[miss_col] = X_val[col].isnull().astype(int)
            X_te[miss_col] = X_test[col].isnull().astype(int)

    # --- 4b: Frequency Encoding for categoricals ---
    for col in cat_cols:
        if col in X_tr.columns:
            freq_map = X_tr[col].value_counts(normalize=True)
            X_tr[f"{col}_freq"] = X_tr[col].map(freq_map)
            X_va[f"{col}_freq"] = X_va[col].map(freq_map).fillna(0)
            X_te[f"{col}_freq"] = X_te[col].map(freq_map).fillna(0)

    # --- 4c: Target Encoding with smoothing ---
    global_mean = y_train.mean()
    smoothing = 10

    for col in cat_cols:
        if col in X_tr.columns:
            agg = y_train.groupby(X_tr[col]).agg(["mean", "count"])
            smooth_mean = (agg["count"] * agg["mean"] + smoothing * global_mean) / (
                agg["count"] + smoothing
            )
            X_tr[f"{col}_target_enc"] = X_tr[col].map(smooth_mean)
            X_va[f"{col}_target_enc"] = X_va[col].map(smooth_mean).fillna(global_mean)
            X_te[f"{col}_target_enc"] = X_te[col].map(smooth_mean).fillna(global_mean)

            agg_std = y_train.groupby(X_tr[col]).std()
            X_tr[f"{col}_target_std"] = X_tr[col].map(agg_std).fillna(0)
            X_va[f"{col}_target_std"] = X_va[col].map(agg_std).fillna(0)
            X_te[f"{col}_target_std"] = X_te[col].map(agg_std).fillna(0)

    # --- 4d: Numeric feature transformations ---
    for col in num_cols:
        if col in X_tr.columns:
            X_tr[f"{col}_slog"] = np.sign(X_tr[col]) * np.log1p(np.abs(X_tr[col]))
            X_va[f"{col}_slog"] = np.sign(X_va[col]) * np.log1p(np.abs(X_va[col]))
            X_te[f"{col}_slog"] = np.sign(X_te[col]) * np.log1p(np.abs(X_te[col]))

            X_tr[f"{col}_sq"] = X_tr[col] ** 2
            X_va[f"{col}_sq"] = X_va[col] ** 2
            X_te[f"{col}_sq"] = X_te[col] ** 2

            X_tr[f"{col}_cube"] = X_tr[col] ** 3
            X_va[f"{col}_cube"] = X_va[col] ** 3
            X_te[f"{col}_cube"] = X_te[col] ** 3

            X_tr[f"{col}_inv"] = np.where(
                np.abs(X_tr[col]) > 1e-8, 1.0 / X_tr[col], 0.0
            )
            X_va[f"{col}_inv"] = np.where(
                np.abs(X_va[col]) > 1e-8, 1.0 / X_va[col], 0.0
            )
            X_te[f"{col}_inv"] = np.where(
                np.abs(X_te[col]) > 1e-8, 1.0 / X_te[col], 0.0
            )

    # --- 4e: Interaction features between numeric columns ---
    if len(num_cols) >= 2:
        for i in range(min(len(num_cols), 6)):
            for j in range(i + 1, min(len(num_cols), 6)):
                c1, c2 = num_cols[i], num_cols[j]
                if c1 in X_tr.columns and c2 in X_tr.columns:
                    X_tr[f"{c1}_x_{c2}"] = X_tr[c1] * X_tr[c2]
                    X_va[f"{c1}_x_{c2}"] = X_va[c1] * X_va[c2]
                    X_te[f"{c1}_x_{c2}"] = X_te[c1] * X_te[c2]

    # --- 4f: Interaction between numeric and categorical ---
    for cat in cat_cols[:4]:
        if cat not in X_tr.columns:
            continue
        for num in num_cols[:4]:
            if num not in X_tr.columns:
                continue
            grp_mean = X_tr.groupby(cat)[num].transform("mean")
            grp_std = X_tr.groupby(cat)[num].transform("std").fillna(0)

            X_tr[f"{num}_grp_{cat}"] = X_tr[num] - grp_mean
            X_va[f"{num}_grp_{cat}"] = X_va[num] - X_va[cat].map(grp_mean).fillna(0)
            X_te[f"{num}_grp_{cat}"] = X_te[num] - X_te[cat].map(grp_mean).fillna(0)

            X_tr[f"{num}_grpstd_{cat}"] = X_tr[num] / (grp_std + 1e-8)
            X_va[f"{num}_grpstd_{cat}"] = X_va[num] / (
                X_va[cat].map(grp_std).fillna(1e-8) + 1e-8
            )
            X_te[f"{num}_grpstd_{cat}"] = X_te[num] / (
                X_te[cat].map(grp_std).fillna(1e-8) + 1e-8
            )

    # --- 4g: Rank features for numeric columns ---
    for col in num_cols:
        if col in X_tr.columns:
            train_rank = X_tr[col].rank(pct=True)
            X_tr[f"{col}_rank"] = train_rank
            # For val/test, map through train's value-to-rank mapping
            val_rank_map = dict(zip(X_tr[col].values, train_rank.values))
            test_rank_map = dict(zip(X_tr[col].values, train_rank.values))
            X_va[f"{col}_rank"] = X_va[col].map(val_rank_map).fillna(0.5)
            X_te[f"{col}_rank"] = X_te[col].map(test_rank_map).fillna(0.5)

    # --- 4h: Z-score features ---
    for col in num_cols:
        if col in X_tr.columns:
            train_mean = X_tr[col].mean()
            train_std = X_tr[col].std() + 1e-8
            X_tr[f"{col}_zscore"] = (X_tr[col] - train_mean) / train_std
            X_va[f"{col}_zscore"] = (X_va[col] - train_mean) / train_std
            X_te[f"{col}_zscore"] = (X_te[col] - train_mean) / train_std

    # --- 4i: Drop original categorical (string) columns ---
    cols_to_drop = [col for col in cat_cols if col in X_tr.columns]
    X_tr = X_tr.drop(columns=cols_to_drop)
    X_va = X_va.drop(columns=cols_to_drop)
    X_te = X_te.drop(columns=cols_to_drop)

    return X_tr, X_va, X_te


X_train_fe, X_val_fe, X_test_fe = engineer_features(
    X_train, X_val, X_test, y_train, cat_cols, num_cols
)

print(
    f"After feature engineering: Train={X_train_fe.shape}, Val={X_val_fe.shape}, Test={X_test_fe.shape}"
)


# ============================================================
# 5. MODEL DEFINITION
# ============================================================
class ExpertNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.2):
        super(ExpertNetwork, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        self.network = nn.Sequential(*layers)

        self.use_residual = input_dim == hidden_dims[-1]
        if not self.use_residual:
            self.residual_proj = nn.Linear(input_dim, hidden_dims[-1])

    def forward(self, x):
        out = self.network(x)
        if self.use_residual:
            return out + x
        else:
            return out + self.residual_proj(x)


class MixtureOfExperts(nn.Module):
    def __init__(
        self,
        input_dim,
        num_experts=4,
        hidden_dims=[128, 64, 32],
        dropout_rate=0.2,
        gate_temperature=1.0,
    ):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.gate_temperature = gate_temperature

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
        )

        expert_hidden = hidden_dims[1:]
        self.experts = nn.ModuleList(
            [
                ExpertNetwork(hidden_dims[0], expert_hidden, dropout_rate)
                for _ in range(num_experts)
            ]
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_dims[0], 32), nn.GELU(), nn.Linear(32, num_experts)
        )

        self.regression_head = nn.Sequential(
            nn.Linear(expert_hidden[-1], 16),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(16, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x_proj = self.input_proj(x)
        gate_logits = self.gate(x_proj)
        gate_weights = F.softmax(gate_logits / self.gate_temperature, dim=-1)
        gate_weights = gate_weights.unsqueeze(-1)

        expert_outputs = torch.stack([expert(x_proj) for expert in self.experts], dim=1)
        combined = (gate_weights * expert_outputs).sum(dim=1)
        prediction = self.regression_head(combined).squeeze(-1)
        return prediction


class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, predictions, targets):
        error = predictions - targets
        abs_error = torch.abs(error)
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        loss = 0.5 * quadratic**2 + self.delta * linear
        return loss.mean()


class WarmupCosineScheduler:
    def __init__(
        self, optimizer, warmup_epochs=5, total_epochs=200, min_lr=1e-5, max_lr=1e-3
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.min_lr + (self.max_lr - self.min_lr) * (
                self.current_epoch / self.warmup_epochs
            )
        else:
            progress = (self.current_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def get_last_lr(self):
        if self.current_epoch <= self.warmup_epochs:
            return self.min_lr + (self.max_lr - self.min_lr) * (
                self.current_epoch / self.warmup_epochs
            )
        progress = (self.current_epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs
        )
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1 + math.cos(math.pi * progress)
        )


# ============================================================
# 6. PREPARE TENSORS
# ============================================================
input_dim = X_train_fe.shape[1]
print(f"Input dimension: {input_dim}")

X_train_np = X_train_fe.values.astype(np.float32)
X_val_np = X_val_fe.values.astype(np.float32)
X_test_np = X_test_fe.values.astype(np.float32)
y_train_np = y_train.values.astype(np.float32)
y_val_np = y_val.values.astype(np.float32)

X_train_np = np.nan_to_num(X_train_np, nan=0.0, posinf=0.0, neginf=0.0)
X_val_np = np.nan_to_num(X_val_np, nan=0.0, posinf=0.0, neginf=0.0)
X_test_np = np.nan_to_num(X_test_np, nan=0.0, posinf=0.0, neginf=0.0)

train_dataset = TensorDataset(
    torch.from_numpy(X_train_np), torch.from_numpy(y_train_np)
)
val_dataset = TensorDataset(torch.from_numpy(X_val_np), torch.from_numpy(y_val_np))
test_dataset = TensorDataset(torch.from_numpy(X_test_np))

BATCH_SIZE = 64
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=False
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)


# ============================================================
# 7. TRAINING FUNCTION
# ============================================================
def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    max_epochs=300,
    patience=50,
    device="cpu",
):
    best_val_rmse = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            n_batches += 1

        model.eval()
        val_preds = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                preds = model(X_batch)
                val_preds.append(preds.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_rmse = np.sqrt(mean_squared_error(y_val_np, val_preds))

        avg_loss = train_loss / n_batches
        print(
            f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Val RMSE: {val_rmse:.4f} | LR: {scheduler.get_last_lr():.2e}"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping at epoch {epoch+1}. Best Val RMSE: {best_val_rmse:.4f}"
                )
                break

    model.load_state_dict(best_model_state)
    return model, best_val_rmse


# ============================================================
# 8. MULTI-SEED ENSEMBLE TRAINING
# ============================================================
device = "cpu"
NUM_SEEDS = 3
MAX_EPOCHS = 300
PATIENCE = 50
HUBER_DELTA = 1.0

criterion = HuberLoss(delta=HUBER_DELTA)

ensemble_val_preds = []
ensemble_test_preds = []
seed_val_scores = []

for seed in range(NUM_SEEDS):
    print(f"\n{'='*50}")
    print(f"SEED {seed+1}/{NUM_SEEDS}")
    print(f"{'='*50}")

    torch.manual_seed(seed * 42 + 17)
    np.random.seed(seed * 42 + 17)

    model = MixtureOfExperts(
        input_dim=input_dim,
        num_experts=4,
        hidden_dims=[128, 64, 32],
        dropout_rate=0.2,
        gate_temperature=1.0,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999)
    )
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=5, total_epochs=MAX_EPOCHS, min_lr=1e-5, max_lr=1e-3
    )

    model, val_rmse = train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        device=device,
    )

    seed_val_scores.append(val_rmse)
    print(f"Seed {seed+1} best Val RMSE: {val_rmse:.4f}")

    model.eval()
    with torch.no_grad():
        val_pred_list = []
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            val_pred_list.append(preds.cpu().numpy())
        val_pred = np.concatenate(val_pred_list)

        test_pred_list = []
        for (X_batch,) in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            test_pred_list.append(preds.cpu().numpy())
        test_pred = np.concatenate(test_pred_list)

    ensemble_val_preds.append(val_pred)
    ensemble_test_preds.append(test_pred)

# ============================================================
# 9. ENSEMBLE AVERAGING & FINAL EVALUATION
# ============================================================
ensemble_val_pred = np.mean(ensemble_val_preds, axis=0)
ensemble_test_pred = np.mean(ensemble_test_preds, axis=0)

val_rmse_ensemble = np.sqrt(mean_squared_error(y_val_np, ensemble_val_pred))

print(f"\n{'='*50}")
print(f"ENSEMBLE RESULTS")
print(f"{'='*50}")
print(f"Individual seed Val RMSEs: {[f'{s:.4f}' for s in seed_val_scores]}")
print(f"Mean individual: {np.mean(seed_val_scores):.4f}")
print(f"Ensemble Val RMSE: {val_rmse_ensemble:.4f}")

# ============================================================
# 10. SAVE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)

submission = pd.DataFrame(
    {"id": range(len(ensemble_test_pred)), "projectPayment": ensemble_test_pred}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(
    f"Prediction stats: mean={ensemble_test_pred.mean():.4f}, std={ensemble_test_pred.std():.4f}, min={ensemble_test_pred.min():.4f}, max={ensemble_test_pred.max():.4f}"
)

# ============================================================
# FINAL OUTPUT
# ============================================================
score = val_rmse_ensemble
print(f"Final Validation Score: {score}")
