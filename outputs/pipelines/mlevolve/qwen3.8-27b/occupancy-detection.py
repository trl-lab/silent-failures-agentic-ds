import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")

# Create directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(f"Columns: {train.columns.tolist()}")
print(
    f"Train date range: {pd.to_datetime(train['date']).min()} → {pd.to_datetime(train['date']).max()}"
)
print(
    f"Test date range: {pd.to_datetime(test['date']).min()} → {pd.to_datetime(test['date']).max()}"
)
print(f"Target balance (train): {train['Occupancy'].mean():.4f}")

# Parse and sort by date (critical for time-series features)
train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])
train = train.sort_values("date").reset_index(drop=True)
test = test.sort_values("date").reset_index(drop=True)

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
sensors = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]


def engineer_features(df):
    """Create temporal and interaction features from sensor readings."""
    df = df.copy()

    # --- Ensure sensor columns are numeric (fix string values) ---
    for s in sensors:
        df[s] = pd.to_numeric(df[s], errors="coerce")

    # --- Cyclical time features ---
    df["hour"] = df["date"].dt.hour
    df["minute"] = df["date"].dt.minute
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.int8)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # --- Lag features (captures temporal momentum) ---
    for s in sensors:
        for lag in [1, 2, 3, 5, 10, 15, 30]:
            df[f"{s}_lag{lag}"] = df[s].shift(lag)

    # --- Rolling statistics (captures local trends) ---
    for s in sensors:
        for w in [5, 10, 30, 60]:
            roll = df[s].rolling(w, min_periods=1)
            df[f"{s}_rmean{w}"] = roll.mean()
            df[f"{s}_rstd{w}"] = roll.std().fillna(0)
            df[f"{s}_rmin{w}"] = roll.min()
            df[f"{s}_rmax{w}"] = roll.max()
            df[f"{s}_rrange{w}"] = df[f"{s}_rmax{w}"] - df[f"{s}_rmin{w}"]

    # --- Rate of change (captures acceleration/deceleration) ---
    for s in sensors:
        for d in [1, 5, 10, 30]:
            df[f"{s}_diff{d}"] = df[s].diff(d)

    # --- Inter-sensor interactions (domain-informed) ---
    df["CO2_x_Light"] = df["CO2"] * df["Light"]
    df["CO2_div_Light"] = df["CO2"] / (df["Light"] + 1)
    df["CO2_x_Temp"] = df["CO2"] * df["Temperature"]
    df["Temp_x_Hum"] = df["Temperature"] * df["Humidity"]
    df["CO2_minus_Hum"] = df["CO2"] - df["Humidity"]
    df["Light_minus_Temp"] = df["Light"] - df["Temperature"]
    df["CO2_sq"] = df["CO2"] ** 2
    df["Light_sq"] = df["Light"] ** 2

    # --- Relative deviation from local baseline ---
    for s in ["CO2", "Light", "Temperature", "Humidity"]:
        base = df[f"{s}_rmean30"]
        std = df[f"{s}_rstd30"] + 1e-8
        df[f"{s}_z30"] = (df[s] - base) / std
        df[f"{s}_dev30"] = df[s] - base

    # --- Composite occupancy proxy features ---
    df["CO2_z_x_Light_z"] = df["CO2_z30"] * df["Light_z30"]
    df["CO2_dev_x_Light_dev"] = df["CO2_dev30"] * df["Light_dev30"]

    # --- Fill NaN from shifts/rolling/diff ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(0)

    return df


# ============================================================
# 3. TIME-BASED SPLIT (respect temporal ordering) — BEFORE feature engineering
# ============================================================
split_idx = int(len(train) * 0.85)
train_split = train.iloc[:split_idx].copy()
val_split = train.iloc[split_idx:].copy()

print(f"\nSplit: train={len(train_split)}, val={len(val_split)}, test={len(test)}")
print(f"Val date range: {val_split['date'].min()} → {val_split['date'].max()}")
print(f"Val target balance: {val_split['Occupancy'].mean():.4f}")

print("\nEngineering features...")
train_split = engineer_features(train_split)
val_split = engineer_features(val_split)
test = engineer_features(test)

# ============================================================
# 4. PREPARE FEATURE MATRICES
# ============================================================
exclude_cols = ["id", "date", "Occupancy"]
feature_cols = [c for c in train_split.columns if c not in exclude_cols]
print(f"Total features: {len(feature_cols)}")

X_train = train_split[feature_cols].values.astype(np.float32)
y_train = train_split["Occupancy"].values.astype(np.float32)
X_val = val_split[feature_cols].values.astype(np.float32)
y_val = val_split["Occupancy"].values.astype(np.float32)
X_test = test[feature_cols].values.astype(np.float32)

# Safety: replace any remaining inf/nan
for arr in [X_train, X_val, X_test]:
    arr[~np.isfinite(arr)] = 0.0

print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
print(f"Class balance - Train: {y_train.mean():.4f}, Val: {y_val.mean():.4f}")


# ============================================================
# 5. MODEL ARCHITECTURE: Gated MLP with Feature Selection
# ============================================================
class GatedMLP(nn.Module):
    """
    MLP with feature-wise gating (inspired by TabNet's feature selection).
    Architecture:
      - Input layer with feature-wise linear projection
      - Gating mechanism: each feature gets a learned importance weight
      - Multiple hidden layers with batch norm + dropout
      - Final sigmoid output for binary classification
    """

    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        super(GatedMLP, self).__init__()

        # Feature selection: learn importance weights for each input feature
        self.feature_gate = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.Sigmoid()
        )

        # Main MLP pathway
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        self.mlp = nn.Sequential(*layers)

        # Output layer
        self.output = nn.Linear(prev_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Feature gating: multiply input by learned importance
        gate = self.feature_gate(x)
        x_gated = x * gate

        # MLP forward pass
        out = self.mlp(x_gated)
        out = self.output(out).squeeze(-1)
        return out  # raw logit (before sigmoid)


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    Focuses training on hard-to-classify samples (boundary cases).
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=0.5, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        p_t = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(
            targets == 1,
            torch.tensor(self.alpha, device=logits.device),
            torch.tensor(1 - self.alpha, device=logits.device),
        )
        loss = -alpha_t * (1 - p_t) ** self.gamma * torch.log(p_t + 1e-8)
        return loss.mean()


# ============================================================
# 6. DEFINE MODEL, LOSS, AND OPTIMIZER
# ============================================================
input_dim = X_train.shape[1]

model = GatedMLP(input_dim=input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = FocalLoss(alpha=0.5, gamma=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)

# ============================================================
# 7. TRAINING LOOP WITH EARLY STOPPING
# ============================================================
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=2)

best_val_auc = 0.0
best_model_state = None
patience_counter = 0
patience = 20
max_epochs = 300
history = []

for epoch in range(max_epochs):
    model.train()
    epoch_loss = 0.0
    n_batches = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1

    scheduler.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)
        val_probs = torch.sigmoid(val_logits).numpy()
        val_auc = roc_auc_score(y_val, val_probs)

    avg_loss = epoch_loss / max(n_batches, 1)
    current_lr = optimizer.param_groups[0]["lr"]
    history.append(
        {"epoch": epoch + 1, "loss": avg_loss, "val_auc": val_auc, "lr": current_lr}
    )

    print(
        f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.6f} | LR: {current_lr:.6f}"
    )

    # Track best model
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} (best AUC: {best_val_auc:.6f})")
            break

# ============================================================
# 8. LOAD BEST MODEL & GENERATE PREDICTIONS
# ============================================================
model.load_state_dict(best_model_state)
model.eval()

with torch.no_grad():
    val_logits = model(X_val_t)
    val_probs = torch.sigmoid(val_logits).numpy()

    test_logits = model(X_test_t)
    test_probs = torch.sigmoid(test_logits).numpy()

# ============================================================
# 9. EVALUATE (Official Metric: ROC-AUC)
# ============================================================
score = roc_auc_score(y_val, val_probs)
print(f"\nBest Validation ROC-AUC: {score:.6f}")
best_epoch = history[np.argmax([h["val_auc"] for h in history])]["epoch"]
print(f"Best epoch: {best_epoch}")

# ============================================================
# 10. SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame({"id": range(len(test_probs)), "Occupancy": test_probs})
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {len(submission)} rows")

# Save model state for future use
torch.save(best_model_state, "./working/best_model_state.pth")
pd.DataFrame(history).to_csv("./working/training_history.csv", index=False)

print(f"Final Validation Score: {score}")