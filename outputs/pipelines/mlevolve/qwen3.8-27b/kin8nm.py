import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

target_col = "y"
feature_cols = [c for c in train_df.columns if c != target_col]
print(f"Feature columns: {feature_cols}")

# ============================================================
# 2. DATA CLEANING
# ============================================================
train_df = train_df.fillna(train_df.median(numeric_only=True))
test_df = test_df.fillna(train_df.median(numeric_only=True))

# ============================================================
# 3. CREATE TRAIN/VAL SPLIT (before feature engineering)
# ============================================================
train_features_raw = train_df[feature_cols].values
train_target = train_df[target_col].values
test_features_raw = test_df[feature_cols].values

X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    train_features_raw, train_target, test_size=0.15, random_state=42
)
print(
    f"Split: Train={X_train_raw.shape[0]}, Val={X_val_raw.shape[0]}, Test={test_features_raw.shape[0]}"
)


# ============================================================
# 4. PHYSICS-INFORMED FEATURE ENGINEERING
# ============================================================
def engineer_features(X_raw):
    """
    Create physics-informed features for 8-link robot arm forward kinematics.
    Features: original angles, cumulative angles, sin/cos of both,
    pairwise coupling terms, squared angles, normalized angles.
    """
    n_angles = X_raw.shape[1]
    features = {}

    # Original angles
    for i in range(n_angles):
        features[f"theta{i+1}"] = X_raw[:, i]

    # Cumulative angles
    cum_angles = np.cumsum(X_raw, axis=1)
    for i in range(n_angles):
        features[f"cum_theta{i+1}"] = cum_angles[:, i]

    # Sin and Cos of original angles
    for i in range(n_angles):
        features[f"sin_theta{i+1}"] = np.sin(X_raw[:, i])
        features[f"cos_theta{i+1}"] = np.cos(X_raw[:, i])

    # Sin and Cos of cumulative angles
    for i in range(n_angles):
        features[f"sin_cum{i+1}"] = np.sin(cum_angles[:, i])
        features[f"cos_cum{i+1}"] = np.cos(cum_angles[:, i])

    # Pairwise coupling terms
    for i in range(n_angles - 1):
        features[f"cos_coupling_{i+1}_{i+2}"] = np.cos(X_raw[:, i]) * np.cos(
            X_raw[:, i + 1]
        )
        features[f"sin_cos_cross_{i+1}_{i+2}"] = np.sin(X_raw[:, i]) * np.cos(
            X_raw[:, i + 1]
        )

    # Squared angles
    for i in range(n_angles):
        features[f"theta{i+1}_sq"] = X_raw[:, i] ** 2

    # Normalized angles
    for i in range(n_angles):
        features[f"theta{i+1}_norm"] = (X_raw[:, i] + np.pi) / (2 * np.pi)

    return pd.DataFrame(features)


X_train = engineer_features(X_train_raw)
X_val = engineer_features(X_val_raw)
X_test = engineer_features(test_features_raw)

print(f"Engineered features: {X_train.shape[1]}")


# ============================================================
# 5. MODEL ARCHITECTURE: Residual MLP
# ============================================================
class ResidualBlock(nn.Module):
    """Single residual block: Linear -> BN -> GELU -> Dropout -> Linear -> BN + skip"""

    def __init__(self, dim, dropout_rate=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class RobotArmFKNet(nn.Module):
    """
    Residual MLP for 8-link robot arm forward kinematics.
    Input projection -> N residual blocks -> Output head.
    """

    def __init__(self, input_dim=78, hidden_dim=256, n_blocks=6, dropout_rate=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout_rate) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.head(x)
        return x.squeeze(-1)


def create_model(input_dim=78, hidden_dim=256, n_blocks=6, dropout_rate=0.1):
    return RobotArmFKNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_blocks=n_blocks,
        dropout_rate=dropout_rate,
    )


def create_optimizer(model, lr=1e-3, weight_decay=1e-4):
    no_decay = [
        p
        for n, p in model.named_parameters()
        if "bn" in n.lower() or "bias" in n.lower()
    ]
    decay = [
        p
        for n, p in model.named_parameters()
        if "bn" not in n.lower() and "bias" not in n.lower()
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    return optimizer


def create_scheduler(optimizer, total_epochs, warmup_epochs=10, eta_min=1e-6):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return eta_min / (optimizer.defaults["lr"]) + (
            1 - eta_min / (optimizer.defaults["lr"])
        ) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# 6. TRAINING SETUP
# ============================================================
criterion = nn.MSELoss()
input_dim = X_train.shape[1]
print(f"Input dimension: {input_dim}")

# Convert to tensors
X_train_t = torch.FloatTensor(X_train.values)
X_val_t = torch.FloatTensor(X_val.values)
X_test_t = torch.FloatTensor(X_test.values)
y_train_t = torch.FloatTensor(y_train)
y_val_t = torch.FloatTensor(y_val)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
val_dataset = TensorDataset(X_val_t, y_val_t)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2)

model = create_model(input_dim=input_dim, hidden_dim=256, n_blocks=6, dropout_rate=0.1)
optimizer = create_optimizer(model, lr=1e-3, weight_decay=1e-4)

MAX_EPOCHS = 300
scheduler = create_scheduler(
    optimizer, total_epochs=MAX_EPOCHS, warmup_epochs=10, eta_min=1e-6
)

# ============================================================
# 7. TRAINING LOOP WITH EARLY STOPPING
# ============================================================
best_val_rmse = float("inf")
best_epoch = 0
best_model_state = None
patience = 30
patience_counter = 0
max_grad_norm = 1.0

print(f"\nStarting training: {MAX_EPOCHS} max epochs, patience={patience}")
print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

for epoch in range(1, MAX_EPOCHS + 1):
    # Training phase
    model.train()
    train_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        train_loss += loss.item()
        n_batches += 1

    train_loss /= n_batches

    # Validation phase
    model.eval()
    val_preds = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            preds = model(X_batch)
            val_preds.append(preds.numpy())

    val_preds = np.concatenate(val_preds)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

    # Log
    if epoch % 10 == 0 or epoch == 1:
        print(
            f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val RMSE: {val_rmse:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

    # Update best model
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_epoch = epoch
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    scheduler.step()

    if patience_counter >= patience:
        print(
            f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}, Best Val RMSE: {best_val_rmse:.6f}"
        )
        break

# ============================================================
# 8. LOAD BEST MODEL AND GENERATE PREDICTIONS
# ============================================================
model.load_state_dict(best_model_state)
model.eval()

with torch.no_grad():
    val_final_preds = model(X_val_t).numpy()
    test_preds = model(X_test_t).numpy()

final_val_rmse = np.sqrt(mean_squared_error(y_val, val_final_preds))
print(f"\nBest model loaded (epoch {best_epoch})")
print(f"Validation RMSE: {final_val_rmse:.6f}")

# ============================================================
# 9. SAVE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame({"id": range(len(test_preds)), "y": test_preds})
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")
print(
    f"Prediction stats: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}, min={test_preds.min():.4f}, max={test_preds.max():.4f}"
)

# ============================================================
# 10. FINAL SCORE
# ============================================================
score = final_val_rmse
print(f"Final Validation Score: {score}")
