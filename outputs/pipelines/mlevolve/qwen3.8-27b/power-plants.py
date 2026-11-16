import numpy as np
import pandas as pd
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Create output directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# ============================================================
# 2. IDENTIFY FEATURE TYPES
# ============================================================
target_col = "NAMPLT_CAP"
id_col = "id" if "id" in train.columns else None

# Separate target
y = train[target_col].copy()
train = train.drop(columns=[target_col])

# Identify feature columns
if id_col:
    train_id = train[id_col].copy()
    test_id = (
        test[id_col].copy() if id_col in test.columns else pd.Series(range(len(test)))
    )
    feature_cols = [c for c in train.columns if c != id_col]
else:
    train_id = pd.Series(range(len(train)))
    test_id = pd.Series(range(len(test)))
    feature_cols = list(train.columns)

# Classify features by type
cat_cols = (
    train[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()
)
num_cols = [c for c in feature_cols if c not in cat_cols]

print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
print(f"Numerical features ({len(num_cols)}): {num_cols}")

# ============================================================
# 3. CREATE TRAIN/VAL SPLIT
# ============================================================
train_data, val_data = train_test_split(train, test_size=0.15, random_state=42)
train_idx = train_data.index
val_idx = val_data.index

y_train = y.loc[train_idx]
y_val = y.loc[val_idx]

print(f"Train split: {len(train_idx)}, Val split: {len(val_idx)}")
print(f"Target stats - Train: mean={y_train.mean():.2f}, std={y_train.std():.2f}")

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================


def engineer_features(
    df,
    cat_cols,
    num_cols,
    fit=False,
    cat_encoders=None,
    num_medians=None,
    year_col=None,
    lat_col=None,
    lon_col=None,
    interaction_encoders=None,
):
    """Apply feature engineering. If fit=True, compute statistics from this df."""
    df = df.copy()

    # Handle missing values in numerical features
    if fit:
        num_medians = {}
        for c in num_cols:
            if c in df.columns:
                num_medians[c] = df[c].median() if df[c].notna().any() else 0.0
    else:
        num_medians = num_medians or {}

    for c in num_cols:
        if c in df.columns:
            med = num_medians.get(c, 0.0)
            missing_mask = df[c].isna()
            df[c] = df[c].fillna(med)
            df[f"{c}_missing"] = missing_mask.astype(int)

    # Handle missing values in categorical features
    if fit:
        cat_encoders = {}
        for c in cat_cols:
            if c in df.columns:
                df[c] = df[c].fillna("__MISSING__").astype(str)
                freq = df[c].value_counts()
                cat_encoders[c] = {k: i for i, k in enumerate(freq.index)}
                df[c] = df[c].map(cat_encoders[c]).astype(int)
            else:
                df[c] = -1
                cat_encoders[c] = {}
    else:
        cat_encoders = cat_encoders or {}
        for c in cat_cols:
            if c in df.columns:
                df[c] = df[c].fillna("__MISSING__").astype(str)
                encoder = cat_encoders.get(c, {})
                df[c] = df[c].map(encoder).fillna(-1).astype(int)
            else:
                df[c] = -1
    # Numerical feature transformations
    for c in num_cols:
        if c in df.columns:
            if df[c].min() >= 0 and df[c].skew() > 2:
                df[f"{c}_log"] = np.log1p(df[c].clip(lower=0))
            if df[c].std() > 0:
                df[f"{c}_z"] = (df[c] - df[c].mean()) / (df[c].std() + 1e-8)

    # Interaction features between categoricals
    if len(cat_cols) >= 2:
        if fit:
            interaction_encoders = {}
        else:
            interaction_encoders = interaction_encoders or {}
        for i in range(min(len(cat_cols), 3)):
            for j in range(i + 1, min(len(cat_cols), 5)):
                c1, c2 = cat_cols[i], cat_cols[j]
                if c1 in df.columns and c2 in df.columns:
                    df[f"{c1}_x_{c2}"] = df[c1].astype(str) + "_" + df[c2].astype(str)
                    if fit:
                        freq = df[f"{c1}_x_{c2}"].value_counts()
                        enc = {k: i for i, k in enumerate(freq.index)}
                        interaction_encoders[f"{c1}_x_{c2}"] = enc
                        df[f"{c1}_x_{c2}"] = (
                            df[f"{c1}_x_{c2}"].map(enc).fillna(-1).astype(int)
                        )
                    else:
                        enc = interaction_encoders.get(f"{c1}_x_{c2}", {})
                        df[f"{c1}_x_{c2}"] = (
                            df[f"{c1}_x_{c2}"].map(enc).fillna(-1).astype(int)
                        )

    # Ensure all categorical/interaction columns are integer type
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(int)
    for c in df.columns:
        if "_x_" in c:
            df[c] = df[c].astype(int)

    # Time-based features
    if year_col and year_col in df.columns:
        current_year = 2024
        df["plant_age"] = (current_year - df[year_col]).clip(lower=0)
        df["plant_decade"] = (df[year_col] // 10) * 10

    # Location-based features
    if lat_col and lon_col and lat_col in df.columns and lon_col in df.columns:
        df["lat_lon_interaction"] = df[lat_col] * df[lon_col]
        df["abs_lat"] = df[lat_col].abs()
        df["abs_lon"] = df[lon_col].abs()

    return df, cat_encoders, num_medians, interaction_encoders


# Identify special columns
year_col = None
lat_col = None
lon_col = None
for c in num_cols:
    cl = c.lower()
    if "year" in cl or "commission" in cl or "date" in cl or "start" in cl:
        year_col = c
    if "lat" in cl:
        lat_col = c
    if "lon" in cl or "long" in cl:
        lon_col = c

print(f"Identified special columns: year={year_col}, lat={lat_col}, lon={lon_col}")

# Apply feature engineering - FIT on training data only
train_processed, cat_encoders, num_medians, interaction_encoders = engineer_features(
    train_data,
    cat_cols,
    num_cols,
    fit=True,
    year_col=year_col,
    lat_col=lat_col,
    lon_col=lon_col,
)

val_processed, _, _, _ = engineer_features(
    val_data,
    cat_cols,
    num_cols,
    fit=False,
    cat_encoders=cat_encoders,
    num_medians=num_medians,
    year_col=year_col,
    lat_col=lat_col,
    lon_col=lon_col,
    interaction_encoders=interaction_encoders,
)

# Process test with fitted parameters
test_processed, _, _, _ = engineer_features(
    test,
    cat_cols,
    num_cols,
    fit=False,
    cat_encoders=cat_encoders,
    num_medians=num_medians,
    year_col=year_col,
    lat_col=lat_col,
    lon_col=lon_col,
    interaction_encoders=interaction_encoders,
)

# ============================================================
# 5. TARGET TRANSFORMATION
# ============================================================
y_train_log = np.log1p(y_train.clip(lower=0)).values.astype(np.float32)
y_val_log = np.log1p(y_val.clip(lower=0)).values.astype(np.float32)

print(
    f"Log-transformed target - Train: mean={y_train_log.mean():.4f}, std={y_train_log.std():.4f}"
)

# ============================================================
# 6. PREPARE FINAL FEATURE MATRICES
# ============================================================
final_feature_cols = list(train_processed.columns)

# Ensure test has same columns
for c in final_feature_cols:
    if c not in test_processed.columns:
        test_processed[c] = 0

test_processed = test_processed[final_feature_cols]
val_processed = val_processed[final_feature_cols]

X_train = train_processed[final_feature_cols].values.astype(np.float32)
X_val = val_processed[final_feature_cols].values.astype(np.float32)
X_test = test_processed[final_feature_cols].values.astype(np.float32)

y_train_orig = y_train.values.astype(np.float32)
y_val_orig = y_val.values.astype(np.float32)

print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

# ============================================================
# 7. IDENTIFY CATEGORICAL FEATURES FOR EMBEDDING
# ============================================================
cat_feature_indices = []
cat_feature_cardinalities = []

for i, col in enumerate(final_feature_cols):
    if col in cat_cols:
        cat_feature_indices.append(i)
        # Use actual max encoded value from processed train data to ensure embedding is large enough
        max_val = int(np.max(train_processed[col].values)) if len(train_processed) > 0 else 0
        cardinality = max_val + 2  # +1 for -1 sentinel (clamped to 0), +1 buffer
        cat_feature_cardinalities.append(min(cardinality, 500))
    elif "_x_" in col:
        # Interaction features from categoricals
        cat_feature_indices.append(i)
        max_val = int(np.max(train_processed[col].values)) if len(train_processed) > 0 else 0
        cardinality = max_val + 2
        cat_feature_cardinalities.append(min(cardinality, 500))

# Fallback: identify low-cardinality columns as categorical
if len(cat_feature_indices) == 0:
    for i, col in enumerate(final_feature_cols):
        unique_count = np.unique(X_train[:, i]).size
        if unique_count < 50:
            cat_feature_indices.append(i)
            cat_feature_cardinalities.append(min(unique_count + 1, 50))

num_feature_indices = [
    i for i in range(X_train.shape[1]) if i not in cat_feature_indices
]

print(f"Total features: {X_train.shape[1]}")
print(f"Categorical features: {len(cat_feature_indices)}")
print(f"Numerical features: {len(num_feature_indices)}")

# ============================================================
# 8. NORMALIZATION (fit on train only)
# ============================================================
num_mean = X_train[:, num_feature_indices].mean(axis=0)
num_std = X_train[:, num_feature_indices].std(axis=0) + 1e-8


def normalize_features(X):
    X = X.copy()
    X[:, num_feature_indices] = (X[:, num_feature_indices] - num_mean) / num_std
    return X


X_train_norm = normalize_features(X_train)
X_val_norm = normalize_features(X_val)
X_test_norm = normalize_features(X_test)

# ============================================================
# 9. MODEL ARCHITECTURE: Mixture Density Network
# ============================================================
NUM_MIXTURE_COMPONENTS = 4


class MDNHead(nn.Module):
    """Mixture Density Network head."""

    def __init__(self, input_dim, num_components=4):
        super().__init__()
        self.num_components = num_components
        self.weight_net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, num_components)
        )
        self.mean_net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, num_components)
        )
        self.log_std_net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, num_components)
        )

    def forward(self, x):
        weights = F.softmax(self.weight_net(x), dim=-1)
        means = self.mean_net(x)
        log_stds = torch.clamp(self.log_std_net(x), min=-5, max=2)
        return weights, means, log_stds

    def predict_mean(self, x):
        weights, means, log_stds = self.forward(x)
        return (weights * means).sum(dim=-1)

    def negative_log_likelihood(self, x, y):
        weights, means, log_stds = self.forward(x)
        stds = torch.exp(log_stds)
        log_probs = -0.5 * (
            (y.unsqueeze(-1) - means) ** 2 / (stds**2) + torch.log(2 * np.pi * stds**2)
        )
        log_weighted_probs = torch.log(weights + 1e-10) + log_probs
        nll = -torch.logsumexp(log_weighted_probs, dim=-1)
        return nll.mean()


class PowerPlantMDN(nn.Module):
    """Neural network with embedding layers for categoricals and MDN head."""

    def __init__(
        self,
        num_features,
        cat_indices,
        cat_cardinalities,
        embedding_dim=8,
        hidden_dims=[128, 64, 32],
        num_components=4,
        dropout_rate=0.2,
    ):
        super().__init__()
        self.cat_indices = cat_indices
        self.num_indices = [i for i in range(num_features) if i not in cat_indices]

        self.embeddings = nn.ModuleDict()
        total_input_dim = len(self.num_indices)

        for idx, card in zip(cat_indices, cat_cardinalities):
            emb_dim = min(embedding_dim, max(2, card // 2))
            self.embeddings[f"cat_{idx}"] = nn.Embedding(card + 1, emb_dim)
            total_input_dim += emb_dim

        layers = []
        prev_dim = total_input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.mdn_head = MDNHead(prev_dim, num_components)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, x):
        num_x = x[:, self.num_indices]
        emb_list = [num_x]
        for idx in self.cat_indices:
            cat_val = x[:, idx].long().clamp(min=0)
            # Clamp to valid range to prevent index out of bounds
            max_idx = self.embeddings[f"cat_{idx}"].num_embeddings - 1
            cat_val = cat_val.clamp(max=max_idx)
            emb = self.embeddings[f"cat_{idx}"](cat_val)
            emb_list.append(emb)
        combined = torch.cat(emb_list, dim=-1)
        h = self.backbone(combined)
        return h

    def predict(self, x):
        h = self.forward(x)
        return self.mdn_head.predict_mean(h)

    def nll_loss(self, x, y):
        h = self.forward(x)
        return self.mdn_head.negative_log_likelihood(h, y)


# ============================================================
# 10. INSTANTIATE MODEL
# ============================================================
num_features = X_train_norm.shape[1]
model = PowerPlantMDN(
    num_features=num_features,
    cat_indices=cat_feature_indices,
    cat_cardinalities=cat_feature_cardinalities,
    embedding_dim=8,
    hidden_dims=[128, 64, 32],
    num_components=NUM_MIXTURE_COMPONENTS,
    dropout_rate=0.2,
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


# ============================================================
# 11. DATASETS AND DATALOADERS
# ============================================================
class DualTargetDataset(Dataset):
    def __init__(self, X, y_log, y_orig):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_log = torch.tensor(y_log, dtype=torch.float32)
        self.y_orig = torch.tensor(y_orig, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_log[idx], self.y_orig[idx]


class SingleTargetDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = DualTargetDataset(X_train_norm, y_train_log, y_train_orig)
val_dataset = SingleTargetDataset(X_val_norm, y_val_log)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=2)


# ============================================================
# 12. HYBRID LOSS: NLL + Original-scale MSE
# ============================================================
def hybrid_loss(model, X_batch, y_log_batch, y_orig_batch, alpha=0.3):
    h = model(X_batch)
    nll = model.mdn_head.negative_log_likelihood(h, y_log_batch)
    pred_log = model.mdn_head.predict_mean(h)
    pred_orig = torch.expm1(pred_log)
    pred_orig = torch.clamp(pred_orig, min=0.0)
    mse_orig = F.mse_loss(pred_orig, y_orig_batch)
    return nll + alpha * mse_orig


# ============================================================
# 13. TRAINING LOOP
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
num_epochs = 150
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=1e-6
)

best_val_rmse = float("inf")
best_model_state = None
patience = 20
patience_counter = 0

print(f"Starting training | Epochs: {num_epochs} | Patience: {patience}")

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    num_batches = 0

    for X_batch, y_log_batch, y_orig_batch in train_loader:
        X_batch = X_batch.to(device)
        y_log_batch = y_log_batch.to(device)
        y_orig_batch = y_orig_batch.to(device)

        optimizer.zero_grad()
        loss = hybrid_loss(model, X_batch, y_log_batch, y_orig_batch, alpha=0.3)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        num_batches += 1

    train_loss /= num_batches
    scheduler.step()

    # Validation
    model.eval()
    val_preds_log = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            pred_log = model.predict(X_batch)
            val_preds_log.append(pred_log.cpu().numpy())

    val_preds_log = np.concatenate(val_preds_log)
    val_preds_orig = np.expm1(val_preds_log)
    val_preds_orig = np.maximum(val_preds_orig, 0.0)
    val_rmse = np.sqrt(mean_squared_error(y_val_orig, val_preds_orig))

    if (epoch + 1) % 5 == 0 or epoch == 0:
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {train_loss:.4f} | Val RMSE: {val_rmse:.4f} | LR: {lr_now:.2e}"
        )

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1} | Best Val RMSE: {best_val_rmse:.4f}")
        break

# ============================================================
# 14. LOAD BEST MODEL AND GENERATE FINAL PREDICTIONS
# ============================================================
model.load_state_dict(best_model_state)
model.eval()

# Validation predictions
with torch.no_grad():
    X_val_tensor = torch.tensor(X_val_norm, dtype=torch.float32).to(device)
    val_pred_log = model.predict(X_val_tensor).cpu().numpy()

val_pred_orig = np.expm1(val_pred_log)
val_pred_orig = np.maximum(val_pred_orig, 0.0)
final_rmse = np.sqrt(mean_squared_error(y_val_orig, val_pred_orig))
print(f"Validation RMSE (official metric): {final_rmse:.4f}")

# Test predictions
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32).to(device)
    test_pred_log = model.predict(X_test_tensor).cpu().numpy()

test_pred_orig = np.expm1(test_pred_log)
test_pred_orig = np.maximum(test_pred_orig, 0.0)

print(
    f"Test predictions - mean: {test_pred_orig.mean():.2f}, std: {test_pred_orig.std():.2f}"
)

# ============================================================
# 15. SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame({"id": test_id.astype(int), "NAMPLT_CAP": test_pred_orig})
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape[0]} rows")

# ============================================================
# 16. FINAL SCORE
# ============================================================
score = final_rmse
print(f"Final Validation Score: {score}")