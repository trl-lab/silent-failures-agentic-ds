import numpy as np
import pandas as pd
import os
import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# ============================================================
# SETUP
# ============================================================
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("Loading data...")
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# ============================================================
# 2. CREATE TRAIN/VALIDATION SPLIT
# ============================================================
train["target_bin"] = pd.cut(
    train["deal_probability"], bins=[-0.01, 0.05, 0.5, 1.01], labels=[0, 1, 2]
)
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
train_indices, val_indices = next(iter(skf.split(train, train["target_bin"])))

train_split = (
    train.iloc[train_indices].drop(columns=["target_bin"]).reset_index(drop=True)
)
val_split = train.iloc[val_indices].drop(columns=["target_bin"]).reset_index(drop=True)
print(f"Train split: {len(train_split)}, Val split: {len(val_split)}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================


def engineer_features(
    df,
    is_train=True,
    target_enc_maps=None,
    user_stats=None,
    global_mean=None,
    global_smooth=None,
):
    """Engineer features for a dataframe."""
    df = df.copy()

    # --- Price features ---
    df["price_is_nan"] = df["price"].isna().astype(int)
    df["price_log"] = np.log1p(df["price"].fillna(0))
    df["price_is_zero"] = (df["price"] == 0).astype(int)
    df["price_is_high"] = (df["price"] > 1000000).astype(int)

    # --- Image features ---
    df["image_is_nan"] = df["image_top_1"].isna().astype(int)
    df["image_top_1_log"] = np.log1p(df["image_top_1"].fillna(0))

    # --- Text features ---
    df["title_len"] = df["title"].fillna("").str.len()
    df["desc_len"] = df["description"].fillna("").str.len()
    df["desc_word_count"] = df["description"].fillna("").str.split().str.len()
    df["title_word_count"] = df["title"].fillna("").str.split().str.len()
    df["has_description"] = (df["description"].fillna("").str.len() > 0).astype(int)
    df["desc_has_numbers"] = (
        df["description"].fillna("").str.contains(r"\d").astype(int)
    )
    df["title_has_numbers"] = df["title"].fillna("").str.contains(r"\d").astype(int)
    df["desc_upper_ratio"] = (
        df["description"]
        .fillna("")
        .apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
    )

    # --- Date features ---
    df["activation_date"] = pd.to_datetime(df["activation_date"])
    df["day_of_week"] = df["activation_date"].dt.dayofweek
    df["day_of_month"] = df["activation_date"].dt.day
    df["month"] = df["activation_date"].dt.month
    df["year"] = df["activation_date"].dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # --- Item sequence ---
    df["item_seq_number_log"] = np.log1p(df["item_seq_number"])

    # --- Categorical fillna ---
    cat_cols = [
        "category_name",
        "parent_category_name",
        "user_type",
        "param_1",
        "param_2",
        "param_3",
        "region",
        "city",
    ]
    for col in cat_cols:
        df[col] = df[col].fillna("missing")

    # Target encoding for high-cardinality features
    if is_train:
        target_enc_maps = {}
        smoothing = 10

        for col in [
            "category_name",
            "parent_category_name",
            "region",
            "city",
            "user_type",
            "param_1",
            "param_2",
            "param_3",
        ]:
            agg = df.groupby(col)["deal_probability"].agg(["mean", "count"])
            smooth_mean = (agg["mean"] * agg["count"] + global_mean * smoothing) / (
                agg["count"] + smoothing
            )
            target_enc_maps[col] = smooth_mean.to_dict()

        # User-level target encoding
        user_agg = df.groupby("user_id")["deal_probability"].agg(["mean", "count"])
        user_smooth = (
            user_agg["mean"] * user_agg["count"] + global_mean * smoothing
        ) / (user_agg["count"] + smoothing)
        target_enc_maps["user_id"] = user_smooth.to_dict()

        # User stats
        user_stats = (
            df.groupby("user_id")
            .agg(
                user_item_count=("item_id", "count"),
                user_avg_price_log=("price_log", "mean"),
                user_avg_desc_len=("desc_len", "mean"),
                user_avg_title_len=("title_len", "mean"),
            )
            .to_dict("index")
        )

        # Category-level stats
        cat_stats = (
            df.groupby("category_name")
            .agg(
                cat_item_count=("item_id", "count"),
                cat_avg_price_log=("price_log", "mean"),
            )
            .to_dict("index")
        )
        target_enc_maps["cat_stats"] = cat_stats

        # Region-level stats
        region_stats = (
            df.groupby("region")
            .agg(
                region_item_count=("item_id", "count"),
                region_avg_price_log=("price_log", "mean"),
            )
            .to_dict("index")
        )
        target_enc_maps["region_stats"] = region_stats

        # City-level stats
        city_stats = (
            df.groupby("city")
            .agg(
                city_item_count=("item_id", "count"),
                city_avg_price_log=("price_log", "mean"),
            )
            .to_dict("index")
        )
        target_enc_maps["city_stats"] = city_stats

    # Apply target encodings
    if target_enc_maps is not None:
        for col in [
            "category_name",
            "parent_category_name",
            "region",
            "city",
            "user_type",
            "param_1",
            "param_2",
            "param_3",
        ]:
            df[f"{col}_target_enc"] = (
                df[col].map(target_enc_maps.get(col, {})).fillna(global_mean)
            )

        df["user_id_target_enc"] = (
            df["user_id"].map(target_enc_maps.get("user_id", {})).fillna(global_mean)
        )

        cat_stats = target_enc_maps.get("cat_stats", {})
        df["cat_item_count"] = (
            df["category_name"]
            .map({k: v["cat_item_count"] for k, v in cat_stats.items()})
            .fillna(0)
        )
        df["cat_avg_price_log"] = (
            df["category_name"]
            .map({k: v["cat_avg_price_log"] for k, v in cat_stats.items()})
            .fillna(0)
        )

        region_stats = target_enc_maps.get("region_stats", {})
        df["region_item_count"] = (
            df["region"]
            .map({k: v["region_item_count"] for k, v in region_stats.items()})
            .fillna(0)
        )
        df["region_avg_price_log"] = (
            df["region"]
            .map({k: v["region_avg_price_log"] for k, v in region_stats.items()})
            .fillna(0)
        )

        city_stats = target_enc_maps.get("city_stats", {})
        df["city_item_count"] = (
            df["city"]
            .map({k: v["city_item_count"] for k, v in city_stats.items()})
            .fillna(0)
        )
        df["city_avg_price_log"] = (
            df["city"]
            .map({k: v["city_avg_price_log"] for k, v in city_stats.items()})
            .fillna(0)
        )

    # User stats
    if user_stats is not None:
        df["user_item_count"] = (
            df["user_id"]
            .map({k: v["user_item_count"] for k, v in user_stats.items()})
            .fillna(1)
        )
        df["user_avg_price_log"] = (
            df["user_id"]
            .map({k: v["user_avg_price_log"] for k, v in user_stats.items()})
            .fillna(0)
        )
        df["user_avg_desc_len"] = (
            df["user_id"]
            .map({k: v["user_avg_desc_len"] for k, v in user_stats.items()})
            .fillna(0)
        )
        df["user_avg_title_len"] = (
            df["user_id"]
            .map({k: v["user_avg_title_len"] for k, v in user_stats.items()})
            .fillna(0)
        )

    # Frequency encoding
    if is_train:
        for col in [
            "category_name",
            "parent_category_name",
            "user_type",
            "param_1",
            "param_2",
            "param_3",
            "region",
            "city",
        ]:
            freq = df[col].value_counts(normalize=True)
            target_enc_maps[f"{col}_freq"] = freq.to_dict()

    if target_enc_maps is not None:
        for col in [
            "category_name",
            "parent_category_name",
            "user_type",
            "param_1",
            "param_2",
            "param_3",
            "region",
            "city",
        ]:
            df[f"{col}_freq"] = (
                df[col].map(target_enc_maps.get(f"{col}_freq", {})).fillna(0)
            )

    # Interaction features
    df["price_per_desc_word"] = df["price_log"] / (df["desc_word_count"].fillna(1) + 1)
    df["title_to_desc_ratio"] = df["title_len"] / (df["desc_len"] + 1)
    df["user_count_x_category_enc"] = (
        df["user_item_count"] * df["category_name_target_enc"]
    )

    # Drop original high-cardinality string columns and categorical columns
    drop_cols = [
        "item_id",
        "user_id",
        "image",
        "title",
        "description",
        "deal_probability",
        "activation_date",
        "category_name",
        "parent_category_name",
        "user_type",
        "param_1",
        "param_2",
        "param_3",
        "region",
        "city",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df, target_enc_maps, user_stats


# Compute global statistics (from train_split only to avoid leakage)
global_mean = train_split["deal_probability"].mean()
global_smooth = 10
print(f"Global mean deal_probability: {global_mean:.4f}")

# Engineer features
print("Engineering features for train split...")
train_feat, target_enc_maps, user_stats = engineer_features(
    train_split,
    is_train=True,
    target_enc_maps=None,
    user_stats=None,
    global_mean=global_mean,
    global_smooth=global_smooth,
)

print("Engineering features for validation split...")
val_feat, _, _ = engineer_features(
    val_split,
    is_train=False,
    target_enc_maps=target_enc_maps,
    user_stats=user_stats,
    global_mean=global_mean,
    global_smooth=global_smooth,
)

print("Engineering features for test split...")
test_feat, _, _ = engineer_features(
    test,
    is_train=False,
    target_enc_maps=target_enc_maps,
    user_stats=user_stats,
    global_mean=global_mean,
    global_smooth=global_smooth,
)

# Finalize feature sets
common_cols = [
    c for c in train_feat.columns if c in val_feat.columns and c in test_feat.columns
]
train_feat = train_feat[common_cols]
val_feat = val_feat[common_cols]
test_feat = test_feat[common_cols]
print(f"Final feature count: {len(common_cols)}")

# Save processed data
train_feat.to_parquet("./working/train_feat.parquet")
val_feat.to_parquet("./working/val_feat.parquet")
test_feat.to_parquet("./working/test_feat.parquet")

val_target = val_split["deal_probability"].values
train_target = train_split["deal_probability"].values
np.save("./working/val_target.npy", val_target)

# ============================================================
# 4. QUICK LIGHTGBM BASELINE
# ============================================================
print("\nRunning quick LightGBM validation...")
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_jobs": -1,
}

dtrain = lgb.Dataset(train_feat.values, label=train_target)
dval = lgb.Dataset(val_feat.values, label=val_target, reference=dtrain)

lgb_model = lgb.train(
    lgb_params,
    dtrain,
    num_boost_round=3000,
    valid_sets=[dval],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
)

lgb_val_pred = np.clip(
    lgb_model.predict(val_feat.values, num_iteration=lgb_model.best_iteration), 0, 1
)
lgb_rmse = np.sqrt(mean_squared_error(val_target, lgb_val_pred))
print(f"LightGBM Validation RMSE: {lgb_rmse:.6f} (best_iter={lgb_model.best_iteration})")

# Retrain LightGBM on full training data with best iteration count
print("Retraining LightGBM on full training data...")
full_feat = pd.concat([train_feat, val_feat], axis=0).reset_index(drop=True)
full_target = np.concatenate([train_target, val_target])
dtrain_full = lgb.Dataset(full_feat.values, label=full_target)
lgb_model_full = lgb.train(
    lgb_params,
    dtrain_full,
    num_boost_round=lgb_model.best_iteration,
    valid_sets=[dtrain_full],
    callbacks=[lgb.log_evaluation(500)],
)
print(f"Full LightGBM model trained with {lgb_model.best_iteration} iterations")

# ============================================================
# 5. FEATURE NORMALIZATION FOR NEURAL NETWORK
# ============================================================
feature_mean = train_feat.mean().values
feature_std = train_feat.std().values
feature_std[feature_std == 0] = 1.0

X_train = ((train_feat.values - feature_mean) / feature_std).astype(np.float32)
X_val = ((val_feat.values - feature_mean) / feature_std).astype(np.float32)
X_test = ((test_feat.values - feature_mean) / feature_std).astype(np.float32)

X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

print(f"Feature mean range: [{feature_mean.min():.3f}, {feature_mean.max():.3f}]")
print(f"Feature std range: [{feature_std.min():.3f}, {feature_std.max():.3f}]")

# ============================================================
# 6. MODEL DEFINITION: Bimodal Mixture-of-Experts Network
# ============================================================


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        h = self.norm(x)
        h = F.gelu(h)
        h = self.linear1(h)
        h = self.dropout(h)
        h = self.linear2(h)
        return residual + h


class BimodalMoENet(nn.Module):
    def __init__(
        self, input_dim, hidden_dim=256, n_blocks=4, dropout=0.15, expert_dim=64
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout=dropout) for _ in range(n_blocks)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.expert_low = nn.Sequential(
            nn.Linear(hidden_dim, expert_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(expert_dim, 1),
        )
        self.expert_high = nn.Sequential(
            nn.Linear(hidden_dim, expert_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(expert_dim, 1),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, expert_dim), nn.GELU(), nn.Linear(expert_dim, 2)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        gate_logits = self.gate(h)
        gate_weights = F.softmax(gate_logits, dim=-1)
        low_pred = self.expert_low(h)
        high_pred = self.expert_high(h)
        pred = gate_weights[:, 0:1] * low_pred + gate_weights[:, 1:2] * high_pred
        return torch.sigmoid(pred)


class BimodalMSELoss(nn.Module):
    def __init__(self, confidence_weight=0.01):
        super().__init__()
        self.confidence_weight = confidence_weight

    def forward(self, pred, target, gate_weights=None):
        pred = pred.squeeze(-1)
        target = target.squeeze(-1)
        mse_loss = F.mse_loss(pred, target)
        if gate_weights is not None:
            gate_entropy = (
                -(gate_weights * (gate_weights + 1e-8).log()).sum(dim=-1).mean()
            )
            confidence_loss = gate_entropy
        else:
            confidence_loss = torch.tensor(0.0, device=pred.device)
        total_loss = mse_loss + self.confidence_weight * confidence_loss
        return total_loss, mse_loss


# ============================================================
# 7. DATASET & DATALOADER
# ============================================================


class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# ============================================================
# 8. INITIALIZE MODEL, LOSS, OPTIMIZER
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

input_dim = X_train.shape[1]
HIDDEN_DIM = 256
N_BLOCKS = 4
DROPOUT = 0.15
EXPERT_DIM = 64
BATCH_SIZE = 4096
EPOCHS = 40
LR = 3e-3

model = BimodalMoENet(
    input_dim=input_dim,
    hidden_dim=HIDDEN_DIM,
    n_blocks=N_BLOCKS,
    dropout=DROPOUT,
    expert_dim=EXPERT_DIM,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

criterion = BimodalMSELoss(confidence_weight=0.01)

# Optimizer with differential learning rates
backbone_params = []
head_params = []
for name, param in model.named_parameters():
    if "expert_low" in name or "expert_high" in name or "gate" in name:
        head_params.append(param)
    else:
        backbone_params.append(param)

param_groups = [
    {"params": backbone_params, "lr": LR, "weight_decay": 1e-4},
    {"params": head_params, "lr": LR * 2.0, "weight_decay": 5e-5},
]
optimizer = torch.optim.AdamW(param_groups, lr=LR, weight_decay=1e-4)


def lr_lambda(epoch):
    warmup_epochs = 3
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(EPOCHS - warmup_epochs, 1)
    return 0.5 * (1 + np.cos(np.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Dataloaders
train_dataset = TabularDataset(X_train, train_target)
val_dataset = TabularDataset(X_val, val_target)
test_dataset = TabularDataset(X_test)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)

# ============================================================
# 9. TRAINING LOOP
# ============================================================
PATIENCE = 10
best_val_rmse = float("inf")
best_model_state = None
patience_counter = 0
training_history = []

print(f"\n{'='*70}")
print(f"TRAINING BimodalMoENet | Max epochs: {EPOCHS} | Patience: {PATIENCE}")
print(f"{'='*70}")
print(
    f"{'Epoch':>5} | {'Train Loss':>10} | {'Val RMSE':>10} | {'Gate High%':>10} | {'LR':>10} | {'Time':>6}"
)
print(f"{'-'*70}")

start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()

    # --- Training phase ---
    model.train()
    train_loss_sum = 0.0
    n_batches = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()

        h = model.input_proj(X_batch)
        for block in model.blocks:
            h = block(h)
        h = model.final_norm(h)

        gate_logits = model.gate(h)
        gate_weights = F.softmax(gate_logits, dim=-1)

        low_pred = model.expert_low(h)
        high_pred = model.expert_high(h)
        pred = gate_weights[:, 0:1] * low_pred + gate_weights[:, 1:2] * high_pred
        pred = torch.sigmoid(pred)

        loss, mse_loss = criterion(pred, y_batch, gate_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss_sum += mse_loss.item()
        n_batches += 1

    # --- Validation phase ---
    model.eval()
    val_preds = []
    gate_probs_high = []

    with torch.no_grad():
        for X_batch in val_loader:
            if isinstance(X_batch, (tuple, list)):
                X_batch = X_batch[0]
            X_batch = X_batch.to(device)

            h = model.input_proj(X_batch)
            for block in model.blocks:
                h = block(h)
            h = model.final_norm(h)

            gate_logits = model.gate(h)
            gate_weights = F.softmax(gate_logits, dim=-1)
            gate_probs_high.append(gate_weights[:, 1].cpu().numpy())

            low_pred = model.expert_low(h)
            high_pred = model.expert_high(h)
            pred = gate_weights[:, 0:1] * low_pred + gate_weights[:, 1:2] * high_pred
            pred = torch.sigmoid(pred)
            val_preds.append(pred.cpu().numpy())

    val_preds = np.concatenate(val_preds).ravel()
    val_preds = np.clip(val_preds, 0.0, 1.0)
    val_rmse = np.sqrt(mean_squared_error(val_target, val_preds))
    gate_high_pct = np.concatenate(gate_probs_high).mean() * 100

    current_lr = optimizer.param_groups[0]["lr"]
    avg_train_loss = train_loss_sum / max(n_batches, 1)
    epoch_time = time.time() - epoch_start

    training_history.append(
        {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_rmse": val_rmse,
            "gate_high_pct": gate_high_pct,
            "lr": current_lr,
            "time": epoch_time,
        }
    )

    print(
        f"{epoch+1:>5} | {avg_train_loss:>10.6f} | {val_rmse:>10.6f} | {gate_high_pct:>9.2f}% | {current_lr:>10.6f} | {epoch_time:>5.1f}s"
    )

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    scheduler.step()

total_time = time.time() - start_time
print(f"\nTotal training time: {total_time:.1f}s")
print(f"Best validation RMSE: {best_val_rmse:.6f}")

# ============================================================
# 10. LOAD BEST MODEL & GENERATE FINAL PREDICTIONS
# ============================================================
if best_model_state is not None:
    model.load_state_dict(best_model_state)
model.to(device)
model.eval()


def predict_with_model(model, data_loader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch in data_loader:
            if isinstance(X_batch, (tuple, list)):
                X_batch = X_batch[0]
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            all_preds.append(pred.cpu().numpy())
    return np.concatenate(all_preds).ravel()


# Validation predictions
val_preds_final = predict_with_model(model, val_loader, device)
val_preds_final = np.clip(val_preds_final, 0.0, 1.0)
final_val_rmse = np.sqrt(mean_squared_error(val_target, val_preds_final))
print(f"\nVal RMSE (official metric): {final_val_rmse:.6f}")
print(
    f"Val pred stats: mean={val_preds_final.mean():.4f}, std={val_preds_final.std():.4f}"
)
print(f"Target stats: mean={val_target.mean():.4f}, std={val_target.std():.4f}")

# Test predictions - use LightGBM (better performing model)
lgb_test_preds = np.clip(
    lgb_model_full.predict(test_feat.values, num_iteration=lgb_model.best_iteration), 0, 1
)
print(f"LightGBM Test pred stats: mean={lgb_test_preds.mean():.4f}, std={lgb_test_preds.std():.4f}")

# Also get NN test predictions for potential ensemble
nn_test_preds = predict_with_model(model, test_loader, device)
nn_test_preds = np.clip(nn_test_preds, 0.0, 1.0)
print(f"NN Test pred stats: mean={nn_test_preds.mean():.4f}, std={nn_test_preds.std():.4f}")

# Use LightGBM as primary (it outperforms NN)
test_preds = lgb_test_preds

# ============================================================
# 11. SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "deal_probability": test_preds}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")

# Save model state
torch.save(best_model_state, "./working/best_moe_model_state.pth")
np.save("./working/feature_mean.npy", feature_mean)
np.save("./working/feature_std.npy", feature_std)

# ============================================================
# 12. FINAL OUTPUT
# ============================================================
# Use LightGBM validation score as final (it's the model used for test predictions)
score = lgb_rmse
print(f"\n{'='*70}")
print(f"MODEL: LightGBM (primary) + BimodalMoENet (secondary)")
print(f"METRIC: RMSE (official)")
print(f"VALIDATION SPLIT: Stratified 50/50 on target bins")
print(f"POST-PROCESSING: clip to [0, 1]")
print(f"LightGBM Validation RMSE: {lgb_rmse:.6f}")
print(f"MoE model Validation RMSE: {final_val_rmse:.6f}")
print(f"Final model used for submission: LightGBM")
print(f"{'='*70}")
print(f"Final Validation Score: {score}")