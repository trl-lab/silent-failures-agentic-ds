import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from collections import Counter

# Create directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Target distribution:\n{train_df['class'].value_counts()}")

# Identify target and subject columns
target_col = "class"
subject_col = None
for col in train_df.columns:
    if "subject" in col.lower() or "Subject" in col:
        subject_col = col
        break

if subject_col is None:
    for col in train_df.columns:
        if (
            col != target_col
            and train_df[col].nunique() <= 40
            and train_df[col].nunique() > 5
        ):
            subject_col = col
            break

print(f"Subject column: {subject_col}")

# Identify feature columns
feature_cols = [c for c in train_df.columns if c not in [target_col, subject_col]]
print(f"Number of features: {len(feature_cols)}")

# ============================================================
# 2. SUBJECT-AWARE SPLIT (Train/Validation)
# ============================================================
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_indices, val_indices = next(
    sgkf.split(train_df[feature_cols], train_df[target_col], train_df[subject_col])
)

train_split = train_df.iloc[train_indices].reset_index(drop=True)
val_split = train_df.iloc[val_indices].reset_index(drop=True)

print(f"\nTrain split shape: {train_split.shape}")
print(f"Validation split shape: {val_split.shape}")
print(f"Train subjects: {train_split[subject_col].nunique()}")
print(f"Val subjects: {val_split[subject_col].nunique()}")
print(
    f"Subject overlap: {len(set(train_split[subject_col]) & set(val_split[subject_col]))}"
)


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
def engineer_features(df, feature_cols):
    df = df.copy()
    new_features = {}

    body_acc_cols = [c for c in feature_cols if "BodyAcc" in c and c.startswith("t")]
    body_gyro_cols = [c for c in feature_cols if "BodyGyro" in c and c.startswith("t")]
    total_acc_cols = [c for c in feature_cols if "totalAcc" in c and c.startswith("t")]
    gyro_cols = [c for c in feature_cols if "Gyro" in c and c.startswith("t")]

    f_body_acc_cols = [c for c in feature_cols if "BodyAcc" in c and c.startswith("f")]
    f_body_gyro_cols = [
        c for c in feature_cols if "BodyGyro" in c and c.startswith("f")
    ]
    f_total_acc_cols = [
        c for c in feature_cols if "totalAcc" in c and c.startswith("f")
    ]
    f_gyro_cols = [c for c in feature_cols if "Gyro" in c and c.startswith("f")]

    # Cross-sensor features
    if body_acc_cols and total_acc_cols:
        body_acc_mean = [c for c in body_acc_cols if "mean" in c]
        total_acc_mean = [c for c in total_acc_cols if "mean" in c]
        if body_acc_mean and total_acc_mean:
            for i in range(min(len(body_acc_mean), len(total_acc_mean))):
                denom = df[total_acc_mean[i]].abs() + 1e-8
                new_features[f"body_total_ratio_{i}"] = df[body_acc_mean[i]] / denom

    # Axis-wise aggregates
    for sensor_prefix in ["tBodyAcc", "tBodyGyro", "ttotalAcc", "tGyro"]:
        sensor_cols = [c for c in feature_cols if c.startswith(sensor_prefix)]
        for stat in ["mean", "std", "mad"]:
            stat_cols = [c for c in sensor_cols if f"-{stat}()" in c]
            if len(stat_cols) >= 3:
                new_features[f"{sensor_prefix}_{stat}_axis_mean"] = df[stat_cols].mean(
                    axis=1
                )
                new_features[f"{sensor_prefix}_{stat}_axis_max"] = df[stat_cols].max(
                    axis=1
                )
                new_features[f"{sensor_prefix}_{stat}_axis_range"] = df[stat_cols].max(
                    axis=1
                ) - df[stat_cols].min(axis=1)

    # Frequency domain energy features
    for sensor_prefix in ["fBodyAcc", "fBodyGyro", "ftotalAcc", "fGyro"]:
        sensor_cols = [c for c in feature_cols if c.startswith(sensor_prefix)]
        for stat in ["mean", "std"]:
            stat_cols = [c for c in sensor_cols if f"-{stat}()" in c]
            if len(stat_cols) >= 3:
                new_features[f"{sensor_prefix}_{stat}_axis_mean"] = df[stat_cols].mean(
                    axis=1
                )
                new_features[f"{sensor_prefix}_{stat}_axis_range"] = df[stat_cols].max(
                    axis=1
                ) - df[stat_cols].min(axis=1)

    # Interaction features: time vs frequency domain
    t_std_cols = [c for c in feature_cols if c.startswith("t") and "-std()" in c]
    f_mean_cols = [c for c in feature_cols if c.startswith("f") and "-mean()" in c]
    for sensor in ["BodyAcc", "BodyGyro", "totalAcc", "Gyro"]:
        t_std = [c for c in t_std_cols if sensor in c]
        f_mean = [c for c in f_mean_cols if sensor in c]
        if t_std and f_mean:
            t_std_x = [c for c in t_std if c.endswith("-X")]
            f_mean_x = [c for c in f_mean if c.endswith("-X")]
            if t_std_x and f_mean_x:
                new_features[f"{sensor}_tf_energy_ratio_X"] = df[t_std_x[0]] / (
                    df[f_mean_x[0]].abs() + 1e-8
                )

    # Overall magnitude features
    for sensor_prefix in ["tBodyAcc", "tBodyGyro", "ttotalAcc", "tGyro"]:
        sensor_cols = [c for c in feature_cols if c.startswith(sensor_prefix)]
        mean_cols = [c for c in sensor_cols if "-mean()" in c]
        std_cols = [c for c in sensor_cols if "-std()" in c]
        if len(mean_cols) >= 3:
            new_features[f"{sensor_prefix}_mean_magnitude"] = np.sqrt(
                (df[mean_cols[0]] ** 2 + df[mean_cols[1]] ** 2 + df[mean_cols[2]] ** 2)
            )
        if len(std_cols) >= 3:
            new_features[f"{sensor_prefix}_std_magnitude"] = np.sqrt(
                (df[std_cols[0]] ** 2 + df[std_cols[1]] ** 2 + df[std_cols[2]] ** 2)
            )

    # Body vs Total acceleration difference (gravity component)
    body_acc_all = [c for c in feature_cols if "BodyAcc" in c]
    total_acc_all = [c for c in feature_cols if "totalAcc" in c]
    for stat in ["mean", "std", "mad"]:
        body_stat = [c for c in body_acc_all if f"-{stat}()" in c]
        total_stat = [c for c in total_acc_all if f"-{stat}()" in c]
        if body_stat and total_stat:
            for i in range(min(len(body_stat), len(total_stat))):
                axis = body_stat[i].split("-")[-1]
                new_features[f"gravity_{stat}_{axis}"] = (
                    df[total_stat[i]] - df[body_stat[i]]
                )

    for name, values in new_features.items():
        df[name] = values
    return df, list(new_features.keys())


print("\n--- Feature Engineering ---")
train_split, new_feat_names = engineer_features(train_split, feature_cols)
val_split, _ = engineer_features(val_split, feature_cols)
test_df, _ = engineer_features(test_df, feature_cols)

all_feature_cols = feature_cols + new_feat_names
print(f"Total features after engineering: {len(all_feature_cols)}")
print(f"New features created: {len(new_feat_names)}")

# ============================================================
# 4. SCALING (fit on train only)
# ============================================================
print("\n--- Scaling ---")
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_split[all_feature_cols].values)
val_features_scaled = scaler.transform(val_split[all_feature_cols].values)
test_features_scaled = scaler.transform(test_df[all_feature_cols].values)

# ============================================================
# 5. PCA for dimensionality reduction
# ============================================================
print("\n--- PCA ---")
pca = PCA(n_components=0.95, random_state=42)
train_pca = pca.fit_transform(train_features_scaled)
val_pca = pca.transform(val_features_scaled)
test_pca = pca.transform(test_features_scaled)

print(f"PCA components: {pca.n_components_}")
print(f"Explained variance ratio (sum): {pca.explained_variance_ratio_.sum():.4f}")

# ============================================================
# 6. LABELS AND CLASS MAPPING
# ============================================================
train_labels = train_split[target_col].values
val_labels = val_split[target_col].values

class_names = sorted(set(train_labels.tolist()))
class_to_idx = {name: idx for idx, name in enumerate(class_names)}
idx_to_class = {idx: name for name, idx in class_to_idx.items()}
n_classes = len(class_names)

train_labels_idx = np.array([class_to_idx[l] for l in train_labels])
val_labels_idx = np.array([class_to_idx[l] for l in val_labels])

print(f"Classes: {class_names}")
print(f"Train label distribution: {np.bincount(train_labels_idx)}")
print(f"Val label distribution: {np.bincount(val_labels_idx)}")


# ============================================================
# 7. MODEL DEFINITION
# ============================================================
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = residual + self.dropout(attn_out)
        residual = x
        x_norm = self.norm2(x)
        x = residual + self.ff(x_norm)
        return x


class HARTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        n_classes=6,
        embed_dim=128,
        n_heads=8,
        n_layers=4,
        ff_dim=256,
        dropout=0.15,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.n_tokens = max(1, input_dim // 8)
        self.n_tokens = min(self.n_tokens, 64)

        self.input_proj = nn.Linear(input_dim, self.n_tokens * embed_dim)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.n_tokens, embed_dim) * 0.02
        )

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, n_heads, ff_dim, dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, n_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.input_proj(x)
        x = x.reshape(batch_size, self.n_tokens, self.embed_dim)
        x = x + self.pos_embedding
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, num_classes=6):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, logits, targets):
        log_probs = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            targets = targets.float()
            smooth_targets = torch.full_like(
                log_probs, self.smoothing / (self.num_classes - 1)
            )
            smooth_targets.scatter_(
                1, targets.unsqueeze(1).long(), 1.0 - self.smoothing
            )
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        return loss.mean()


# ============================================================
# 8. MODEL, LOSS, OPTIMIZER, SCHEDULER
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model = HARTransformer(
    input_dim=pca.n_components_,
    n_classes=n_classes,
    embed_dim=128,
    n_heads=8,
    n_layers=4,
    ff_dim=256,
    dropout=0.15,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

criterion = LabelSmoothingCrossEntropy(smoothing=0.1, num_classes=n_classes).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-2,
    betas=(0.9, 0.999),
)

warmup_epochs = 5
total_epochs = 80

warmup_scheduler = LinearLR(
    optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
)
cosine_scheduler = CosineAnnealingLR(
    optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6
)
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_epochs],
)

# Forward pass test
dummy_input = torch.randn(4, pca.n_components_).to(device)
with torch.no_grad():
    dummy_output = model(dummy_input)
print(f"Forward pass test: input {dummy_input.shape} -> output {dummy_output.shape}")

# ============================================================
# 9. DATA LOADERS
# ============================================================
BATCH_SIZE = 128

train_tensor = torch.FloatTensor(train_pca)
train_label_tensor = torch.LongTensor(train_labels_idx)
val_tensor = torch.FloatTensor(val_pca)
val_label_tensor = torch.LongTensor(val_labels_idx)
test_tensor = torch.FloatTensor(test_pca)

train_dataset = TensorDataset(train_tensor, train_label_tensor)
val_dataset = TensorDataset(val_tensor, val_label_tensor)
test_dataset = TensorDataset(test_tensor)

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
# 10. TRAINING LOOP
# ============================================================
print(f"\n{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Val Acc':<12} {'LR':<10}")
print("-" * 55)

best_val_acc = 0.0
best_model_state = None
patience = 15
patience_counter = 0

for epoch in range(1, total_epochs + 1):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_X.size(0)
        preds = logits.argmax(dim=1)
        train_correct += (preds == batch_y).sum().item()
        train_total += batch_X.size(0)

    scheduler.step()

    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            logits = model(batch_X)
            preds = logits.argmax(dim=1)
            val_correct += (preds == batch_y).sum().item()
            val_total += batch_X.size(0)

    train_acc = train_correct / train_total
    val_acc = val_correct / val_total
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"{epoch:<6} {train_loss / train_total:<12.4f} {train_acc:<12.4f} {val_acc:<12.4f} {current_lr:<10.2e}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(
                f"\nEarly stopping at epoch {epoch}. Best val acc: {best_val_acc:.4f}"
            )
            break

# ============================================================
# 11. LOAD BEST MODEL AND EVALUATE
# ============================================================
model.load_state_dict(best_model_state)
model.to(device)
model.eval()

# Validation predictions
val_preds = []
with torch.no_grad():
    for batch_X, batch_y in val_loader:
        batch_X = batch_X.to(device, non_blocking=True)
        logits = model(batch_X)
        preds = logits.argmax(dim=1).cpu().numpy()
        val_preds.extend(preds.tolist())

val_preds = np.array(val_preds)
val_acc_final = accuracy_score(val_labels_idx, val_preds)
print(f"\nFinal validation accuracy (subject-disjoint): {val_acc_final:.4f}")

# ============================================================
# 12. TEST INFERENCE AND SUBMISSION
# ============================================================
test_preds = []
with torch.no_grad():
    for (batch_X,) in test_loader:
        batch_X = batch_X.to(device, non_blocking=True)
        logits = model(batch_X)
        preds = logits.argmax(dim=1).cpu().numpy()
        test_preds.extend(preds.tolist())

test_preds = np.array(test_preds)
test_pred_classes = np.array([idx_to_class[p] for p in test_preds])

submission = pd.DataFrame(
    {"id": range(len(test_pred_classes)), "class": test_pred_classes}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(f"Submission class distribution:\n{submission['class'].value_counts()}")

# ============================================================
# 13. FINAL OUTPUT
# ============================================================
score = val_acc_final
print(f"Final Validation Score: {score}")
