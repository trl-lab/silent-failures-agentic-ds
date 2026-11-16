import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import json
import os

# ============================================================
# NORMALIZED GINI (Official Metric)
# ============================================================


def normalized_gini(y_true, y_pred):
    """Compute normalized Gini coefficient (official Porto Seguro metric)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = len(y_true)
    if n == 0:
        return 0.0
    order = np.argsort(y_pred)
    y_sorted = y_true[order]
    ranks = np.arange(1, n + 1)
    numerator = 2.0 * np.sum(y_sorted * ranks) - (n + 1) * np.sum(y_sorted)
    denominator = n * np.sum(y_sorted)
    if denominator == 0:
        return 0.0
    return numerator / denominator


# ============================================================
# STEP 1: DATA PROCESSING AND FEATURE ENGINEERING
# ============================================================

print("=" * 60)
print("STEP 1: DATA PROCESSING AND FEATURE ENGINEERING")
print("=" * 60)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

test_ids = test["id"].values if "id" in test.columns else np.arange(len(test))
target = train["target"].values

calc_cols = [c for c in train.columns if c.startswith("ps_calc")]
car_cols = [c for c in train.columns if c.startswith("ps_car")]
ind_cols = [c for c in train.columns if c.startswith("ps_ind")]
reg_cols = [c for c in train.columns if c.startswith("ps_reg")]

feature_cols = calc_cols + car_cols + ind_cols + reg_cols
print(
    f"Feature groups: calc={len(calc_cols)}, car={len(car_cols)}, ind={len(ind_cols)}, reg={len(reg_cols)}"
)

# Split BEFORE feature engineering
train_idx, val_idx = train_test_split(
    np.arange(len(train)), test_size=0.15, random_state=42, stratify=target
)
train_split = train.iloc[train_idx].reset_index(drop=True)
val_split = train.iloc[val_idx].reset_index(drop=True)
target_train = target[train_idx]
target_val = target[val_idx]

print(f"Train split: {len(train_split)}, Val split: {len(val_split)}")
print(
    f"Train pos rate: {target_train.mean():.4f}, Val pos rate: {target_val.mean():.4f}"
)


# Feature engineering
def engineer_features(
    df_train,
    df_val,
    df_test,
    feature_cols,
    calc_cols,
    car_cols,
    ind_cols,
    reg_cols,
    target_train,
):
    # 3a: Sentinel (-1) indicator features
    sentinel_cols = []
    for col in feature_cols:
        if (df_train[col] == -1).any():
            sentinel_col = f"{col}_is_neg1"
            df_train[sentinel_col] = (df_train[col] == -1).astype(np.int8)
            df_val[sentinel_col] = (df_val[col] == -1).astype(np.int8)
            df_test[sentinel_col] = (df_test[col] == -1).astype(np.int8)
            sentinel_cols.append(sentinel_col)
    print(f"Created {len(sentinel_cols)} sentinel indicator features")

    # 3b: Imputed versions for numeric operations
    for col in feature_cols:
        if (df_train[col] == -1).any():
            col_imp = f"{col}_imp"
            median_val = df_train.loc[df_train[col] != -1, col].median()
            df_train[col_imp] = df_train[col].replace(-1, median_val)
            df_val[col_imp] = df_val[col].replace(-1, median_val)
            df_test[col_imp] = df_test[col].replace(-1, median_val)

    # 3c: Frequency encoding for low-cardinality categoricals
    freq_encoded_cols = []
    for col in feature_cols:
        nunique = df_train[col].nunique()
        if nunique <= 12:
            freq_map = df_train[col].value_counts(normalize=True)
            freq_col = f"{col}_freq"
            df_train[freq_col] = (
                df_train[col].map(freq_map).fillna(0).astype(np.float32)
            )
            df_val[freq_col] = df_val[col].map(freq_map).fillna(0).astype(np.float32)
            df_test[freq_col] = df_test[col].map(freq_map).fillna(0).astype(np.float32)
            freq_encoded_cols.append(freq_col)
    print(f"Created {len(freq_encoded_cols)} frequency encoding features")

    # 3d: Group-level aggregate features
    for group_name, group_cols in [
        ("calc", calc_cols),
        ("car", car_cols),
        ("ind", ind_cols),
        ("reg", reg_cols),
    ]:
        imp_cols = [f"{c}_imp" for c in group_cols if f"{c}_imp" in df_train.columns]
        if not imp_cols:
            imp_cols = [c for c in group_cols]

        mean_col = f"{group_name}_group_mean"
        df_train[mean_col] = df_train[imp_cols].mean(axis=1).astype(np.float32)
        df_val[mean_col] = df_val[imp_cols].mean(axis=1).astype(np.float32)
        df_test[mean_col] = df_test[imp_cols].mean(axis=1).astype(np.float32)

        std_col = f"{group_name}_group_std"
        df_train[std_col] = df_train[imp_cols].std(axis=1).fillna(0).astype(np.float32)
        df_val[std_col] = df_val[imp_cols].std(axis=1).fillna(0).astype(np.float32)
        df_test[std_col] = df_test[imp_cols].std(axis=1).fillna(0).astype(np.float32)

        range_col = f"{group_name}_group_range"
        df_train[range_col] = (
            df_train[imp_cols].max(axis=1) - df_train[imp_cols].min(axis=1)
        ).astype(np.float32)
        df_val[range_col] = (
            df_val[imp_cols].max(axis=1) - df_val[imp_cols].min(axis=1)
        ).astype(np.float32)
        df_test[range_col] = (
            df_test[imp_cols].max(axis=1) - df_test[imp_cols].min(axis=1)
        ).astype(np.float32)
    print("Created group aggregate features")

    # 3e: Cross-group interaction features
    interaction_cols = []
    key_car_cols = [c for c in car_cols if df_train[c].nunique() <= 10][:5]
    key_ind_cols = [c for c in ind_cols if df_train[c].nunique() <= 10][:5]

    def get_imp_col(df, col):
        """Return the _imp column if it exists, otherwise the original column."""
        imp_name = f"{col}_imp"
        if imp_name in df.columns:
            return df[imp_name]
        return df[col]

    for cc in key_car_cols:
        for ic in key_ind_cols[:3]:
            inter_col = f"{cc}_x_{ic}"
            df_train[inter_col] = (
                get_imp_col(df_train, cc) * get_imp_col(df_train, ic)
            ).astype(np.float32)
            df_val[inter_col] = (
                get_imp_col(df_val, cc) * get_imp_col(df_val, ic)
            ).astype(np.float32)
            df_test[inter_col] = (
                get_imp_col(df_test, cc) * get_imp_col(df_test, ic)
            ).astype(np.float32)
            interaction_cols.append(inter_col)

    key_reg_cols = [c for c in reg_cols if f"{c}_imp" in df_train.columns]
    key_calc_cols = [c for c in calc_cols if f"{c}_imp" in df_train.columns][:4]

    for rc in key_reg_cols:
        for kc in key_calc_cols:
            inter_col = f"{rc}_x_{kc}"
            df_train[inter_col] = (
                get_imp_col(df_train, rc) * get_imp_col(df_train, kc)
            ).astype(np.float32)
            df_val[inter_col] = (
                get_imp_col(df_val, rc) * get_imp_col(df_val, kc)
            ).astype(np.float32)
            df_test[inter_col] = (
                get_imp_col(df_test, rc) * get_imp_col(df_test, kc)
            ).astype(np.float32)
            interaction_cols.append(inter_col)
    print(f"Created {len(interaction_cols)} interaction features")

    # 3f: Ratio features within groups
    ratio_cols = []
    for group_name, group_cols in [("calc", calc_cols), ("ind", ind_cols)]:
        # Use _imp columns where available, fall back to original
        usable_cols = []
        for c in group_cols:
            imp_name = f"{c}_imp"
            usable_cols.append(imp_name if imp_name in df_train.columns else c)
        if len(usable_cols) > 2:
            group_mean = df_train[usable_cols].mean(axis=1)
            group_mean_safe = group_mean.replace(0, np.nan).fillna(1e-6)

            for col in usable_cols[:6]:
                ratio_col = f"{col}_ratio_{group_name}"
                df_train[ratio_col] = (df_train[col] / group_mean_safe).astype(
                    np.float32
                )
                val_group_mean = df_val[usable_cols].mean(axis=1)
                val_group_mean_safe = val_group_mean.replace(0, np.nan).fillna(1e-6)
                df_val[ratio_col] = (df_val[col] / val_group_mean_safe).astype(
                    np.float32
                )
                test_group_mean = df_test[usable_cols].mean(axis=1)
                test_group_mean_safe = test_group_mean.replace(0, np.nan).fillna(1e-6)
                df_test[ratio_col] = (df_test[col] / test_group_mean_safe).astype(
                    np.float32
                )
                ratio_cols.append(ratio_col)
    print(f"Created {len(ratio_cols)} ratio features")

    # 3g: Binary feature combinations
    binary_cols = [c for c in feature_cols if df_train[c].nunique() == 2]
    bin_combo_cols = []
    if len(binary_cols) > 10:
        corr_with_target = (
            df_train[binary_cols]
            .corrwith(pd.Series(target_train, index=df_train.index))
            .abs()
            .sort_values(ascending=False)
        )
        top_binary = corr_with_target.head(10).index.tolist()
    else:
        top_binary = binary_cols

    for i in range(len(top_binary)):
        for j in range(i + 1, min(i + 4, len(top_binary))):
            combo_col = f"{top_binary[i]}_x_{top_binary[j]}"
            df_train[combo_col] = (
                df_train[top_binary[i]] * df_train[top_binary[j]]
            ).astype(np.int8)
            df_val[combo_col] = (df_val[top_binary[i]] * df_val[top_binary[j]]).astype(
                np.int8
            )
            df_test[combo_col] = (
                df_test[top_binary[i]] * df_test[top_binary[j]]
            ).astype(np.int8)
            bin_combo_cols.append(combo_col)
    print(f"Created {len(bin_combo_cols)} binary combination features")

    # 3h: Polynomial features
    poly_cols = []
    continuous_cols = [
        c
        for c in feature_cols
        if df_train[c].dtype == "float64" and df_train[c].nunique() > 10
    ]
    for col in continuous_cols[:8]:
        imp_col = f"{col}_imp" if f"{col}_imp" in df_train.columns else col
        sq_col = f"{col}_sq"
        df_train[sq_col] = (df_train[imp_col] ** 2).astype(np.float32)
        df_val[sq_col] = (df_val[imp_col] ** 2).astype(np.float32)
        df_test[sq_col] = (df_test[imp_col] ** 2).astype(np.float32)
        poly_cols.append(sq_col)
    print(f"Created {len(poly_cols)} polynomial features")

    # 3i: Target encoding with smoothing (from train_split only)
    target_enc_cols = []
    global_mean = target_train.mean()
    smoothing = 20

    for col in feature_cols:
        nunique = df_train[col].nunique()
        if 2 <= nunique <= 20:
            temp_df = df_train[[col]].copy()
            temp_df["target"] = target_train
            agg = temp_df.groupby(col)["target"].agg(["mean", "count"])
            smoothed = (agg["mean"] * agg["count"] + global_mean * smoothing) / (
                agg["count"] + smoothing
            )

            te_col = f"{col}_target_enc"
            df_train[te_col] = (
                df_train[col].map(smoothed).fillna(global_mean).astype(np.float32)
            )
            df_val[te_col] = (
                df_val[col].map(smoothed).fillna(global_mean).astype(np.float32)
            )
            df_test[te_col] = (
                df_test[col].map(smoothed).fillna(global_mean).astype(np.float32)
            )
            target_enc_cols.append(te_col)
    print(f"Created {len(target_enc_cols)} target encoding features")

    return df_train, df_val, df_test


train_eng, val_eng, test_eng = engineer_features(
    train_split.copy(),
    val_split.copy(),
    test.copy(),
    feature_cols,
    calc_cols,
    car_cols,
    ind_cols,
    reg_cols,
    target_train,
)

# Finalize feature set
exclude_cols = ["id", "target"]
final_feature_cols = [c for c in train_eng.columns if c not in exclude_cols]
print(f"Final feature count: {len(final_feature_cols)}")

# Fill NaNs
for df in [train_eng, val_eng, test_eng]:
    df[final_feature_cols] = df[final_feature_cols].fillna(0)

# ============================================================
# STEP 2: MODEL DESIGN - TabNet-Inspired Neural Network
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: MODEL DESIGN")
print("=" * 60)

n_features = len(final_feature_cols)
print(f"Number of features for model: {n_features}")


class SparseMaxAttention(nn.Module):
    def __init__(self, d_model, n_features, n_steps):
        super().__init__()
        self.n_steps = n_steps
        self.attention = nn.Linear(d_model, n_features)

    def forward(self, x):
        logits = self.attention(x)
        weights = F.softmax(logits / 0.5, dim=-1)
        return weights


class FeatureBlock(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out * 2)
        self.bn = nn.BatchNorm1d(d_out * 2)
        self.d_out = d_out

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        return F.glu(x, dim=-1)


class DecisionBlock(nn.Module):
    def __init__(self, d_model, n_features):
        super().__init__()
        self.attention = SparseMaxAttention(d_model, n_features, 1)
        self.feature_block = FeatureBlock(n_features, d_model)
        self.bn = nn.BatchNorm1d(d_model)

    def forward(self, x, features):
        attn_weights = self.attention(x)
        attended = attn_weights * features
        out = self.feature_block(attended)
        out = self.bn(out)
        out = x + out
        return out, attn_weights


class TabNetModel(nn.Module):
    def __init__(self, n_features, d_model=64, n_steps=4, n_shared=2, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.n_steps = n_steps
        self.n_shared = n_shared

        self.initial_transform = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.prior_scales = nn.Parameter(torch.ones(n_steps, d_model) * 0.5)
        self.prior_transform = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(n_steps)]
        )

        self.decision_blocks = nn.ModuleList(
            [DecisionBlock(d_model, n_features) for _ in range(n_steps)]
        )

        self.shared_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.BatchNorm1d(d_model),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for _ in range(n_shared)
            ]
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model * n_steps, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        h = self.initial_transform(x)
        decision_outputs = []
        prior = torch.zeros(x.size(0), self.d_model, device=x.device)

        for i, block in enumerate(self.decision_blocks):
            scaled_prior = prior * self.prior_scales[i]
            h, attn_weights = block(scaled_prior, x)
            decision_outputs.append(h)
            prior = torch.tanh(self.prior_transform[i](h))

        shared_outputs = []
        for i, dec_out in enumerate(decision_outputs):
            out = dec_out
            for shared_block in self.shared_blocks:
                out = shared_block(out)
            shared_outputs.append(out)

        aggregated = torch.cat(shared_outputs, dim=-1)
        logit = self.classifier(aggregated)
        return logit.squeeze(-1)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = torch.where(
            targets == 1,
            torch.tensor(self.alpha, device=logits.device),
            torch.tensor(1 - self.alpha, device=logits.device),
        )
        loss = alpha_t * focal_weight * bce
        return loss.mean()


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


# Instantiate model
D_MODEL = 64
N_STEPS = 4
N_SHARED = 2
DROPOUT = 0.15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 4096
EPOCHS = 40
PATIENCE = 7

model = TabNetModel(
    n_features=n_features,
    d_model=D_MODEL,
    n_steps=N_STEPS,
    n_shared=N_SHARED,
    dropout=DROPOUT,
)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {n_params:,}")
print(
    f"Architecture: d_model={D_MODEL}, n_steps={N_STEPS}, n_shared={N_SHARED}, dropout={DROPOUT}"
)

criterion = FocalLoss(alpha=0.75, gamma=2.0)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=2, eta_min=1e-6
)

# ============================================================
# STEP 3: TRAINING AND EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: TRAINING AND EVALUATION")
print("=" * 60)

# Extract feature matrices
X_train = train_eng[final_feature_cols].values.astype(np.float32)
X_val = val_eng[final_feature_cols].values.astype(np.float32)
X_test = test_eng[final_feature_cols].values.astype(np.float32)

X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Positive rate - Train: {target_train.mean():.4f}, Val: {target_val.mean():.4f}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

train_dataset = TabularDataset(X_train, target_train)
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)

val_X_tensor = torch.FloatTensor(X_val).to(device)
val_y = target_val

best_gini = -np.inf
best_model_state = None
patience_counter = 0

print(f"\n{'='*60}")
print(f"Starting training: {EPOCHS} epochs, patience={PATIENCE}")
print(f"{'='*60}")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    scheduler.step()
    avg_loss = total_loss / n_batches

    model.eval()
    with torch.no_grad():
        val_logits = model(val_X_tensor)
        val_probs = torch.sigmoid(val_logits).cpu().numpy()

    val_gini = normalized_gini(val_y, val_probs)
    print(
        f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Gini: {val_gini:.6f}"
    )

    if val_gini > best_gini:
        best_gini = val_gini
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

# Restore best model
model.load_state_dict(best_model_state)
model.to(device)
model.eval()

print(f"\nBest validation Gini: {best_gini:.6f}")

# Final validation
with torch.no_grad():
    val_logits = model(val_X_tensor)
    val_probs = torch.sigmoid(val_logits).cpu().numpy()

final_val_gini = normalized_gini(val_y, val_probs)
print(f"Confirmed validation Gini: {final_val_gini:.6f}")

# Test inference
test_probs_list = []
BATCH_SIZE_TEST = 8192

with torch.no_grad():
    for i in range(0, len(X_test), BATCH_SIZE_TEST):
        batch = torch.FloatTensor(X_test[i : i + BATCH_SIZE_TEST]).to(device)
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        test_probs_list.append(probs)

test_probs = np.concatenate(test_probs_list)

# Save submission
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame({"id": test_ids.astype(int), "target": test_probs})
submission.to_csv("./submission/submission.csv", index=False)

print(f"\nSubmission saved: {submission.shape}")
print(f"Prediction stats: mean={test_probs.mean():.4f}, std={test_probs.std():.4f}")
print(f"Prediction range: [{test_probs.min():.6f}, {test_probs.max():.6f}]")

# Save model state
os.makedirs("./working", exist_ok=True)
torch.save(best_model_state, "./working/best_model_state.pt")

# Final score
score = final_val_gini
print(f"Final Validation Score: {score}")
