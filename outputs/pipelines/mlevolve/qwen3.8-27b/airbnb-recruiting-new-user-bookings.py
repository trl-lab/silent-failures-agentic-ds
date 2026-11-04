import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import json
import os
from datetime import datetime

# Create output directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("Loading data...")
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================


def engineer_features(
    df, is_train=True, ref_date=None, freq_maps=None, target_enc_maps=None
):
    df = df.copy()

    # Parse dates
    df["date_account_created"] = pd.to_datetime(df["date_account_created"])
    df["timestamp_first_active"] = pd.to_datetime(
        df["timestamp_first_active"].astype(str), format="%Y%m%d%H%M%S", errors="coerce"
    )

    if is_train:
        df["date_first_booking"] = pd.to_datetime(df["date_first_booking"])

    # Temporal features
    if ref_date is not None:
        df["days_since_account_creation"] = (
            ref_date - df["date_account_created"]
        ).dt.days
        df["days_since_first_active"] = (
            ref_date - df["timestamp_first_active"]
        ).dt.days
    else:
        ref_date = df["date_account_created"].max() + pd.Timedelta(days=1)
        df["days_since_account_creation"] = (
            ref_date - df["date_account_created"]
        ).dt.days
        df["days_since_first_active"] = (
            ref_date - df["timestamp_first_active"]
        ).dt.days

    df["days_creation_to_active"] = (
        df["timestamp_first_active"] - df["date_account_created"]
    ).dt.days

    df["first_active_hour"] = df["timestamp_first_active"].dt.hour
    df["first_active_dayofweek"] = df["timestamp_first_active"].dt.dayofweek
    df["first_active_month"] = df["timestamp_first_active"].dt.month

    df["account_creation_dayofweek"] = df["date_account_created"].dt.dayofweek
    df["account_creation_month"] = df["date_account_created"].dt.month
    df["account_creation_year"] = df["date_account_created"].dt.year

    # Booking status (CRITICAL feature)
    if is_train:
        df["has_first_booking"] = (~df["date_first_booking"].isna()).astype(int)
        df["days_since_first_booking"] = (ref_date - df["date_first_booking"]).dt.days
        df["days_creation_to_booking"] = (
            df["date_first_booking"] - df["date_account_created"]
        ).dt.days
    else:
        df["has_first_booking"] = 0
        df["days_since_first_booking"] = np.nan
        df["days_creation_to_booking"] = np.nan

    # Age features
    df.loc[df["age"] > 100, "age"] = np.nan
    df["age_decade"] = (df["age"].fillna(0) // 10 * 10).astype(int)
    df["is_young"] = ((df["age"] >= 18) & (df["age"] <= 25)).astype(int)
    df["is_adult"] = ((df["age"] > 25) & (df["age"] <= 45)).astype(int)
    df["is_senior"] = (df["age"] > 45).astype(int)

    # Interaction features
    df["device_browser"] = (
        df["first_device_type"].astype(str) + "_" + df["first_browser"].astype(str)
    )
    df["signup_app_method"] = (
        df["signup_app"].astype(str) + "_" + df["signup_method"].astype(str)
    )
    df["channel_provider"] = (
        df["affiliate_channel"].astype(str) + "_" + df["affiliate_provider"].astype(str)
    )
    df["gender_device"] = (
        df["gender"].astype(str) + "_" + df["first_device_type"].astype(str)
    )
    df["language_app"] = df["language"].astype(str) + "_" + df["signup_app"].astype(str)

    # Frequency encoding (placeholder - will be filled after split)
    for col in [
        "affiliate_provider",
        "first_browser",
        "device_browser",
        "channel_provider",
    ]:
        df[f"{col}_freq"] = 0.0

    # Target encoding (placeholder - will be filled after split)
    for col in [
        "first_device_type",
        "first_browser",
        "signup_app",
        "affiliate_channel",
        "language",
    ]:
        df[f"{col}_target_enc"] = 0.0

    # Numeric transformations
    df["signup_flow_log"] = np.log1p(df["signup_flow"])

    return df, ref_date


# Identify target and features
target_col = "country_destination"
id_col = "id"

y_all = train[target_col].values
test_ids = test[id_col].values if id_col in test.columns else np.arange(len(test))

# Drop target and id from feature frames
train_features = train.drop(
    columns=[target_col] + ([id_col] if id_col in train.columns else [])
)
test_features = test.drop(columns=[id_col] if id_col in test.columns else [])

# Determine reference date from training data
ref_date = pd.to_datetime(train["date_account_created"]).max() + pd.Timedelta(days=1)

classes = sorted(train[target_col].unique())
n_classes = len(classes)

print(f"Reference date: {ref_date}")
print(f"Number of classes: {n_classes}")
print(f"Class distribution:\n{pd.Series(y_all).value_counts()}")

# Class index mappings (needed for target encoding and later tensor conversion)
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}

# Apply feature engineering WITHOUT freq/target enc first (use None to avoid leakage)
print("Engineering features...")
train_feat, _ = engineer_features(
    train_features,
    is_train=True,
    ref_date=ref_date,
    freq_maps=None,
    target_enc_maps=None,
)
test_feat, _ = engineer_features(
    test_features,
    is_train=False,
    ref_date=ref_date,
    freq_maps=None,
    target_enc_maps=None,
)

# ============================================================
# 3. PREPARE CATEGORICAL FEATURES
# ============================================================
cat_cols = [
    "affiliate_channel",
    "affiliate_provider",
    "first_affiliate_tracked",
    "first_browser",
    "first_device_type",
    "gender",
    "language",
    "signup_app",
    "signup_method",
    "device_browser",
    "signup_app_method",
    "channel_provider",
    "gender_device",
    "language_app",
]

for col in cat_cols:
    if col in train_feat.columns:
        train_feat[col] = train_feat[col].fillna("-unknown-").astype(str)
    if col in test_feat.columns:
        test_feat[col] = test_feat[col].fillna("-unknown-").astype(str)

for col in cat_cols:
    if col in train_feat.columns:
        train_feat[col] = train_feat[col].astype("category")
    if col in test_feat.columns:
        train_cats = set(train_feat[col].astype(str).unique())
        test_feat[col] = test_feat[col].astype("category")
        # Add any train categories not in test so dtypes align
        test_cats = set(test_feat[col].cat.categories)
        missing_in_test = train_cats - test_cats
        if missing_in_test:
            test_feat[col] = test_feat[col].cat.add_categories(sorted(missing_in_test))
        # Remove test-only categories (map them to <UNK> later via encoder)
        extra_in_test = test_cats - train_cats
        if extra_in_test:
            test_feat[col] = test_feat[col].cat.remove_categories(sorted(extra_in_test))

num_cols = [
    "age",
    "age_decade",
    "days_since_account_creation",
    "days_since_first_active",
    "days_creation_to_active",
    "days_since_first_booking",
    "days_creation_to_booking",
    "first_active_hour",
    "first_active_dayofweek",
    "first_active_month",
    "account_creation_dayofweek",
    "account_creation_month",
    "account_creation_year",
    "has_first_booking",
    "is_young",
    "is_adult",
    "is_senior",
    "signup_flow",
    "signup_flow_log",
]

for col in num_cols:
    if col in train_feat.columns:
        train_feat[col] = train_feat[col].fillna(0).astype(np.float64)
    if col in test_feat.columns:
        test_feat[col] = test_feat[col].fillna(0).astype(np.float64)

for col in [
    "affiliate_provider",
    "first_browser",
    "device_browser",
    "channel_provider",
]:
    if f"{col}_freq" in train_feat.columns:
        train_feat[f"{col}_freq"] = (
            train_feat[f"{col}_freq"].fillna(0).astype(np.float64)
        )
    if f"{col}_freq" in test_feat.columns:
        test_feat[f"{col}_freq"] = test_feat[f"{col}_freq"].fillna(0).astype(np.float64)

for col in [
    "first_device_type",
    "first_browser",
    "signup_app",
    "affiliate_channel",
    "language",
]:
    if f"{col}_target_enc" in train_feat.columns:
        train_feat[f"{col}_target_enc"] = (
            train_feat[f"{col}_target_enc"].fillna(0).astype(np.float64)
        )
    if f"{col}_target_enc" in test_feat.columns:
        test_feat[f"{col}_target_enc"] = (
            test_feat[f"{col}_target_enc"].fillna(0).astype(np.float64)
        )

print(f"Train features shape: {train_feat.shape}")
print(f"Test features shape: {test_feat.shape}")

# ============================================================
# 4. SPLIT INTO TRAIN / VALIDATION
# ============================================================
X_train, X_val, y_train_split, y_val_split = train_test_split(
    train_feat, y_all, test_size=0.2, random_state=42, stratify=y_all
)

print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")

# ============================================================
# 4b. COMPUTE FREQ & TARGET ENCODING FROM TRAIN SPLIT ONLY
# ============================================================
smoothing = 10

# Frequency maps from X_train only
freq_maps = {}
for col in ["affiliate_provider", "first_browser", "device_browser", "channel_provider"]:
    if col in X_train.columns:
        freq_maps[col] = X_train[col].astype(str).value_counts(normalize=True)
    else:
        freq_maps[col] = {}

# Target encoding maps from X_train only (using y_train_split)
target_enc_maps = {}
for col in ["first_device_type", "first_browser", "signup_app", "affiliate_channel", "language"]:
    if col not in X_train.columns:
        continue
    te_dict = {}
    # Compute mean one-hot target per category (smoothed)
    y_arr = np.array([class_to_idx[c] for c in y_train_split])
    n_total = len(y_arr)
    for cat in X_train[col].unique():
        mask = (X_train[col] == cat).values
        n_cat = int(mask.sum())
        if n_cat == 0:
            te_dict[cat] = 0.0
        else:
            # Mean probability of each class within this category
            cat_targets = y_arr[mask]
            class_counts = np.bincount(cat_targets, minlength=n_classes).astype(float)
            class_probs = class_counts / n_cat
            # Smooth toward prior
            prior = np.ones(n_classes) / n_classes
            weight = n_cat / (n_cat + smoothing)
            smoothed = weight * class_probs + (1 - weight) * prior
            # Use the max class probability as scalar target encoding
            te_dict[cat] = float(np.max(smoothed))
    target_enc_maps[col] = te_dict

# Apply freq and target enc to all splits using train-derived maps
for col in ["affiliate_provider", "first_browser", "device_browser", "channel_provider"]:
    if col in X_train.columns:
        X_train[f"{col}_freq"] = X_train[col].astype(str).map(freq_maps.get(col, {})).fillna(0).astype(np.float64)
        X_val[f"{col}_freq"] = X_val[col].astype(str).map(freq_maps.get(col, {})).fillna(0).astype(np.float64)
        test_feat[f"{col}_freq"] = test_feat[col].astype(str).map(freq_maps.get(col, {})).fillna(0).astype(np.float64)

for col in ["first_device_type", "first_browser", "signup_app", "affiliate_channel", "language"]:
    if col in X_train.columns:
        X_train[f"{col}_target_enc"] = X_train[col].astype(str).map(target_enc_maps.get(col, {})).fillna(0).astype(np.float64)
        X_val[f"{col}_target_enc"] = X_val[col].astype(str).map(target_enc_maps.get(col, {})).fillna(0).astype(np.float64)
        test_feat[f"{col}_target_enc"] = test_feat[col].astype(str).map(target_enc_maps.get(col, {})).fillna(0).astype(np.float64)

# ============================================================
# 5. IDENTIFY FEATURE TYPES FOR NEURAL NETWORK
# ============================================================
categorical_cols = [c for c in cat_cols if c in X_train.columns]
numerical_cols = [
    "age",
    "age_decade",
    "days_since_account_creation",
    "days_since_first_active",
    "days_creation_to_active",
    "days_since_first_booking",
    "days_creation_to_booking",
    "first_active_hour",
    "first_active_dayofweek",
    "first_active_month",
    "account_creation_dayofweek",
    "account_creation_month",
    "account_creation_year",
    "has_first_booking",
    "is_young",
    "is_adult",
    "is_senior",
    "signup_flow",
    "signup_flow_log",
    "affiliate_provider_freq",
    "first_browser_freq",
    "device_browser_freq",
    "channel_provider_freq",
    "first_device_type_target_enc",
    "first_browser_target_enc",
    "signup_app_target_enc",
    "affiliate_channel_target_enc",
    "language_target_enc",
]
numerical_cols = [c for c in numerical_cols if c in X_train.columns]

print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
print(f"Numerical features ({len(numerical_cols)}): {numerical_cols}")

# ============================================================
# 6. PREPARE CATEGORICAL VOCABULARIES
# ============================================================
cat_vocab_sizes = {}
cat_encoders = {}

for col in categorical_cols:
    train_cats = X_train[col].astype(str).unique()
    vocab = ["<PAD>", "<UNK>"] + sorted(train_cats.tolist())
    cat_vocab_sizes[col] = len(vocab)
    cat_encoders[col] = {v: i for i, v in enumerate(vocab)}

print("Vocabulary sizes:", {k: v for k, v in cat_vocab_sizes.items()})


# ============================================================
# 7. DATA PREPARATION FOR NEURAL NETWORK
# ============================================================
def encode_categoricals(df, cat_encoders, categorical_cols):
    encoded = {}
    for col in categorical_cols:
        encoder = cat_encoders[col]
        encoded[col] = (
            df[col].astype(str).map(encoder).fillna(1).astype(np.int64).values
        )
    return encoded


train_cat_enc = encode_categoricals(X_train, cat_encoders, categorical_cols)
val_cat_enc = encode_categoricals(X_val, cat_encoders, categorical_cols)
test_cat_enc = encode_categoricals(test_feat, cat_encoders, categorical_cols)


def prepare_numericals(df, numerical_cols):
    num_data = df[numerical_cols].values.astype(np.float32)
    num_data = np.nan_to_num(num_data, nan=0.0)
    return num_data


train_num = prepare_numericals(X_train, numerical_cols)
val_num = prepare_numericals(X_val, numerical_cols)
test_num = prepare_numericals(test_feat, numerical_cols)
if test_num.shape[1] < len(numerical_cols):
    pad_width = len(numerical_cols) - test_num.shape[1]
    test_num = np.pad(test_num, ((0, 0), (0, pad_width)), constant_values=0)

# Normalize numerical features (fit on train only)
num_mean = train_num.mean(axis=0)
num_std = train_num.std(axis=0) + 1e-8

train_num = (train_num - num_mean) / num_std
val_num = (val_num - num_mean) / num_std
test_num = (test_num - num_mean) / num_std

# Target encoding
y_train_tensor = torch.tensor(
    [class_to_idx[c] for c in y_train_split], dtype=torch.long
)
y_val_tensor = torch.tensor([class_to_idx[c] for c in y_val_split], dtype=torch.long)

print(f"Y train shape: {y_train_tensor.shape}, Y val shape: {y_val_tensor.shape}")


# ============================================================
# 8. DATASET AND DATALOADER
# ============================================================
class TabularDataset(Dataset):
    def __init__(self, cat_data, num_data, labels=None):
        self.cat_data = cat_data
        self.num_data = num_data
        self.labels = labels

    def __len__(self):
        return len(self.num_data)

    def __getitem__(self, idx):
        item = {"num": torch.tensor(self.num_data[idx], dtype=torch.float32)}
        for col in categorical_cols:
            item[col] = torch.tensor(self.cat_data[col][idx], dtype=torch.long)
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


train_dataset = TabularDataset(train_cat_enc, train_num, y_train_tensor)
val_dataset = TabularDataset(val_cat_enc, val_num, y_val_tensor)
test_dataset = TabularDataset(test_cat_enc, test_num, None)

train_loader = DataLoader(
    train_dataset, batch_size=256, shuffle=True, num_workers=2, drop_last=False
)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=2)


# ============================================================
# 9. MODEL ARCHITECTURE
# ============================================================
class FieldEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.layer_norm(self.embedding(x))


class FieldInteractionBlock(nn.Module):
    def __init__(self, n_fields, embed_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_fields = n_fields
        self.embed_dim = embed_dim
        self.n_heads = n_heads

        self.field_pos = nn.Parameter(torch.randn(1, n_fields, embed_dim) * 0.02)

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, field_embeddings):
        x = field_embeddings + self.field_pos
        attn_out, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + ffn_out)
        return x, None


class TabularNDCGNet(nn.Module):
    def __init__(
        self,
        n_num_features,
        cat_vocab_sizes,
        n_classes,
        embed_dim=32,
        n_heads=4,
        n_attention_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        self.n_num_features = n_num_features
        self.embed_dim = embed_dim
        self.n_fields = len(cat_vocab_sizes) + 1

        self.cat_embeddings = nn.ModuleDict(
            {
                col: FieldEmbedding(vocab_size, embed_dim)
                for col, vocab_size in cat_vocab_sizes.items()
            }
        )

        self.num_projection = nn.Sequential(
            nn.Linear(n_num_features, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

        self.attention_layers = nn.ModuleList(
            [
                FieldInteractionBlock(self.n_fields, embed_dim, n_heads, dropout)
                for _ in range(n_attention_layers)
            ]
        )

        head_input_dim = self.n_fields * embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(head_input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, batch):
        num_emb = self.num_projection(batch["num"])
        cat_embs = []
        for col in categorical_cols:
            cat_embs.append(self.cat_embeddings[col](batch[col]))
        all_embs = torch.stack([num_emb] + cat_embs, dim=1)
        for attn_layer in self.attention_layers:
            all_embs, _ = attn_layer(all_embs)
        flat = all_embs.reshape(all_embs.size(0), -1)
        logits = self.classifier(flat)
        return logits


# ============================================================
# 10. LOSS FUNCTION
# ============================================================
class NDCGAwareLoss(nn.Module):
    def __init__(
        self,
        n_classes,
        ndcg_weight=0.3,
        ce_weight=1.0,
        calib_weight=0.1,
        temperature=1.0,
    ):
        super().__init__()
        self.ndcg_weight = ndcg_weight
        self.ce_weight = ce_weight
        self.calib_weight = calib_weight
        self.temperature = temperature
        self.n_classes = n_classes

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets)

        probs = F.softmax(logits / self.temperature, dim=1)
        true_scores = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        mask = torch.ones_like(probs)
        mask.scatter_(1, targets.unsqueeze(1), 0)

        diff = (probs - true_scores.unsqueeze(1)) * mask
        diff = torch.clamp(diff, min=0)
        ranking_loss = F.softplus(diff * 10).mean()

        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
        calib_loss = (entropy**2).mean() * 0.01

        total_loss = (
            self.ce_weight * ce_loss
            + self.ndcg_weight * ranking_loss
            + self.calib_weight * calib_loss
        )
        return total_loss, {
            "ce": ce_loss.item(),
            "ranking": ranking_loss.item(),
            "calib": calib_loss.item(),
        }


# ============================================================
# 11. INSTANTIATE MODEL, LOSS, OPTIMIZER
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = TabularNDCGNet(
    n_num_features=len(numerical_cols),
    cat_vocab_sizes=cat_vocab_sizes,
    n_classes=n_classes,
    embed_dim=32,
    n_heads=4,
    n_attention_layers=2,
    dropout=0.2,
).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {n_params:,}")

criterion = NDCGAwareLoss(
    n_classes=n_classes,
    ndcg_weight=0.3,
    ce_weight=1.0,
    calib_weight=0.1,
    temperature=1.0,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-5
)


# ============================================================
# 12. NDCG@3 EVALUATION (Official Metric)
# ============================================================
def compute_ndcg(y_true_idx, y_prob, k=3):
    """Compute mean NDCG@k for multi-class classification.

    For each sample, relevance is binary: 1 if the predicted class at rank r
    equals the true class, 0 otherwise. DCG sums rel(r)/log2(r+1) for r=1..k.
    IDCG = 1/log2(1) = 1 (best case: true class at rank 1).
    """
    n_samples = len(y_true_idx)
    top_k_indices = np.argsort(-y_prob, axis=1)[:, :k]
    ndcg_scores = np.zeros(n_samples)
    for i in range(n_samples):
        dcg = 0.0
        for rank in range(k):
            if top_k_indices[i, rank] == y_true_idx[i]:
                dcg += 1.0 / np.log2(rank + 2)
        # IDCG: best possible is finding the true class at rank 1
        idcg = 1.0 / np.log2(1 + 1)  # = 1.0
        ndcg_scores[i] = dcg / idcg
    return np.mean(ndcg_scores)


# ============================================================
# 13. TRAINING LOOP
# ============================================================
def train_epoch(epoch):
    model.train()
    total_loss = 0
    n_batches = 0
    for batch in train_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch)
        loss, loss_dict = criterion(logits, batch["label"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    avg_loss = total_loss / n_batches
    print(
        f"Epoch {epoch}: loss={avg_loss:.4f} (ce={loss_dict['ce']:.4f}, rank={loss_dict['ranking']:.4f})"
    )
    return avg_loss


@torch.no_grad()
def evaluate():
    model.eval()
    all_probs = []
    all_labels = []
    for batch in val_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = model(batch)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(batch["label"].cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    ndcg = compute_ndcg(all_labels, all_probs, k=3)
    return ndcg, all_probs, all_labels


# ============================================================
# 14. TRAIN WITH EARLY STOPPING
# ============================================================
best_ndcg = 0
patience = 15
patience_counter = 0
max_epochs = 60

print("\nStarting training...")
for epoch in range(1, max_epochs + 1):
    train_epoch(epoch)
    scheduler.step()

    val_ndcg, val_probs, val_labels = evaluate()
    print(f"  Val NDCG@3: {val_ndcg:.6f}")

    if val_ndcg > best_ndcg:
        best_ndcg = val_ndcg
        patience_counter = 0
        torch.save(model.state_dict(), "./working/best_model_state.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# ============================================================
# 15. LOAD BEST MODEL AND GENERATE PREDICTIONS
# ============================================================
model.load_state_dict(torch.load("./working/best_model_state.pt"))
model.eval()

test_probs_list = []
with torch.no_grad():
    for batch in test_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = model(batch)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        test_probs_list.append(probs)

test_probs = np.concatenate(test_probs_list, axis=0)

# ============================================================
# 16. GENERATE SUBMISSION
# ============================================================
# Official format: 3 rows per id, each row has one of the top-3 predicted countries
# Space-separated top-3 countries in a single column per id group
rows = []
for i in range(len(test_probs)):
    top3_idx = np.argsort(-test_probs[i])[:3]
    top3_countries = [classes[j] for j in top3_idx]
    for c in top3_countries:
        rows.append({"id": i, "country_destination": c})

submission = pd.DataFrame(rows, columns=["id", "country_destination"])
submission["id"] = submission["id"].astype(int)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved. Shape: {submission.shape}")
print(f"Unique ids: {submission['id'].nunique()}")

# ============================================================
# 17. FINAL VALIDATION SCORE
# ============================================================
score = best_ndcg
print(f"\nBest Validation NDCG@3: {score:.6f}")
print(f"Final Validation Score: {score}")