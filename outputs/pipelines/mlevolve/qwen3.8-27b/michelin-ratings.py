import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
import warnings

warnings.filterwarnings("ignore")

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# STEP 1: DATA LOADING & FEATURE ENGINEERING
# ============================================================

print("Loading data...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Columns: {list(train_df.columns)}")
print(f"\nTarget distribution:\n{train_df['awardFor'].value_counts()}")

target_col = "awardFor"
id_col = None
for col in ["id", "Id", "ID", "index"]:
    if col in train_df.columns:
        id_col = col
        break

y_train_full = train_df[target_col].copy()
X_train_full = train_df.drop(columns=[target_col] + ([id_col] if id_col else []))
X_test = test_df.drop(columns=[id_col] if id_col and id_col in test_df.columns else [])

# Stratified split
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
)

print(f"\nTrain split: {X_train.shape}, Val split: {X_val.shape}")

# Identify feature types
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
bool_cols = [c for c in X_train.columns if X_train[c].dtype == bool]

for c in bool_cols:
    X_train[c] = X_train[c].astype(int)
    X_val[c] = X_val[c].astype(int)
    X_test[c] = X_test[c].astype(int)

cat_cols = [c for c in cat_cols if c not in bool_cols]

print(f"Categorical features ({len(cat_cols)}): {cat_cols[:10]}...")
print(f"Numerical features ({len(num_cols)}): {num_cols[:10]}...")

# Fill missing values
for c in cat_cols:
    X_train[c] = X_train[c].fillna("MISSING").astype(str)
    X_val[c] = X_val[c].fillna("MISSING").astype(str)
    X_test[c] = X_test[c].fillna("MISSING").astype(str)

for c in num_cols:
    train_median = X_train[c].median()
    X_train[c] = X_train[c].fillna(train_median)
    X_val[c] = X_val[c].fillna(train_median)
    X_test[c] = X_test[c].fillna(train_median)

# --- Feature Engineering ---

# 1. Frequency Encoding
print("Engineering frequency features...")
for c in cat_cols:
    freq = X_train[c].value_counts(normalize=True)
    X_train[f"{c}_freq"] = X_train[c].map(freq).fillna(0)
    X_val[f"{c}_freq"] = X_val[c].map(freq).fillna(0)
    X_test[f"{c}_freq"] = X_test[c].map(freq).fillna(0)

# 2. Target Encoding with K-Fold
print("Engineering target encoding features...")
n_splits_te = 5
skf_te = StratifiedKFold(n_splits=n_splits_te, shuffle=True, random_state=42)

le_te = LabelEncoder()
y_encoded = le_te.fit_transform(y_train)
n_classes_te = len(le_te.classes_)

for c in cat_cols:
    te_col = f"{c}_target_enc"
    X_train[te_col] = 0.0

    for fold_idx, (tr_idx, va_idx) in enumerate(skf_te.split(X_train, y_train)):
        fold_tr = X_train.iloc[tr_idx]
        fold_y = y_encoded[tr_idx]
        cat_means = pd.Series(fold_y, index=fold_tr[c]).groupby(level=0).mean()
        X_train.loc[X_train.index[va_idx], te_col] = (
            X_train.loc[X_train.index[va_idx], c]
            .map(cat_means)
            .fillna(y_encoded.mean())
        )

    cat_means_full = pd.Series(y_encoded, index=X_train[c]).groupby(level=0).mean()
    X_val[te_col] = X_val[c].map(cat_means_full).fillna(y_encoded.mean())
    X_test[te_col] = X_test[c].map(cat_means_full).fillna(y_encoded.mean())

# 3. Interaction Features
print("Engineering interaction features...")
cat_freq = {c: X_train[c].nunique() for c in cat_cols}
low_card_cats = [c for c in cat_cols if cat_freq[c] <= 50]

interaction_pairs = []
for i in range(len(low_card_cats)):
    for j in range(i + 1, min(i + 5, len(low_card_cats))):
        interaction_pairs.append((low_card_cats[i], low_card_cats[j]))
interaction_pairs = interaction_pairs[:30]

for c1, c2 in interaction_pairs:
    inter_col = f"{c1}_x_{c2}"
    X_train[inter_col] = X_train[c1] + "||" + X_train[c2]
    X_val[inter_col] = X_val[c1] + "||" + X_val[c2]
    X_test[inter_col] = X_test[c1] + "||" + X_test[c2]
    inter_freq = X_train[inter_col].value_counts(normalize=True)
    X_train[f"{inter_col}_freq"] = X_train[inter_col].map(inter_freq).fillna(0)
    X_val[f"{inter_col}_freq"] = X_val[inter_col].map(inter_freq).fillna(0)
    X_test[f"{inter_col}_freq"] = X_test[inter_col].map(inter_freq).fillna(0)

# 4. Numeric feature engineering
print("Engineering numeric features...")
if len(num_cols) > 0:
    for c in num_cols:
        if X_train[c].max() > 0 and X_train[c].min() >= 0:
            skew = X_train[c].skew()
            if abs(skew) > 1:
                X_train[f"{c}_log"] = np.log1p(X_train[c])
                X_val[f"{c}_log"] = np.log1p(X_val[c])
                X_test[f"{c}_log"] = np.log1p(X_test[c])

    if len(num_cols) >= 2:
        for i in range(min(5, len(num_cols))):
            for j in range(i + 1, min(i + 3, len(num_cols))):
                c1, c2 = num_cols[i], num_cols[j]
                ratio_col = f"{c1}_div_{c2}"
                X_train[ratio_col] = X_train[c1] / (X_train[c2] + 1e-8)
                X_val[ratio_col] = X_val[c1] / (X_val[c2] + 1e-8)
                X_test[ratio_col] = X_test[c1] / (X_test[c2] + 1e-8)

    for c in num_cols[:10]:
        bin_col = f"{c}_bin"
        bins = pd.qcut(X_train[c], q=5, duplicates="drop")
        X_train[bin_col] = bins.cat.codes
        X_val[bin_col] = pd.cut(X_val[c], bins=bins.cat.categories).cat.codes
        X_test[bin_col] = pd.cut(X_test[c], bins=bins.cat.categories).cat.codes

# 5. Missing value indicators
print("Adding missing value indicators...")
for c in cat_cols + num_cols:
    miss_col = f"{c}_is_missing"
    if c in cat_cols:
        X_train[miss_col] = 0
        X_val[miss_col] = 0
        X_test[miss_col] = 0
    else:
        X_train[miss_col] = X_train[c].isna().astype(int)
        X_val[miss_col] = X_val[c].isna().astype(int)
        X_test[miss_col] = X_test[c].isna().astype(int)

# 6. Text-based features
print("Checking for text features...")
text_cols = [c for c in cat_cols if X_train[c].str.len().mean() > 20]
if text_cols:
    for c in text_cols[:3]:
        X_train[f"{c}_len"] = X_train[c].str.len()
        X_val[f"{c}_len"] = X_val[c].str.len()
        X_test[f"{c}_len"] = X_test[c].str.len()
        X_train[f"{c}_word_count"] = X_train[c].str.split().str.len()
        X_val[f"{c}_word_count"] = X_val[c].str.split().str.len()
        X_test[f"{c}_word_count"] = X_test[c].str.split().str.len()

# 7. Aggregate features
print("Adding aggregate features...")
for c in cat_cols[:15]:
    n_unique = X_train[c].value_counts()
    X_train[f"{c}_count"] = X_train[c].map(n_unique)
    X_val[f"{c}_count"] = X_val[c].map(n_unique)
    X_test[f"{c}_count"] = X_test[c].map(n_unique)

# Convert remaining object columns to category
for c in X_train.columns:
    if X_train[c].dtype == object:
        X_train[c] = X_train[c].astype("category")
        X_val[c] = X_val[c].astype("category")
        X_test[c] = X_test[c].astype("category")

print(
    f"\nFinal feature counts: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}"
)
final_cat = X_train.select_dtypes(include=["category"]).columns.tolist()
final_num = X_train.select_dtypes(include=["number"]).columns.tolist()
print(f"Final categorical: {len(final_cat)}, Final numerical: {len(final_num)}")

# ============================================================
# STEP 2: MODEL DESIGN & DATA PREPARATION
# ============================================================

print("\nPreparing model data...")

cat_cols_final = X_train.select_dtypes(include=["category", "object"]).columns.tolist()
num_cols_final = X_train.select_dtypes(include=["number"]).columns.tolist()

print(f"Categorical features for model: {len(cat_cols_final)}")
print(f"Numerical features for model: {len(num_cols_final)}")

# Encode labels
classes = sorted(y_train.unique())
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}
n_classes = len(classes)

y_train_enc = y_train.map(class_to_idx).values
y_val_enc = y_val.map(class_to_idx).values

print(f"Number of classes: {n_classes}")
print(f"Class distribution: {np.bincount(y_train_enc)}")

# Prepare numerical features
num_train = X_train[num_cols_final].values.astype(np.float32)
num_val = X_val[num_cols_final].values.astype(np.float32)
num_test = X_test[num_cols_final].values.astype(np.float32)

num_train = np.nan_to_num(num_train, nan=0.0, posinf=0.0, neginf=0.0)
num_val = np.nan_to_num(num_val, nan=0.0, posinf=0.0, neginf=0.0)
num_test = np.nan_to_num(num_test, nan=0.0, posinf=0.0, neginf=0.0)

num_mean = num_train.mean(axis=0)
num_std = num_train.std(axis=0) + 1e-8
num_train = (num_train - num_mean) / num_std
num_val = (num_val - num_mean) / num_std
num_test = (num_test - num_mean) / num_std

# Prepare categorical features
cat_vocab_sizes = []
cat_train = []
cat_val = []
cat_test = []

for c in cat_cols_final:
    train_vals = X_train[c].astype(str).values
    vocab = {v: i + 1 for i, v in enumerate(sorted(set(train_vals)))}
    vocab_size = len(vocab) + 1
    cat_vocab_sizes.append(vocab_size)

    enc_train = np.array([vocab.get(v, 0) for v in train_vals], dtype=np.int64)
    cat_train.append(enc_train)

    val_vals = X_val[c].astype(str).values
    enc_val = np.array([vocab.get(v, 0) for v in val_vals], dtype=np.int64)
    cat_val.append(enc_val)

    test_vals = X_test[c].astype(str).values
    enc_test = np.array([vocab.get(v, 0) for v in test_vals], dtype=np.int64)
    cat_test.append(enc_test)

print(
    f"Vocab sizes: min={min(cat_vocab_sizes)}, max={max(cat_vocab_sizes)}, mean={np.mean(cat_vocab_sizes):.1f}"
)


# Define the model
class TabularEmbeddingNet(nn.Module):
    def __init__(
        self,
        cat_vocab_sizes,
        num_features,
        n_classes,
        embed_dim=32,
        hidden_dims=[256, 128, 64],
        dropout=0.3,
    ):
        super().__init__()
        self.n_cat = len(cat_vocab_sizes)
        self.num_features = num_features
        self.embed_dim = embed_dim

        self.embeddings = nn.ModuleList(
            [nn.Embedding(vs, embed_dim, padding_idx=0) for vs in cat_vocab_sizes]
        )

        input_dim = self.n_cat * embed_dim + num_features

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, n_classes))
        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
                if m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].fill_(0)

    def forward(self, cat_features, num_features):
        embeds = []
        for i, emb in enumerate(self.embeddings):
            e = emb(cat_features[i])
            embeds.append(e)
        cat_embed = torch.cat(embeds, dim=-1)
        x = torch.cat([cat_embed, num_features], dim=-1)
        logits = self.mlp(x)
        return logits


model = TabularEmbeddingNet(
    cat_vocab_sizes=cat_vocab_sizes,
    num_features=len(num_cols_final),
    n_classes=n_classes,
    embed_dim=32,
    hidden_dims=[256, 128, 64],
    dropout=0.3,
)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {n_params:,}")

# Loss with class weighting
class_weights = np.bincount(y_train_enc)
class_weights = 1.0 / (class_weights + 1e-6)
class_weights = class_weights / class_weights.sum() * n_classes
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# Optimizer & scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)

# Create DataLoaders
cat_train_tensors = [torch.tensor(c, dtype=torch.long) for c in cat_train]
cat_val_tensors = [torch.tensor(c, dtype=torch.long) for c in cat_val]
cat_test_tensors = [torch.tensor(c, dtype=torch.long) for c in cat_test]

num_train_t = torch.tensor(num_train, dtype=torch.float32)
num_val_t = torch.tensor(num_val, dtype=torch.float32)
num_test_t = torch.tensor(num_test, dtype=torch.float32)

y_train_t = torch.tensor(y_train_enc, dtype=torch.long)
y_val_t = torch.tensor(y_val_enc, dtype=torch.long)

train_dataset = TensorDataset(*cat_train_tensors, num_train_t, y_train_t)
val_dataset = TensorDataset(*cat_val_tensors, num_val_t, y_val_t)
test_dataset = TensorDataset(*cat_test_tensors, num_test_t)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=2)

# ============================================================
# STEP 3: TRAINING & EVALUATION
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
model.to(device)

max_epochs = 150
patience = 15
best_val_acc = 0.0
best_model_state = None
patience_counter = 0

print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

for epoch in range(max_epochs):
    model.train()
    train_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        cat_batch = [b.to(device) for b in batch[:-2]]
        num_batch = batch[-2].to(device)
        y_batch = batch[-1].to(device)

        optimizer.zero_grad()
        logits = model(cat_batch, num_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        n_batches += 1

    scheduler.step()
    avg_train_loss = train_loss / n_batches

    # Validation
    model.eval()
    val_preds = []
    with torch.no_grad():
        for batch in val_loader:
            cat_batch = [b.to(device) for b in batch[:-2]]
            num_batch = batch[-2].to(device)
            logits = model(cat_batch, num_batch)
            preds = logits.argmax(dim=-1)
            val_preds.append(preds.cpu().numpy())

    val_preds = np.concatenate(val_preds)
    val_acc = accuracy_score(y_val_enc, val_preds)

    print(
        f"Epoch {epoch+1:3d}/{max_epochs} | Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(
                f"Early stopping at epoch {epoch+1} (best val acc: {best_val_acc:.4f})"
            )
            break

# Load best model
model.load_state_dict(best_model_state)
model.to(device)
model.eval()

# Final validation
val_preds_final = []
with torch.no_grad():
    for batch in val_loader:
        cat_batch = [b.to(device) for b in batch[:-2]]
        num_batch = batch[-2].to(device)
        logits = model(cat_batch, num_batch)
        preds = logits.argmax(dim=-1)
        val_preds_final.append(preds.cpu().numpy())

val_preds_final = np.concatenate(val_preds_final)
final_val_acc = accuracy_score(y_val_enc, val_preds_final)
print(f"\nBest validation accuracy: {final_val_acc:.4f}")

# Test inference
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        cat_batch = [b.to(device) for b in batch[:-1]]
        num_batch = batch[-1].to(device)
        logits = model(cat_batch, num_batch)
        preds = logits.argmax(dim=-1)
        test_preds.append(preds.cpu().numpy())

test_preds = np.concatenate(test_preds)
test_pred_labels = np.array([idx_to_class[p] for p in test_preds])

# Save submission
submission = pd.DataFrame(
    {"id": range(len(test_pred_labels)), "awardFor": test_pred_labels}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(
    f"Prediction distribution:\n{pd.Series(test_pred_labels).value_counts().to_string()}"
)

torch.save(best_model_state, "./working/best_model_state.pth")

print(f"Final Validation Score: {final_val_acc}")
