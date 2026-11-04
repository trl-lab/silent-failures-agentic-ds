import pandas as pd
import numpy as np
import os
import re
import pickle
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Create directories
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
print(f"\nTrain columns: {train.columns.tolist()}")
print(f"\nTarget distribution:\n{train['stars'].value_counts().sort_index()}")

# ============================================================
# 2. IDENTIFY COLUMN TYPES
# ============================================================
target_col = "stars"
id_col = "id" if "id" in train.columns else None

text_cols = []
cat_cols = []
num_cols = []
drop_cols = []

for col in train.columns:
    if col == target_col:
        continue
    if col == id_col:
        continue
    dtype = train[col].dtype
    if (
        dtype == "object"
        or str(dtype) == "string"
        or train[col].apply(lambda x: isinstance(x, str)).sum() > len(train) * 0.5
    ):
        non_null = train[col].dropna()
        if len(non_null) > 0:
            avg_len = non_null.astype(str).str.len().mean()
            nunique = non_null.nunique()
            if avg_len > 50 or nunique > 1000:
                text_cols.append(col)
            else:
                cat_cols.append(col)
        else:
            drop_cols.append(col)
    elif dtype in ["category"]:
        cat_cols.append(col)
    elif dtype in [
        "int64",
        "int32",
        "float64",
        "float32",
        "int16",
        "float16",
        "int8",
        "bool",
    ]:
        num_cols.append(col)
    else:
        try:
            train[col] = pd.to_numeric(train[col], errors="coerce")
            if train[col].notna().sum() > len(train) * 0.1:
                num_cols.append(col)
            else:
                drop_cols.append(col)
        except:
            drop_cols.append(col)

print(f"\nText columns: {text_cols}")
print(f"Categorical columns: {cat_cols}")
print(f"Numerical columns: {num_cols}")
print(f"Drop columns: {drop_cols}")

# Remove business_id from text_cols - it's an ID with no predictive value
if 'business_id' in text_cols:
    text_cols.remove('business_id')
    drop_cols.append('business_id')
    print(f"Removed business_id from text_cols, added to drop_cols")
    print(f"Updated text columns: {text_cols}")

# ============================================================
# 3. CREATE COMBINED DATASET FOR FEATURE ENGINEERING
# ============================================================
train["is_train"] = 1
test["is_train"] = 0
if id_col is None:
    train["id"] = range(len(train))
    test["id"] = range(len(train), len(train) + len(test))
    id_col = "id"

for col in train.columns:
    if col not in test.columns:
        test[col] = np.nan if col in num_cols else (np.nan if col in cat_cols else "")

combined = pd.concat([train, test], axis=0, ignore_index=True)
train_mask = combined["is_train"] == 1

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
print("\nEngineering features...")

# --- Parse categories field: extract top 50 most frequent categories as binary features ---
print("Engineering category features...")
categories_col = 'categories'
if categories_col in combined.columns:
    # Categories are typically pipe-separated
    all_categories = []
    for val in combined.loc[train_mask, categories_col].dropna():
        cats = str(val).split('|')
        all_categories.extend([c.strip() for c in cats if c.strip()])

    from collections import Counter
    cat_counter = Counter(all_categories)
    top_50_categories = [c for c, _ in cat_counter.most_common(50)]

    for cat in top_50_categories:
        cat_clean = (
            cat.replace(" ", "_").replace("/", "_").replace("(", "")
            .replace(")", "").replace(",", "").replace(".", "").replace("&", "and")[:40]
        )
        combined[f"cat_{cat_clean}"] = 0

    def extract_categories(val):
        result = {}
        if pd.isna(val) or val == '':
            return result
        cats = str(val).split('|')
        for cat in cats:
            cat = cat.strip()
            cat_clean = (
                cat.replace(" ", "_").replace("/", "_").replace("(", "")
                .replace(")", "").replace(",", "").replace(".", "").replace("&", "and")[:40]
            )
            if f"cat_{cat_clean}" in combined.columns:
                result[f"cat_{cat_clean}"] = 1
        return result

    cat_features = combined[categories_col].apply(extract_categories).apply(pd.Series).fillna(0)
    for col in cat_features.columns:
        if col in combined.columns:
            combined[col] = cat_features[col].values

    combined[f"categories_n"] = combined[categories_col].astype(str).apply(
        lambda x: len(x.split('|')) if x and x != 'nan' else 0
    )
    print(f"  Extracted {len(top_50_categories)} category features")

# --- Parse attributes field: extract top 30 most frequent attribute keys as binary features ---
print("Engineering attribute features...")
attributes_col = 'attributes'
if attributes_col in combined.columns:
    # Attributes are typically JSON-like: {"key": "value", ...}
    all_attr_keys = []
    for val in combined.loc[train_mask, attributes_col].dropna():
        try:
            attr_dict = eval(str(val)) if str(val).startswith('{') else {}
            all_attr_keys.extend(attr_dict.keys())
        except:
            pass

    attr_counter = Counter(all_attr_keys)
    top_30_attrs = [a for a, _ in attr_counter.most_common(30)]

    for attr in top_30_attrs:
        attr_clean = (
            attr.replace(" ", "_").replace("/", "_").replace("(", "")
            .replace(")", "").replace(",", "").replace(".", "").replace("&", "and")[:40]
        )
        combined[f"attr_{attr_clean}"] = 0

    def extract_attributes(val):
        result = {}
        if pd.isna(val) or val == '' or val == 'nan':
            return result
        try:
            attr_dict = eval(str(val)) if str(val).startswith('{') else {}
        except:
            return result
        for key, value in attr_dict.items():
            attr_clean = (
                key.replace(" ", "_").replace("/", "_").replace("(", "")
                .replace(")", "").replace(",", "").replace(".", "").replace("&", "and")[:40]
            )
            if f"attr_{attr_clean}" in combined.columns:
                # Binary: 1 if attribute is present and truthy
                result[f"attr_{attr_clean}"] = 1 if (value and value != 'False' and value != 'false' and value != 'No') else 0
        return result

    attr_features = combined[attributes_col].apply(extract_attributes).apply(pd.Series).fillna(0)
    for col in attr_features.columns:
        if col in combined.columns:
            combined[col] = attr_features[col].values

    combined[f"attributes_n"] = combined[attributes_col].astype(str).apply(
        lambda x: len(eval(x).keys()) if x and x.startswith('{') else 0
    )
    print(f"  Extracted {len(top_30_attrs)} attribute features")

# --- Parse hours field: extract day-of-week and hour features ---
print("Engineering hours features...")
hours_col = 'hours'
if hours_col in combined.columns:
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    for day in day_names:
        combined[f"hours_{day}"] = 0

    def extract_hours(val):
        result = {}
        if pd.isna(val) or val == '' or val == 'nan':
            return result
        try:
            hours_dict = eval(str(val)) if str(val).startswith('{') else {}
        except:
            return result
        for day, hours in hours_dict.items():
            if day in day_names and hours and hours != '':
                result[f"hours_{day}"] = 1
        return result

    hours_features = combined[hours_col].apply(extract_hours).apply(pd.Series).fillna(0)
    for col in hours_features.columns:
        if col in combined.columns:
            combined[col] = hours_features[col].values

    # Additional hours features
    combined[f"hours_n_days"] = combined[hours_col].astype(str).apply(
        lambda x: len(eval(x).keys()) if x and x.startswith('{') else 0
    )
    combined[f"hours_open_weekend"] = (
        combined.get('hours_Saturday', 0) + combined.get('hours_Sunday', 0)
    )
    print(f"  Extracted {len(day_names)} day features + open_weekend")

# --- Text Feature Engineering ---
positive_words = r"\b(great|excellent|amazing|awesome|fantastic|outstanding|wonderful|perfect|incredible|best|love|loved|beautiful|delicious|friendly|helpful|recommend|impressed|surprised|enjoy|enjoyed|favorite|favourite)\b"
negative_words = r"\b(terrible|awful|horrible|disgusting|worst|disappointing|disappointed|rude|dirty|overpriced|waste|never|avoid|poor|bad|badly|unacceptable|offensive|pathetic|mediocre)\b"

for col in text_cols:
    col_str = combined[col].astype(str).fillna("")

    combined[f"{col}_len"] = col_str.str.len()
    combined[f"{col}_word_count"] = col_str.str.split().str.len()
    combined[f"{col}_char_count"] = col_str.str.len()
    combined[f"{col}_avg_word_len"] = combined[f"{col}_len"] / (
        combined[f"{col}_word_count"] + 1
    )
    combined[f"{col}_excl_count"] = col_str.str.count(r"[!]+")
    combined[f"{col}_quest_count"] = col_str.str.count(r"[?]+")
    combined[f"{col}_upper_ratio"] = col_str.apply(
        lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1)
    )
    combined[f"{col}_digit_count"] = col_str.str.count(r"\d")
    combined[f"{col}_punct_count"] = col_str.str.count(r"[^\w\s]")
    combined[f"{col}_pos_words"] = col_str.str.lower().str.count(positive_words)
    combined[f"{col}_neg_words"] = col_str.str.lower().str.count(negative_words)
    combined[f"{col}_sentiment_score"] = (
        combined[f"{col}_pos_words"] - combined[f"{col}_neg_words"]
    )
    combined[f"{col}_sentiment_ratio"] = combined[f"{col}_sentiment_score"] / (
        combined[f"{col}_pos_words"] + combined[f"{col}_neg_words"] + 1
    )
    combined[f"{col}_sentence_count"] = col_str.str.count(r"[.!?]+")
    combined[f"{col}_para_count"] = col_str.str.count(r"\n\n")

# --- Categorical Feature Engineering ---
for col in cat_cols:
    col_filled_train = combined.loc[train_mask, col].astype(str).replace("nan", "__MISSING__")
    col_filled_all = combined[col].astype(str).replace("nan", "__MISSING__")

    freq_map = col_filled_train.value_counts(normalize=True)
    combined[f"{col}_freq"] = col_filled_all.map(freq_map).fillna(0)

    count_map = col_filled_train.value_counts()
    combined[f"{col}_count"] = col_filled_all.map(count_map).fillna(0)

    top_cats = combined.loc[train_mask, col].value_counts().head(10).index
    for cat in top_cats:
        cat_clean = (
            str(cat)
            .replace(" ", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(",", "")[:30]
        )
        combined[f"{col}_is_{cat_clean}"] = (combined[col] == cat).astype(int)

    if combined[col].astype(str).str.contains(",").sum() > len(combined) * 0.1:
        combined[f"{col}_n_labels"] = (
            combined[col].astype(str).apply(lambda x: len(x.split(",")) if x else 0)
        )

    # Target encoding (fit on train only, with smoothing)
    train_subset = combined.loc[train_mask]
    if len(train_subset) > 0:
        global_mean = train_subset[target_col].mean()
        agg = train_subset.groupby(col)[target_col].agg(["mean", "count"])
        smoothing = 10
        smoothed_mean = (agg["mean"] * agg["count"] + global_mean * smoothing) / (
            agg["count"] + smoothing
        )
        combined[f"{col}_target_enc"] = (
            combined[col].map(smoothed_mean).fillna(global_mean)
        )

# --- Numerical Feature Engineering ---
for col in num_cols:
    combined[col] = pd.to_numeric(combined[col], errors="coerce")
    positive_mask = combined[col] > 0
    if positive_mask.sum() > 0 and positive_mask.sum() < len(combined):
        combined[f"{col}_log"] = np.log1p(combined[col].clip(lower=0))
    elif (combined[col] >= 0).sum() > len(combined) * 0.9:
        combined[f"{col}_log"] = np.log1p(combined[col])

# --- Interaction Features ---
if len(num_cols) >= 2:
    train_num = combined.loc[train_mask, num_cols].dropna()
    variances = train_num.var().sort_values(ascending=False)
    top_num_cols = variances.index[:5].tolist()

    for i in range(len(top_num_cols)):
        for j in range(i + 1, len(top_num_cols)):
            c1, c2 = top_num_cols[i], top_num_cols[j]
            combined[f"{c1}_x_{c2}"] = combined[c1] * combined[c2]
            combined[f"{c1}_div_{c2}"] = combined[c1] / (combined[c2].abs() + 1e-8)

# --- Cross-feature interactions (text + numeric) ---
for col in text_cols[:3]:
    for ncol in num_cols[:3]:
        combined[f"{col}_len_x_{ncol}"] = combined[f"{col}_len"] * combined[ncol]

# ============================================================
# 5. SPLIT INTO TRAIN / VALIDATION
# ============================================================
print("\nCreating train/validation split...")

train_processed = combined[combined["is_train"] == 1].copy()
test_processed = combined[combined["is_train"] == 0].copy()

# Exclude: original text cols, cat cols, target, id, is_train, drop cols, and the raw structured fields
exclude_cols = set(text_cols + cat_cols + [target_col, id_col, "is_train"] + drop_cols)
# Also exclude the raw structured fields that we've already parsed
exclude_cols.update(['categories', 'attributes', 'hours'])
feature_cols = [c for c in train_processed.columns if c not in exclude_cols]

print(f"Total feature columns: {len(feature_cols)}")

train_final, val_final = train_test_split(
    train_processed,
    test_size=0.15,
    random_state=42,
    stratify=train_processed[target_col],
)

print(f"Train final shape: {train_final.shape}")
print(f"Val final shape: {val_final.shape}")
print(f"Test shape: {test_processed.shape}")

# ============================================================
# 6. PREPARE FEATURE MATRICES
# ============================================================
train_features = train_final[feature_cols].copy()
val_features = val_final[feature_cols].copy()
test_features = test_processed[feature_cols].copy()

for df in [train_features, val_features, test_features]:
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1)

print(f"\nTrain features shape: {train_features.shape}")
print(f"Val features shape: {val_features.shape}")
print(f"Test features shape: {test_features.shape}")

# ============================================================
# 7. PREPARE TENSORS
# ============================================================
X_train = train_features.values.astype(np.float32)
X_val = val_features.values.astype(np.float32)
X_test = test_features.values.astype(np.float32)
y_train = train_final[target_col].values
y_val = val_final[target_col].values

n_classes = train_final[target_col].nunique()
classes = sorted(train_final[target_col].unique().tolist())
label_map = {c: i for i, c in enumerate(classes)}
y_train_mapped = np.array([label_map[v] for v in y_train])
y_val_mapped = np.array([label_map[v] for v in y_val])

print(f"Features: {X_train.shape[1]}")
print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
print(f"Classes: {classes}")
print(f"Class distribution (train): {np.bincount(y_train_mapped, minlength=n_classes)}")


# ============================================================
# 8. DEFINE NEURAL NETWORK MODEL
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out = self.dropout(out)
        return self.act(residual + out)


class TabularOrdinalNet(nn.Module):
    def __init__(
        self, input_dim, hidden_dim=256, n_blocks=4, n_classes=5, dropout_rate=0.2
    ):
        super().__init__()
        self.n_classes = n_classes
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(n_blocks)]
        )
        self.ordinal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, n_classes - 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        ordinal_logits = self.ordinal_head(x)
        return ordinal_logits

    def get_probs(self, x):
        """Compute class probabilities from ordinal cumulative link structure only."""
        x = self.input_norm(x)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        ordinal_logits = self.ordinal_head(x)

        # Ordinal head: derive class probabilities from cumulative structure
        ordinal_probs = torch.sigmoid(ordinal_logits)  # P(star > threshold_i) for i in 0..n_classes-2
        n = self.n_classes
        ordinal_class_probs = torch.zeros(x.size(0), n, device=x.device)
        # P(class_0) = 1 - p_0 (star <= threshold_0)
        ordinal_class_probs[:, 0] = 1.0 - ordinal_probs[:, 0]
        for i in range(1, n - 1):
            # P(class_i) = p_{i-1} * (1 - p_i)
            ordinal_class_probs[:, i] = ordinal_probs[:, i - 1] * (1.0 - ordinal_probs[:, i])
        # P(class_{n-1}) = p_{n-2} (star > threshold_{n-2})
        ordinal_class_probs[:, n - 1] = ordinal_probs[:, n - 2]

        # Normalize to ensure they sum to 1 (numerical safety)
        ordinal_class_probs = ordinal_class_probs / ordinal_class_probs.sum(dim=1, keepdim=True)
        return ordinal_class_probs


class OrdinalClassificationLoss(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.n_classes = n_classes
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, ordinal_logits, targets):
        ordinal_targets = torch.zeros(
            targets.size(0), self.n_classes - 1, device=targets.device
        )
        for i in range(self.n_classes - 1):
            ordinal_targets[:, i] = (targets > i).float()
        ordinal_loss = self.bce_loss(ordinal_logits, ordinal_targets)
        return ordinal_loss


class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# ============================================================
# 9. TRAIN NEURAL NETWORK
# ============================================================
print("\n" + "=" * 60)
print("Training Neural Network")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

input_dim = X_train.shape[1]
hidden_dim = 256
n_blocks = 4
dropout_rate = 0.25
batch_size = 1024
total_epochs = 60
lr = 1e-3

model = TabularOrdinalNet(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    n_blocks=n_blocks,
    n_classes=n_classes,
    dropout_rate=dropout_rate,
).to(device)

criterion = OrdinalClassificationLoss(
    n_classes=n_classes
)

param_groups = [
    {"params": model.input_norm.parameters(), "lr": lr * 0.1},
    {"params": model.input_proj.parameters(), "lr": lr * 0.5},
    {"params": model.blocks.parameters(), "lr": lr},
    {"params": model.ordinal_head.parameters(), "lr": lr * 2.0},
]
optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

warmup_epochs = 5


def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * (1 + np.cos(np.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

train_dataset = TabularDataset(X_train, y_train_mapped)
val_dataset = TabularDataset(X_val, y_val_mapped)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)

best_val_acc = 0.0
best_model_state = None
patience = 10
patience_counter = 0

for epoch in range(total_epochs):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        ordinal_logits = model(X_batch)
        loss = criterion(ordinal_logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    scheduler.step()

    model.eval()
    val_preds = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            probs = model.get_probs(X_batch)
            preds = probs.argmax(dim=1).cpu().numpy()
            val_preds.extend(preds)

    val_preds = np.array(val_preds)
    val_acc = accuracy_score(y_val_mapped, val_preds)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(
            f"Epoch {epoch+1}/{total_epochs} | Loss: {total_loss/n_batches:.4f} | Val Acc: {val_acc:.4f}"
        )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

model.load_state_dict(best_model_state)
print(f"Best NN Val Accuracy: {best_val_acc:.4f}")

# ============================================================
# 10. TRAIN LIGHTGBM ORDINAL REGRESSION (8 binary classifiers)
# ============================================================
print("\n" + "=" * 60)
print("Training LightGBM Ordinal Regression (8 binary classifiers)")
print("=" * 60)

# Thresholds: P(stars > 1.0), P(stars > 1.5), ..., P(stars > 4.5)
ordinal_thresholds = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
n_thresholds = len(ordinal_thresholds)

y_train_raw = train_final[target_col].values.astype(np.float32)
y_val_raw = val_final[target_col].values.astype(np.float32)

ordinal_lgb_models = []
ordinal_val_probs = np.zeros((len(y_val_raw), n_thresholds))
ordinal_test_probs = np.zeros((len(X_test), n_thresholds))

for t_idx, threshold in enumerate(ordinal_thresholds):
    # Create binary labels: 1 if stars > threshold, 0 otherwise
    y_train_binary = (y_train_raw > threshold).astype(np.float32)
    y_val_binary = (y_val_raw > threshold).astype(np.float32)

    lgb_params_ord = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42 + t_idx,
    }

    lgb_train_ord = lgb.Dataset(X_train, label=y_train_binary)
    lgb_val_ord = lgb.Dataset(X_val, label=y_val_binary, reference=lgb_train_ord)

    model_t = lgb.train(
        lgb_params_ord,
        lgb_train_ord,
        num_boost_round=2000,
        valid_sets=[lgb_val_ord],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    ordinal_lgb_models.append(model_t)

    ordinal_val_probs[:, t_idx] = model_t.predict(X_val, num_iteration=model_t.best_iteration)
    ordinal_test_probs[:, t_idx] = model_t.predict(X_test, num_iteration=model_t.best_iteration)
    print(f"  Threshold >{threshold}: best_iter={model_t.best_iteration}")

# Derive class probabilities from ordinal structure
# P(class_0) = 1 - p_0
# P(class_i) = p_{i-1} * (1 - p_i) for i=1..7
# P(class_8) = p_7
def ordinal_probs_to_class_probs(ordinal_probs, n_classes=9):
    """Convert ordinal threshold probabilities to class probabilities.
    ordinal_probs: (n_samples, 8) array where column i = P(stars > threshold_i)
    Returns: (n_samples, n_classes) class probability matrix
    """
    n = n_classes
    class_probs = np.zeros((ordinal_probs.shape[0], n))
    # P(class_0) = 1 - p_0
    class_probs[:, 0] = 1.0 - ordinal_probs[:, 0]
    # P(class_i) = p_{i-1} * (1 - p_i) for i=1..7
    for i in range(1, n - 1):
        class_probs[:, i] = ordinal_probs[:, i - 1] * (1.0 - ordinal_probs[:, i])
    # P(class_8) = p_7
    class_probs[:, n - 1] = ordinal_probs[:, n - 2]
    # Normalize for numerical safety
    class_probs = class_probs / class_probs.sum(axis=1, keepdims=True)
    return class_probs

lgb_ord_val_probs = ordinal_probs_to_class_probs(ordinal_val_probs, n_classes)
lgb_ord_test_probs = ordinal_probs_to_class_probs(ordinal_test_probs, n_classes)

lgb_ord_val_preds = lgb_ord_val_probs.argmax(axis=1)
lgb_ord_val_acc = accuracy_score(y_val_mapped, lgb_ord_val_preds)
print(f"LightGBM Ordinal Val Accuracy: {lgb_ord_val_acc:.4f}")

# ============================================================
# 10b. TRAIN LIGHTGBM MULTICLASS (optimized hyperparameters, diversity source)
# ============================================================
print("\n" + "=" * 60)
print("Training LightGBM Multiclass")
print("=" * 60)

class_counts = np.bincount(y_train_mapped, minlength=n_classes)
class_weights = len(y_train_mapped) / (n_classes * class_counts)
sample_weights = np.array([class_weights[y] for y in y_train_mapped])

lgb_params_mc = {
    "objective": "multiclass",
    "num_class": n_classes,
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "is_unbalance": True,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

lgb_train_mc = lgb.Dataset(X_train, label=y_train_mapped, weight=sample_weights)
lgb_val_mc = lgb.Dataset(X_val, label=y_val_mapped, reference=lgb_train_mc)

lgb_model_mc = lgb.train(
    lgb_params_mc,
    lgb_train_mc,
    num_boost_round=2000,
    valid_sets=[lgb_val_mc],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)

lgb_mc_val_probs = lgb_model_mc.predict(X_val, num_iteration=lgb_model_mc.best_iteration)
lgb_mc_test_probs = lgb_model_mc.predict(X_test, num_iteration=lgb_model_mc.best_iteration)

lgb_mc_val_preds = lgb_mc_val_probs.argmax(axis=1)
lgb_mc_val_acc = accuracy_score(y_val_mapped, lgb_mc_val_preds)
print(
    f"LightGBM Multiclass Val Accuracy: {lgb_mc_val_acc:.4f} (best iter: {lgb_model_mc.best_iteration})"
)

# ============================================================
# 11. ENSEMBLE PREDICTIONS (3 models: LGB Ordinal + LGB Multiclass + NN Ordinal)
# ============================================================
print("\n" + "=" * 60)
print("Generating Ensemble Predictions")
print("=" * 60)

# NN test predictions (ordinal head only)
model.eval()
test_dataset = TabularDataset(X_test)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)

nn_test_probs = []
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        probs = model.get_probs(X_batch).cpu().numpy()
        nn_test_probs.append(probs)
nn_test_probs = np.vstack(nn_test_probs)

# NN val probs (ordinal head only)
nn_val_probs = []
with torch.no_grad():
    for X_batch, _ in val_loader:
        X_batch = X_batch.to(device)
        probs = model.get_probs(X_batch).cpu().numpy()
        nn_val_probs.append(probs)
nn_val_probs = np.vstack(nn_val_probs)

# Use equal weights to avoid data leakage from grid-searching on validation set
print("Using equal ensemble weights (no leakage)...")

nn_val_acc = accuracy_score(y_val_mapped, nn_val_probs.argmax(axis=1))
print(f"NN Val Accuracy (ordinal only): {nn_val_acc:.4f}")

nn_weight, lgb_ord_weight, lgb_mc_weight = 1/3, 1/3, 1/3
print(f"Weights - NN: {nn_weight:.3f}, LGB Ordinal: {lgb_ord_weight:.3f}, LGB Multiclass: {lgb_mc_weight:.3f}")

# Combined probabilities
ensemble_test_probs = (
    nn_weight * nn_test_probs
    + lgb_ord_weight * lgb_ord_test_probs
    + lgb_mc_weight * lgb_mc_test_probs
)
ensemble_val_probs = (
    nn_weight * nn_val_probs
    + lgb_ord_weight * lgb_ord_val_probs
    + lgb_mc_weight * lgb_mc_val_probs
)

# Final predictions
ensemble_val_preds = ensemble_val_probs.argmax(axis=1)
ensemble_test_preds = ensemble_test_probs.argmax(axis=1)

# Final validation accuracy (official metric: accuracy)
final_val_acc = accuracy_score(y_val_mapped, ensemble_val_preds)
print(f"Ensemble Val Accuracy: {final_val_acc:.4f}")

# ============================================================
# 12. SAVE SUBMISSION
# ============================================================
pred_stars = np.array([classes[p] for p in ensemble_test_preds])

os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame(
    {id_col: test_processed[id_col].values, target_col: pred_stars}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved. Shape: {submission.shape}")
print(f"Prediction distribution:\n{submission[target_col].value_counts().sort_index()}")

# ============================================================
# 13. FINAL SCORE
# ============================================================
score = final_val_acc
print(f"Final Validation Score: {score}")