import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# STEP 1: DATA LOADING & FEATURE ENGINEERING
# ============================================================

print("Loading data...")
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

target_col = "requester_received_pizza"
id_col = "id" if "id" in train.columns else None

y = train[target_col].copy()
if id_col:
    train_ids = train[id_col].copy()
    test_ids = test[id_col].copy() if id_col in test.columns else None
else:
    train_ids = pd.Series(range(len(train)))
    test_ids = pd.Series(range(len(test)))

feature_cols = [c for c in train.columns if c not in [target_col, id_col]]
common_cols = [c for c in feature_cols if c in test.columns]

train_features = train[common_cols].copy()
test_features = test[common_cols].copy()

# Identify column types
datetime_cols = []
text_cols = []
cat_cols = []
num_cols = []

for col in common_cols:
    if (
        train_features[col].dtype == "object"
        or str(train_features[col].dtype) == "category"
    ):
        sample = train_features[col].dropna().astype(str)
        if len(sample) > 0:
            avg_len = sample.str.len().mean()
            if avg_len > 50:
                text_cols.append(col)
            else:
                cat_cols.append(col)
    elif np.issubdtype(train_features[col].dtype, np.number):
        num_cols.append(col)
    elif pd.api.types.is_datetime64_any_dtype(train_features[col]):
        datetime_cols.append(col)
    else:
        try:
            parsed = pd.to_datetime(train_features[col].dropna().head(100))
            if parsed.notna().sum() > 0:
                datetime_cols.append(col)
                train_features[col] = pd.to_datetime(train_features[col])
                test_features[col] = pd.to_datetime(test_features[col])
            else:
                cat_cols.append(col)
        except:
            cat_cols.append(col)

print(f"Datetime cols: {datetime_cols}")
print(f"Text cols: {text_cols}")
print(f"Categorical cols: {cat_cols}")
print(f"Numeric cols: {num_cols}")

# 1. Temporal features
for col in datetime_cols:
    if col in train_features.columns:
        train_features[f"{col}_hour"] = train_features[col].dt.hour
        train_features[f"{col}_day"] = train_features[col].dt.day
        train_features[f"{col}_weekday"] = train_features[col].dt.weekday
        train_features[f"{col}_month"] = train_features[col].dt.month
        train_features[f"{col}_year"] = train_features[col].dt.year
        train_features[f"{col}_is_weekend"] = (
            train_features[col].dt.weekday >= 5
        ).astype(int)
        train_features[f"{col}_hour_sin"] = np.sin(
            2 * np.pi * train_features[col].dt.hour / 24
        )
        train_features[f"{col}_hour_cos"] = np.cos(
            2 * np.pi * train_features[col].dt.hour / 24
        )
        train_features[f"{col}_weekday_sin"] = np.sin(
            2 * np.pi * train_features[col].dt.weekday / 7
        )
        train_features[f"{col}_weekday_cos"] = np.cos(
            2 * np.pi * train_features[col].dt.weekday / 7
        )

        if col in test_features.columns:
            test_features[f"{col}_hour"] = test_features[col].dt.hour
            test_features[f"{col}_day"] = test_features[col].dt.day
            test_features[f"{col}_weekday"] = test_features[col].dt.weekday
            test_features[f"{col}_month"] = test_features[col].dt.month
            test_features[f"{col}_year"] = test_features[col].dt.year
            test_features[f"{col}_is_weekend"] = (
                test_features[col].dt.weekday >= 5
            ).astype(int)
            test_features[f"{col}_hour_sin"] = np.sin(
                2 * np.pi * test_features[col].dt.hour / 24
            )
            test_features[f"{col}_hour_cos"] = np.cos(
                2 * np.pi * test_features[col].dt.hour / 24
            )
            test_features[f"{col}_weekday_sin"] = np.sin(
                2 * np.pi * test_features[col].dt.weekday / 7
            )
            test_features[f"{col}_weekday_cos"] = np.cos(
                2 * np.pi * test_features[col].dt.weekday / 7
            )

        train_features[col] = train_features[col].astype(np.int64) // 10**9
        test_features[col] = test_features[col].astype(np.int64) // 10**9
        if col not in num_cols:
            num_cols.append(col)

# 2. Text features
for col in text_cols:
    train_features[f"{col}_len"] = train_features[col].astype(str).str.len()
    train_features[f"{col}_word_count"] = (
        train_features[col].astype(str).str.split().str.len()
    )
    train_features[f"{col}_char_count"] = train_features[col].astype(str).str.len()
    train_features[f"{col}_space_count"] = (
        train_features[col].astype(str).str.count(" ")
    )
    train_features[f"{col}_punct_count"] = (
        train_features[col].astype(str).str.count(r"[^\w\s]")
    )
    train_features[f"{col}_digit_count"] = (
        train_features[col].astype(str).str.count(r"\d")
    )
    train_features[f"{col}_upper_ratio"] = (
        train_features[col]
        .astype(str)
        .apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
    )
    train_features[f"{col}_has_exclamation"] = (
        train_features[col].astype(str).str.contains(r"!").astype(int)
    )
    train_features[f"{col}_has_question"] = (
        train_features[col].astype(str).str.contains(r"\?").astype(int)
    )
    train_features[f"{col}_avg_word_len"] = (
        train_features[col]
        .astype(str)
        .str.split()
        .apply(
            lambda words: (
                np.mean([len(w) for w in words]) if words and len(words) > 0 else 0
            )
        )
    )
    train_features[f"{col}_unique_word_ratio"] = (
        train_features[col]
        .astype(str)
        .str.split()
        .apply(lambda words: len(set(words)) / max(len(words), 1) if words else 0)
    )

    if col in test_features.columns:
        test_features[f"{col}_len"] = test_features[col].astype(str).str.len()
        test_features[f"{col}_word_count"] = (
            test_features[col].astype(str).str.split().str.len()
        )
        test_features[f"{col}_char_count"] = test_features[col].astype(str).str.len()
        test_features[f"{col}_space_count"] = (
            test_features[col].astype(str).str.count(" ")
        )
        test_features[f"{col}_punct_count"] = (
            test_features[col].astype(str).str.count(r"[^\w\s]")
        )
        test_features[f"{col}_digit_count"] = (
            test_features[col].astype(str).str.count(r"\d")
        )
        test_features[f"{col}_upper_ratio"] = (
            test_features[col]
            .astype(str)
            .apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
        )
        test_features[f"{col}_has_exclamation"] = (
            test_features[col].astype(str).str.contains(r"!").astype(int)
        )
        test_features[f"{col}_has_question"] = (
            test_features[col].astype(str).str.contains(r"\?").astype(int)
        )
        test_features[f"{col}_avg_word_len"] = (
            test_features[col]
            .astype(str)
            .str.split()
            .apply(
                lambda words: (
                    np.mean([len(w) for w in words]) if words and len(words) > 0 else 0
                )
            )
        )
        test_features[f"{col}_unique_word_ratio"] = (
            test_features[col]
            .astype(str)
            .str.split()
            .apply(lambda words: len(set(words)) / max(len(words), 1) if words else 0)
        )

    train_features.drop(columns=[col], inplace=True)
    test_features.drop(columns=[col], inplace=True)

# 8. Stratified train/validation split (BEFORE data-dependent preprocessing)
train_final, val_final, y_train, y_val = train_test_split(
    train_features, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Categorical features - Frequency encoding (fit on train portion only)
for col in cat_cols:
    if col not in train_final.columns:
        continue

    # Frequency encoding (on original dtype, before category conversion)
    freq_map = train_final[col].value_counts(normalize=True)
    train_final[f"{col}_freq"] = train_final[col].map(freq_map).fillna(0)
    if col in val_final.columns:
        val_final[f"{col}_freq"] = val_final[col].map(freq_map).fillna(0)
    if col in test_features.columns:
        test_features[f"{col}_freq"] = test_features[col].map(freq_map).fillna(0)

    # Convert to category for code encoding
    train_final[col] = train_final[col].astype("category")
    if col in val_final.columns:
        val_final[col] = val_final[col].astype("category")
    if col in test_features.columns:
        test_features[col] = test_features[col].astype("category")

    train_final[f"{col}_code"] = train_final[col].cat.codes
    if col in val_final.columns:
        val_final[f"{col}_code"] = val_final[col].cat.codes
    if col in test_features.columns:
        all_cats = list(train_final[col].cat.categories)
        test_features[f"{col}_code"] = (
            test_features[col].astype(object)
            .map({c: i for i, c in enumerate(all_cats)})
            .fillna(-1)
            .astype(int)
        )

    train_final.drop(columns=[col], inplace=True)
    if col in val_final.columns:
        val_final.drop(columns=[col], inplace=True)
    if col in test_features.columns:
        test_features.drop(columns=[col], inplace=True)

# 4. Numeric features (fit on train portion only)
for col in num_cols:
    if col not in train_final.columns:
        continue
    train_final[f"{col}_missing"] = train_final[col].isnull().astype(int)
    if col in val_final.columns:
        val_final[f"{col}_missing"] = val_final[col].isnull().astype(int)
    if col in test_features.columns:
        test_features[f"{col}_missing"] = test_features[col].isnull().astype(int)

    median_val = train_final[col].median()
    train_final[col] = train_final[col].fillna(median_val)
    if col in val_final.columns:
        val_final[col] = val_final[col].fillna(median_val)
    if col in test_features.columns:
        test_features[col] = test_features[col].fillna(median_val)

    if train_final[col].min() >= 0 and train_final[col].skew() > 2:
        train_final[f"{col}_log"] = np.log1p(train_final[col])
        if col in val_final.columns:
            val_final[f"{col}_log"] = np.log1p(val_final[col])
        if col in test_features.columns:
            test_features[f"{col}_log"] = np.log1p(test_features[col])

# 5. Interaction features
available_num = [c for c in num_cols if c in train_final.columns]
if len(available_num) >= 2:
    for i in range(min(len(available_num), 10)):
        for j in range(i + 1, min(len(available_num), 10)):
            c1, c2 = available_num[i], available_num[j]
            if c1 in train_final.columns and c2 in train_final.columns:
                train_final[f"{c1}_x_{c2}"] = train_final[c1] * train_final[c2]
                if c2 in val_final.columns and c1 in val_final.columns:
                    val_final[f"{c1}_x_{c2}"] = val_final[c1] * val_final[c2]
                if c2 in test_features.columns and c1 in test_features.columns:
                    test_features[f"{c1}_x_{c2}"] = (
                        test_features[c1] * test_features[c2]
                    )

# 6. Group size features (use only train portion counts)
group_cols = [c for c in cat_cols if c in train.columns and train[c].nunique() < 50]
for gcol in group_cols:
    if gcol in train.columns and y is not None:
        # Use only the train portion's index for counting
        train_portion_mask = train_final.index
        train_group_counts = train.loc[train_portion_mask, gcol].value_counts()
        train_final[f"{gcol}_group_size"] = train.loc[train_final.index, gcol].map(
            train_group_counts
        ).fillna(0)
        if gcol in val_final.columns:
            val_final[f"{gcol}_group_size"] = train.loc[val_final.index, gcol].map(
                train_group_counts
            ).fillna(0)
        if gcol in test.columns:
            test_features[f"{gcol}_group_size"] = (
                test[gcol].map(train_group_counts).fillna(0)
            )

# 7. Ensure all columns are numeric
for col in train_final.columns:
    if train_final[col].dtype == "object":
        freq_map = train_final[col].value_counts(normalize=True)
        train_final[col] = train_final[col].map(freq_map).fillna(0)
    elif train_final[col].dtype == "bool":
        train_final[col] = train_final[col].astype(int)

for col in val_final.columns:
    if val_final[col].dtype == "object":
        if col in train_final.columns:
            train_unique = train_final[col].unique()
            val_final[col] = (
                val_final[col]
                .map({v: i for i, v in enumerate(train_unique)})
                .fillna(-1)
            )
        else:
            val_final[col] = 0
    elif val_final[col].dtype == "bool":
        val_final[col] = val_final[col].astype(int)

for col in test_features.columns:
    if test_features[col].dtype == "object":
        if col in train_final.columns:
            train_unique = train_final[col].unique()
            test_features[col] = (
                test_features[col]
                .map({v: i for i, v in enumerate(train_unique)})
                .fillna(-1)
            )
        else:
            test_features[col] = 0
    elif test_features[col].dtype == "bool":
        test_features[col] = test_features[col].astype(int)

# Align columns
all_cols = list(train_final.columns)
for col in all_cols:
    if col not in val_final.columns:
        val_final[col] = 0
    if col not in test_features.columns:
        test_features[col] = 0

val_final = val_final[all_cols]
test_features = test_features[all_cols]

print(f"Final train shape: {train_final.shape}")
print(f"Validation shape: {val_final.shape}")
print(f"Test shape: {test_features.shape}")

# ============================================================
# STEP 2: MODEL DESIGN (TabularNet with categorical embeddings)
# ============================================================

feature_cols = list(train_final.columns)
cat_code_cols = [c for c in feature_cols if c.endswith("_code")]
num_feature_cols = [c for c in feature_cols if c not in cat_code_cols]

cat_cardinalities = {}
for col in cat_code_cols:
    max_val = int(train_final[col].max()) + 1
    min_val = int(train_final[col].min())
    cardinality = max_val - min_val
    cat_cardinalities[col] = cardinality

print(f"Numeric features: {len(num_feature_cols)}")
print(f"Categorical features (with embeddings): {len(cat_code_cols)}")

X_train = train_final.values.astype(np.float32)
X_val = val_final.values.astype(np.float32)
X_test = test_features.values.astype(np.float32)

cat_indices = [feature_cols.index(c) for c in cat_code_cols]
num_indices = [feature_cols.index(c) for c in num_feature_cols]

cat_dims = [cat_cardinalities[c] for c in cat_code_cols]
n_num = len(num_indices)
n_cat = len(cat_code_cols)


class TabularNet(nn.Module):
    """Tabular neural network with categorical embeddings and MLP backbone."""

    def __init__(
        self,
        n_num_features,
        cat_dims,
        embedding_dims,
        hidden_dims=[128, 64, 32],
        dropout=0.3,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(dim + 1, ed) for dim, ed in zip(cat_dims, embedding_dims)]
        )

        total_input = n_num_features + sum(embedding_dims)

        layers = []
        prev_size = total_input
        for h in hidden_dims:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = h
        layers.append(nn.Linear(prev_size, 1))
        self.mlp = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, x):
        x_num = x[:, self.num_indices]
        emb_outputs = []
        for i, emb in enumerate(self.embeddings):
            cat_val = x[:, self.cat_indices[i]].long() + 1
            cat_val = cat_val.clamp(0, emb.weight.shape[0] - 1)
            emb_out = emb(cat_val)
            emb_outputs.append(emb_out)
        x_concat = torch.cat([x_num] + emb_outputs, dim=1)
        return self.mlp(x_concat).squeeze(-1)

    def set_indices(self, num_indices, cat_indices):
        self.num_indices = num_indices
        self.cat_indices = cat_indices


def get_embedding_dim(cardinality):
    if cardinality <= 4:
        return 2
    elif cardinality <= 16:
        return 4
    elif cardinality <= 64:
        return 8
    elif cardinality <= 256:
        return 16
    else:
        return 32


embedding_dims = [get_embedding_dim(d) for d in cat_dims]

model = TabularNet(
    n_num_features=n_num,
    cat_dims=cat_dims,
    embedding_dims=embedding_dims,
    hidden_dims=[128, 64, 32],
    dropout=0.3,
)
model.set_indices(num_indices, cat_indices)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train.values.astype(np.float32))
X_val_t = torch.FloatTensor(X_val)
X_test_t = torch.FloatTensor(X_test)

# ============================================================
# STEP 3: TRAINING & EVALUATION
# ============================================================

batch_size = 256
n_epochs = 300
patience = 30

train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
)

best_auc = 0.0
best_model_state = None
patience_counter = 0

for epoch in range(n_epochs):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    scheduler.step()
    avg_loss = total_loss / n_batches

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)
        val_probs = torch.sigmoid(val_logits).numpy()
        val_auc = roc_auc_score(y_val.values, val_probs)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}")

    if val_auc > best_auc:
        best_auc = val_auc
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} (best AUC: {best_auc:.4f})")
            break

print(f"Training complete. Best Val AUC: {best_auc:.4f}")

# Load best model
model.load_state_dict(best_model_state)
model.eval()

# Generate predictions
with torch.no_grad():
    val_probs = torch.sigmoid(model(X_val_t)).numpy()
    test_probs = torch.sigmoid(model(X_test_t)).numpy()

# Compute final validation score using official metric (ROC-AUC)
final_score = roc_auc_score(y_val.values, val_probs)
print(f"Validation ROC-AUC: {final_score:.4f}")

# Save submission
submission = pd.DataFrame(
    {"id": test_ids.astype(int), "requester_received_pizza": test_probs}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")
print(
    f"Prediction stats: mean={test_probs.mean():.4f}, min={test_probs.min():.4f}, max={test_probs.max():.4f}"
)

print(f"Final Validation Score: {final_score}")