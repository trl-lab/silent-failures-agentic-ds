import gc
import os
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
import torch.optim as optim

# =========================================================================
# 0. Environment & Directory Setup
# =========================================================================
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================================
# 1. Data Loading & Feature Engineering
# =========================================================================
print("Loading train and test datasets...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify columns
text_col = "comment_text" if "comment_text" in train_df.columns else "text"
target_col = "target"
id_col = "id" if "id" in test_df.columns else test_df.columns[0]

train_df[text_col] = train_df[text_col].fillna("").astype(str)
test_df[text_col] = test_df[text_col].fillna("").astype(str)


def extract_dense_features(texts):
    """Extract structural and linguistic surface features from text."""
    text_list = list(texts)
    char_lens = np.array([len(t) for t in text_list], dtype=np.float32)
    words = [t.split() for t in text_list]
    word_lens = np.array([len(w) for w in words], dtype=np.float32)
    avg_word_lens = char_lens / np.maximum(word_lens, 1.0)

    caps_counts = np.array(
        [sum(1 for c in t if c.isupper()) for t in text_list], dtype=np.float32
    )
    caps_ratios = caps_counts / np.maximum(char_lens, 1.0)

    excl_counts = np.array([t.count("!") for t in text_list], dtype=np.float32)
    quest_counts = np.array([t.count("?") for t in text_list], dtype=np.float32)

    punct_chars = set("!@#$%^&*()_+-=[]{}|;:\",.<>?/`~\\'")
    punct_counts = np.array(
        [sum(1 for c in t if c in punct_chars) for t in text_list],
        dtype=np.float32,
    )
    punct_ratios = punct_counts / np.maximum(char_lens, 1.0)

    digit_counts = np.array(
        [sum(1 for c in t if c.isdigit()) for t in text_list], dtype=np.float32
    )
    digit_ratios = digit_counts / np.maximum(char_lens, 1.0)

    upper_word_counts = np.array(
        [sum(1 for w in w_list if len(w) > 1 and w.isupper()) for w_list in words],
        dtype=np.float32,
    )
    upper_word_ratios = upper_word_counts / np.maximum(word_lens, 1.0)

    newline_counts = np.array([t.count("\n") for t in text_list], dtype=np.float32)
    curse_symbol_counts = np.array(
        [sum(1 for c in t if c in "*$#@!%") for t in text_list],
        dtype=np.float32,
    )

    dense_matrix = np.column_stack(
        [
            char_lens,
            word_lens,
            avg_word_lens,
            caps_counts,
            caps_ratios,
            excl_counts,
            quest_counts,
            punct_counts,
            punct_ratios,
            digit_counts,
            digit_ratios,
            upper_word_counts,
            upper_word_ratios,
            newline_counts,
            curse_symbol_counts,
        ]
    )
    return dense_matrix


# Target-stratified validation partition
print("Splitting dataset into train and validation sets...")
y = train_df[target_col].values.astype(np.float32)
strat_bins = np.digitize(y, bins=[0.0, 0.05, 0.15, 0.3, 0.5, 0.7, 1.0])

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.10,
    random_state=SEED,
    stratify=strat_bins,
)

train_split_df = train_df.iloc[train_idx].reset_index(drop=True)
val_split_df = train_df.iloc[val_idx].reset_index(drop=True)

y_train = train_split_df[target_col].values.astype(np.float32)
y_val = val_split_df[target_col].values.astype(np.float32)
test_ids = test_df[id_col].values

# Extract and scale dense features
print("Extracting dense linguistic features...")
dense_train = extract_dense_features(train_split_df[text_col])
dense_val = extract_dense_features(val_split_df[text_col])
dense_test = extract_dense_features(test_df[text_col])

scaler = RobustScaler()
dense_train_scaled = scaler.fit_transform(dense_train).astype(np.float32)
dense_val_scaled = scaler.transform(dense_val).astype(np.float32)
dense_test_scaled = scaler.transform(dense_test).astype(np.float32)

# TF-IDF representations (fit exclusively on train split)
print("Extracting word and character TF-IDF representations...")
word_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=40000,
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"\w{1,}",
    min_df=3,
    dtype=np.float32,
)

char_vectorizer = TfidfVectorizer(
    ngram_range=(2, 5),
    max_features=35000,
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="char_wb",
    min_df=5,
    dtype=np.float32,
)

word_vectorizer.fit(train_split_df[text_col])
char_vectorizer.fit(train_split_df[text_col])

train_word = word_vectorizer.transform(train_split_df[text_col])
val_word = word_vectorizer.transform(val_split_df[text_col])
test_word = word_vectorizer.transform(test_df[text_col])

train_char = char_vectorizer.transform(train_split_df[text_col])
val_char = char_vectorizer.transform(val_split_df[text_col])
test_char = char_vectorizer.transform(test_df[text_col])

# Stack all features into final CSR sparse matrices
print("Stacking feature representations...")
X_train_sparse = sp.hstack(
    [train_word, train_char, sp.csr_matrix(dense_train_scaled)], format="csr"
)
X_val_sparse = sp.hstack(
    [val_word, val_char, sp.csr_matrix(dense_val_scaled)], format="csr"
)
X_test_sparse = sp.hstack(
    [test_word, test_char, sp.csr_matrix(dense_test_scaled)], format="csr"
)

del (
    train_df,
    train_split_df,
    val_split_df,
    test_df,
    dense_train,
    dense_val,
    dense_test,
)
del (
    train_word,
    val_word,
    test_word,
    train_char,
    val_char,
    test_char,
    dense_train_scaled,
    dense_val_scaled,
    dense_test_scaled,
)
gc.collect()

print(
    f"Preprocessed train shape: {X_train_sparse.shape}, val shape: {X_val_sparse.shape}, test shape: {X_test_sparse.shape}"
)


# =========================================================================
# 2. Model Architecture Definition
# =========================================================================
class SparseToxicityMLP(nn.Module):
    """Deep Multi-Layer Perceptron optimized for high-dimensional sparse text

    and dense linguistic tabular features with continuous target
    probabilities.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128],
        dropout_rate: float = 0.3,
    ):
        super(SparseToxicityMLP, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.hidden_layer = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2.0),
        )

        self.output_head = nn.Linear(hidden_dims[1], 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.input_layer(x)
        h2 = self.hidden_layer(h1)
        logits = self.output_head(h2).squeeze(-1)
        return logits


def build_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 6,
    steps_per_epoch: int = 100,
):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
    total_steps = epochs * steps_per_epoch
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )

    return optimizer, scheduler


def predict_sparse_dataset(
    model: nn.Module, X_sparse: sp.csr_matrix, batch_size: int = 4096
) -> np.ndarray:
    """Chunked forward inference over sparse CSR matrices."""
    model.eval()
    num_samples = X_sparse.shape[0]
    all_preds = np.zeros(num_samples, dtype=np.float32)

    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_dense = X_sparse[start_idx:end_idx].toarray()
            batch_tensor = torch.from_numpy(batch_dense).float().to(device)
            logits = model(batch_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds[start_idx:end_idx] = probs

    return all_preds


# =========================================================================
# 3. Model Training & Validation Loop
# =========================================================================
input_dim = X_train_sparse.shape[1]
n_train_samples = X_train_sparse.shape[0]
batch_size = 2048
epochs = 6
steps_per_epoch = int(np.ceil(n_train_samples / batch_size))

model = SparseToxicityMLP(
    input_dim=input_dim, hidden_dims=[256, 128], dropout_rate=0.3
).to(device)
criterion = nn.BCEWithLogitsLoss()

optimizer, scheduler = build_optimizer_and_scheduler(
    model=model,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
)

best_val_auc = 0.0
best_model_path = "./working/best_sparse_mlp.pt"
y_val_binary = (y_val >= 0.5).astype(int)

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    permutation = np.random.permutation(n_train_samples)

    for start_idx in range(0, n_train_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_train_samples)
        batch_indices = permutation[start_idx:end_idx]

        batch_dense = X_train_sparse[batch_indices].toarray()
        batch_targets = y_train[batch_indices]

        inputs = torch.from_numpy(batch_dense).float().to(device)
        targets = torch.from_numpy(batch_targets).float().to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * (end_idx - start_idx)

    train_loss = running_loss / n_train_samples

    # Validation evaluation
    val_preds = predict_sparse_dataset(model, X_val_sparse, batch_size=4096)
    val_auc = roc_auc_score(y_val_binary, val_preds)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), best_model_path)

    print(
        f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} - Val ROC-AUC: {val_auc:.5f} - Best ROC-AUC: {best_val_auc:.5f}"
    )

# =========================================================================
# 4. Final Evaluation & Submission Generation
# =========================================================================
# Load best checkpoint
model.load_state_dict(torch.load(best_model_path))
model.eval()

# Final validation score
final_val_preds = predict_sparse_dataset(model, X_val_sparse, batch_size=4096)
final_val_auc = roc_auc_score(y_val_binary, final_val_preds)

# Test predictions
test_preds = predict_sparse_dataset(model, X_test_sparse, batch_size=4096)

# Generate submission file
submission_df = pd.DataFrame({"id": test_ids, "target": test_preds.astype(float)})
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_val_auc}")
