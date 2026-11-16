import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


# Set random seeds for deterministic reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)

# Ensure required working and submission output directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Select computation device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load raw dataset files
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Determine maximum sequence length across train and test sets
max_len = max(train_df["seq_length"].max(), test_df["seq_length"].max())


def parse_structure(structure):
    """Parses secondary structure dot-bracket notation to obtain base pair partner indices."""
    n = len(structure)
    bp = np.full(n, -1, dtype=int)
    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                bp[i] = j
                bp[j] = i
    return bp


def extract_features(df, max_len):
    """Extracts 1D sequence, positional, structural, and 2D graph adjacency features."""
    n_samples = len(df)

    nuc_map = {"A": 0, "C": 1, "G": 2, "U": 3}
    struct_map = {".": 0, "(": 1, ")": 2}
    loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    seq_enc = np.zeros((n_samples, max_len), dtype=np.int64)
    struct_enc = np.zeros((n_samples, max_len), dtype=np.int64)
    loop_enc = np.zeros((n_samples, max_len), dtype=np.int64)
    pair_type_enc = np.zeros((n_samples, max_len), dtype=np.int64)
    triplet_enc = np.zeros((n_samples, max_len), dtype=np.int64)

    bp_offsets = np.zeros((n_samples, max_len), dtype=np.float32)
    bp_dists = np.zeros((n_samples, max_len), dtype=np.float32)
    pos_norms = np.zeros((n_samples, max_len), dtype=np.float32)
    scored_masks = np.zeros((n_samples, max_len), dtype=np.float32)

    adj_matrices = np.zeros((n_samples, max_len, max_len), dtype=np.float32)
    cont_features = np.zeros((n_samples, max_len, 12), dtype=np.float32)

    for idx, row in df.reset_index(drop=True).iterrows():
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]
        seq_len = row["seq_length"]
        seq_scored = row["seq_scored"]

        bp_partner = parse_structure(struct)

        for i in range(seq_len):
            nuc = seq[i]
            s_char = struct[i]
            l_char = loop[i]

            seq_enc[idx, i] = nuc_map.get(nuc, 4)
            struct_enc[idx, i] = struct_map.get(s_char, 3)
            loop_enc[idx, i] = loop_map.get(l_char, 7)

            prev_nuc = nuc_map.get(seq[i - 1], 4) if i > 0 else 4
            curr_nuc = nuc_map.get(nuc, 4)
            next_nuc = nuc_map.get(seq[i + 1], 4) if i < seq_len - 1 else 4
            triplet_enc[idx, i] = prev_nuc * 25 + curr_nuc * 5 + next_nuc

            j = bp_partner[i]
            p_type = 0
            if j != -1:
                bp_offsets[idx, i] = (j - i) / float(max_len)
                bp_dists[idx, i] = abs(j - i) / float(max_len)
                pair_str = (seq[i] + seq[j]).upper()
                if pair_str in ["AU", "UA"]:
                    p_type = 1
                elif pair_str in ["GC", "CG"]:
                    p_type = 2
                elif pair_str in ["GU", "UG"]:
                    p_type = 3
                else:
                    p_type = 4
            pair_type_enc[idx, i] = p_type

            pos_norms[idx, i] = i / float(seq_len)
            if i < seq_scored:
                scored_masks[idx, i] = 1.0

            # Construct 2D adjacency matrix (Self-loops + Sequence Backbone + Secondary Base Pairs)
            adj_matrices[idx, i, i] = 1.0
            if i + 1 < seq_len:
                adj_matrices[idx, i, i + 1] = 1.0
                adj_matrices[idx, i + 1, i] = 1.0
            if j != -1:
                adj_matrices[idx, i, j] = 1.0

            # Construct continuous positional feature array
            cont_features[idx, i, 0] = seq_enc[idx, i]
            cont_features[idx, i, 1] = struct_enc[idx, i]
            cont_features[idx, i, 2] = loop_enc[idx, i]
            cont_features[idx, i, 3] = pair_type_enc[idx, i]
            cont_features[idx, i, 4] = triplet_enc[idx, i]
            cont_features[idx, i, 5] = bp_offsets[idx, i]
            cont_features[idx, i, 6] = bp_dists[idx, i]
            cont_features[idx, i, 7] = pos_norms[idx, i]
            cont_features[idx, i, 8] = (seq_len - 1 - i) / float(seq_len)
            cont_features[idx, i, 9] = 1.0 if j != -1 else 0.0
            cont_features[idx, i, 10] = prev_nuc
            cont_features[idx, i, 11] = next_nuc

    return {
        "seq": seq_enc,
        "struct": struct_enc,
        "loop": loop_enc,
        "pair_type": pair_type_enc,
        "triplet": triplet_enc,
        "bp_offsets": bp_offsets,
        "bp_dists": bp_dists,
        "pos_norms": pos_norms,
        "scored_mask": scored_masks,
        "adj": adj_matrices,
        "cont_features": cont_features,
    }


def extract_targets(df, max_len):
    """Extracts reactivity target vector padded to max_len."""
    n_samples = len(df)
    y = np.zeros((n_samples, max_len), dtype=np.float32)

    for idx, row in df.reset_index(drop=True).iterrows():
        if "reactivity" in row and row["reactivity"] is not None:
            vals = np.array(row["reactivity"], dtype=np.float32)
            L = min(len(vals), max_len)
            y[idx, :L] = vals[:L]
    return y


# Quality filter training samples if signal-to-noise filter column is present
if "SN_filter" in train_df.columns:
    filtered_df = train_df[train_df["SN_filter"] == 1].reset_index(drop=True)
    if len(filtered_df) > 500:
        train_df = filtered_df

# Sequence-level train/validation split (80/20) with strict isolation before scaling
train_data_df, val_data_df = train_test_split(train_df, test_size=0.2, random_state=42)
train_data_df = train_data_df.reset_index(drop=True)
val_data_df = val_data_df.reset_index(drop=True)

# Extract features
train_feat = extract_features(train_data_df, max_len)
val_feat = extract_features(val_data_df, max_len)
test_feat = extract_features(test_df, max_len)

# Extract targets
y_train = extract_targets(train_data_df, max_len)
y_val = extract_targets(val_data_df, max_len)

# Fit StandardScaler ONLY on train set continuous features to avoid data leakage
N_tr, L_tr, C_tr = train_feat["cont_features"].shape
scaler = StandardScaler()
X_train_cont_scaled = scaler.fit_transform(
    train_feat["cont_features"].reshape(-1, C_tr)
).reshape(N_tr, L_tr, C_tr)

N_val, L_val, C_val = val_feat["cont_features"].shape
X_val_cont_scaled = scaler.transform(
    val_feat["cont_features"].reshape(-1, C_val)
).reshape(N_val, L_val, C_val)

N_te, L_te, C_te = test_feat["cont_features"].shape
X_test_cont_scaled = scaler.transform(
    test_feat["cont_features"].reshape(-1, C_te)
).reshape(N_te, L_te, C_te)


# PyTorch Dataset implementation
class RNADataset(Dataset):

    def __init__(self, seq, struct, loop, cont, adj, mask, targets=None):
        self.seq = torch.tensor(seq, dtype=torch.long)
        self.struct = torch.tensor(struct, dtype=torch.long)
        self.loop = torch.tensor(loop, dtype=torch.long)
        self.cont = torch.tensor(cont, dtype=torch.float32)
        self.adj = torch.tensor(adj, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.targets = (
            torch.tensor(targets, dtype=torch.float32) if targets is not None else None
        )

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        item = {
            "seq": self.seq[idx],
            "struct": self.struct[idx],
            "loop": self.loop[idx],
            "cont": self.cont[idx],
            "adj": self.adj[idx],
            "mask": self.mask[idx],
        }
        if self.targets is not None:
            item["target"] = self.targets[idx]
        return item


train_dataset = RNADataset(
    train_feat["seq"],
    train_feat["struct"],
    train_feat["loop"],
    X_train_cont_scaled,
    train_feat["adj"],
    train_feat["scored_mask"],
    y_train,
)
val_dataset = RNADataset(
    val_feat["seq"],
    val_feat["struct"],
    val_feat["loop"],
    X_val_cont_scaled,
    val_feat["adj"],
    val_feat["scored_mask"],
    y_val,
)
test_dataset = RNADataset(
    test_feat["seq"],
    test_feat["struct"],
    test_feat["loop"],
    X_test_cont_scaled,
    test_feat["adj"],
    test_feat["scored_mask"],
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)


# Graph Convolution Layer
class GraphConvLayer(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.SiLU()

    def forward(self, x, adj):
        deg = torch.sum(adj, dim=-1, keepdim=True) + 1e-8
        adj_norm = adj / deg
        out = torch.bmm(adj_norm, x)
        out = self.linear(out)
        return self.activation(out)


# Hybrid GCN-BiGRU-Conv1D-SelfAttention Deep Architecture
class RNADegradationModel(nn.Module):

    def __init__(
        self,
        seq_vocab=5,
        struct_vocab=4,
        loop_vocab=8,
        cont_dim=12,
        embed_dim=32,
        hidden_dim=256,
        dropout=0.25,
        max_len=500,
    ):
        super().__init__()

        self.seq_embed = nn.Embedding(seq_vocab, embed_dim)
        self.struct_embed = nn.Embedding(struct_vocab, embed_dim // 2)
        self.loop_embed = nn.Embedding(loop_vocab, embed_dim // 2)

        total_in_dim = embed_dim + embed_dim // 2 + embed_dim // 2 + cont_dim
        self.in_proj = nn.Linear(total_in_dim, hidden_dim)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)

        # 2D Graph Convolutions for secondary base-pair connectivity
        self.gcn1 = GraphConvLayer(hidden_dim, hidden_dim)
        self.gcn2 = GraphConvLayer(hidden_dim, hidden_dim)

        # Multi-scale 1D Spatial Convolutions
        self.conv_k3 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.conv_k5 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=5, padding=2)

        self.norm_conv = nn.LayerNorm(hidden_dim)
        self.drop_conv = nn.Dropout(dropout)

        # Bidirectional Recurrent Backbone
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # Multi-Head Self-Attention Block after BiGRU
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2, num_heads=8, batch_first=True, dropout=dropout
        )
        self.norm_attn = nn.LayerNorm(hidden_dim * 2)
        self.drop_attn = nn.Dropout(dropout)

        # Output prediction head
        self.out_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, seq, struct, loop, cont, adj):
        emb_seq = self.seq_embed(seq)
        emb_struct = self.struct_embed(struct)
        emb_loop = self.loop_embed(loop)

        x = torch.cat([emb_seq, emb_struct, emb_loop, cont], dim=-1)
        x = F.silu(self.in_proj(x))

        pos_ids = torch.arange(seq.size(1), device=seq.device).unsqueeze(0)
        x = x + self.pos_embed(pos_ids)

        g_feat = self.gcn1(x, adj)
        g_feat = self.gcn2(g_feat, adj)
        x = x + g_feat

        x_trans = x.transpose(1, 2)
        c3 = F.silu(self.conv_k3(x_trans))
        c5 = F.silu(self.conv_k5(x_trans))
        c_concat = torch.cat([c3, c5], dim=1).transpose(1, 2)

        x = self.norm_conv(x + self.drop_conv(c_concat))

        gru_out, _ = self.gru(x)
        attn_out, _ = self.mha(gru_out, gru_out, gru_out)
        x_attn = self.norm_attn(gru_out + self.drop_attn(attn_out))
        out = self.out_head(x_attn).squeeze(-1)
        return out


# Masked MCRMSE Loss matching competition metric
class MaskedMCRMSELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, preds, targets, mask):
        diff_sq = ((preds - targets) ** 2) * mask
        loss = torch.sqrt(torch.sum(diff_sq) / (torch.sum(mask) + 1e-8) + 1e-8)
        return loss


# Instantiate Model, Loss, Optimizer, and Learning Rate Scheduler
model = RNADegradationModel().to(device)
criterion = MaskedMCRMSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

epochs = 50
warmup_epochs = 5
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs - warmup_epochs, eta_min=1e-5
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_epochs],
)
best_val_score = float("inf")
best_model_path = "./working/best_rna_model.pt"

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    total_batches = 0

    for batch in train_loader:
        seq = batch["seq"].to(device)
        struct = batch["struct"].to(device)
        loop = batch["loop"].to(device)
        cont = batch["cont"].to(device)
        adj = batch["adj"].to(device)
        mask = batch["mask"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()
        preds = model(seq, struct, loop, cont, adj)
        loss = criterion(preds, target, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        total_batches += 1

    scheduler.step()
    train_loss /= total_batches

    # Validation Phase
    model.eval()
    val_sq_diff_sum = 0.0
    val_mask_sum = 0.0

    with torch.no_grad():
        for batch in val_loader:
            seq = batch["seq"].to(device)
            struct = batch["struct"].to(device)
            loop = batch["loop"].to(device)
            cont = batch["cont"].to(device)
            adj = batch["adj"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)

            preds = model(seq, struct, loop, cont, adj)
            diff_sq = ((preds - target) ** 2) * mask

            val_sq_diff_sum += torch.sum(diff_sq).item()
            val_mask_sum += torch.sum(mask).item()

    val_mcrmse = float(np.sqrt(val_sq_diff_sum / (val_mask_sum + 1e-8)))
    print(
        f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.5f} | Val"
        f" MCRMSE: {val_mcrmse:.5f}"
    )

    if val_mcrmse < best_val_score:
        best_val_score = val_mcrmse
        torch.save(model.state_dict(), best_model_path)

# Load best checkpoint for test set inference
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path))

model.eval()
test_preds_list = []

with torch.no_grad():
    for batch in test_loader:
        seq = batch["seq"].to(device)
        struct = batch["struct"].to(device)
        loop = batch["loop"].to(device)
        cont = batch["cont"].to(device)
        adj = batch["adj"].to(device)

        preds = model(seq, struct, loop, cont, adj)
        test_preds_list.append(preds.cpu().numpy())

test_preds = np.vstack(test_preds_list).reshape(-1)

# Format test predictions and write to submission file
sub_df = pd.DataFrame({"id": np.arange(len(test_preds)), "reactivity": test_preds})
sub_df.to_csv("./submission/submission.csv", index=False)

# Final required validation score log output
print(f"Final Validation Score: {best_val_score}")
