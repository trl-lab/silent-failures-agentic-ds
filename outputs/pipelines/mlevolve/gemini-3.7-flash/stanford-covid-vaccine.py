import os
import random
import re
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset


# =========================================================================
# Determinism & Environment Setup
# =========================================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


SEED = 42
seed_everything(SEED)

INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(INPUT_DIR, "train.parquet")
TEST_PATH = os.path.join(INPUT_DIR, "test.parquet")

train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)


# =========================================================================
# Feature Engineering: Biophysical & Structural Descriptors
# =========================================================================
def extract_sequence_features(seq: str) -> Dict[str, float]:
    """Extracts nucleotide composition, skews, entropy, regional, and k-mer statistics."""
    if not isinstance(seq, str) or len(seq) == 0:
        return {}

    n = float(len(seq))
    seq_norm = seq.replace("T", "U")
    counts = {
        "A": seq_norm.count("A"),
        "C": seq_norm.count("C"),
        "G": seq_norm.count("G"),
        "U": seq_norm.count("U"),
    }

    g_plus_c = float(counts["G"] + counts["C"])
    a_plus_u = float(counts["A"] + counts["U"])
    gc_skew = (counts["G"] - counts["C"]) / (g_plus_c + 1e-6)
    au_skew = (counts["A"] - counts["U"]) / (a_plus_u + 1e-6)

    # Shannon entropy of base composition
    probs = [c / n for c in counts.values() if c > 0]
    entropy = -sum(p * np.log2(p + 1e-9) for p in probs)

    feats = {
        "seq_len": n,
        "frac_A": counts["A"] / n,
        "frac_C": counts["C"] / n,
        "frac_G": counts["G"] / n,
        "frac_U": counts["U"] / n,
        "gc_content": g_plus_c / n,
        "gc_skew": gc_skew,
        "au_skew": au_skew,
        "base_entropy": entropy,
        "purine_ratio": (counts["A"] + counts["G"]) / n,
        "pyrimidine_ratio": (counts["C"] + counts["U"]) / n,
        "keto_ratio": (counts["G"] + counts["U"]) / (n + 1e-6),
        "amino_ratio": (counts["A"] + counts["C"]) / (n + 1e-6),
    }

    bases = ["A", "C", "G", "U"]
    denom_di = max(1.0, n - 1.0)
    for b1 in bases:
        for b2 in bases:
            di = b1 + b2
            feats[f"di_{di}"] = seq_norm.count(di) / denom_di

    # Tri-nucleotide (3-mer) frequencies
    denom_tri = max(1.0, n - 2.0)
    for b1 in bases:
        for b2 in bases:
            for b3 in bases:
                tri = b1 + b2 + b3
                feats[f"tri_{tri}"] = seq_norm.count(tri) / denom_tri

    # Positional quartile compositions
    q_len = max(1, int(n // 4))
    for q_idx in range(4):
        q_sub = seq_norm[q_idx * q_len : (q_idx + 1) * q_len if q_idx < 3 else int(n)]
        q_n = max(1.0, float(len(q_sub)))
        for b in bases:
            feats[f"q{q_idx}_{b}"] = q_sub.count(b) / q_n

    return feats


def extract_structure_features(
    structure: str, seq: Optional[str] = None
) -> Dict[str, float]:
    """Extracts RNA secondary structure topology, stem, stacking energetics, and pairing statistics."""
    if not isinstance(structure, str) or len(structure) == 0:
        return {}

    n = float(len(structure))
    paired_count = structure.count("(") + structure.count(")")
    unpaired_count = structure.count(".")

    feats = {
        "struct_paired_frac": paired_count / n,
        "struct_unpaired_frac": unpaired_count / n,
        "struct_num_pairs": paired_count / 2.0,
        "struct_pair_to_unpair_ratio": paired_count / (unpaired_count + 1e-6),
    }

    stack = []
    pair_distances = []
    gc_pairs = 0
    au_pairs = 0
    gu_pairs = 0
    other_pairs = 0

    seq_norm = seq.replace("T", "U") if seq is not None else None

    # Track paired positions across quartiles
    q_len = max(1, int(n // 4))
    quartile_pairs = [0, 0, 0, 0]

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
            q_id = min(3, i // q_len)
            quartile_pairs[q_id] += 1
        elif char == ")":
            q_id = min(3, i // q_len)
            quartile_pairs[q_id] += 1
            if stack:
                j = stack.pop()
                dist = i - j
                pair_distances.append(dist)
                if seq_norm is not None and j < len(seq_norm) and i < len(seq_norm):
                    pair = "".join(sorted([seq_norm[j], seq_norm[i]]))
                    if pair == "CG":
                        gc_pairs += 1
                    elif pair == "AU":
                        au_pairs += 1
                    elif pair == "GU":
                        gu_pairs += 1
                    else:
                        other_pairs += 1

    for q_idx in range(4):
        feats[f"struct_q{q_idx}_paired_frac"] = quartile_pairs[q_idx] / (
            paired_count + 1e-6
        )

    if pair_distances:
        feats["mean_pair_distance"] = float(np.mean(pair_distances))
        feats["std_pair_distance"] = float(np.std(pair_distances))
        feats["max_pair_distance"] = float(np.max(pair_distances))
        feats["min_pair_distance"] = float(np.min(pair_distances))
        feats["median_pair_distance"] = float(np.median(pair_distances))
    else:
        feats["mean_pair_distance"] = 0.0
        feats["std_pair_distance"] = 0.0
        feats["max_pair_distance"] = 0.0
        feats["min_pair_distance"] = 0.0
        feats["median_pair_distance"] = 0.0

    total_bp = max(1.0, float(len(pair_distances)))
    feats["frac_gc_pairs"] = gc_pairs / total_bp
    feats["frac_au_pairs"] = au_pairs / total_bp
    feats["frac_gu_pairs"] = gu_pairs / total_bp
    feats["frac_other_pairs"] = other_pairs / total_bp
    # Thermodynamic pairing energy proxy (GC: 3 H-bonds, AU: 2, GU: 1)
    feats["pairing_energy_score"] = (
        3.0 * gc_pairs + 2.0 * au_pairs + 1.0 * gu_pairs
    ) / total_bp

    stems = [len(m) for m in re.findall(r"\(+|\)+", structure)]
    unpaired_runs = [len(m) for m in re.findall(r"\.+", structure)]

    feats["num_stems"] = float(len(stems))
    feats["mean_stem_len"] = float(np.mean(stems)) if stems else 0.0
    feats["max_stem_len"] = float(np.max(stems)) if stems else 0.0
    feats["stems_ge_3_frac"] = (
        float(sum(1 for s in stems if s >= 3)) / float(len(stems)) if stems else 0.0
    )
    feats["num_unpaired_runs"] = float(len(unpaired_runs))
    feats["mean_unpaired_len"] = float(np.mean(unpaired_runs)) if unpaired_runs else 0.0
    feats["max_unpaired_len"] = float(np.max(unpaired_runs)) if unpaired_runs else 0.0

    return feats


def extract_loop_type_features(
    loop_type: str, seq: Optional[str] = None
) -> Dict[str, float]:
    """Extracts loop type distributions, motif complexity ratios, and sequence-loop interactions."""
    if not isinstance(loop_type, str) or len(loop_type) == 0:
        return {}

    n = float(len(loop_type))
    types = ["S", "M", "I", "B", "H", "E", "X"]
    feats = {}
    counts = {}
    for t in types:
        cnt = loop_type.count(t)
        counts[t] = cnt
        feats[f"loop_frac_{t}"] = cnt / n

    # Structural motif complexity ratios
    unpaired_structural = float(counts["H"] + counts["I"] + counts["B"] + counts["M"])
    feats["motif_hairpin_ratio"] = counts["H"] / (unpaired_structural + 1e-6)
    feats["motif_internal_ratio"] = counts["I"] / (unpaired_structural + 1e-6)
    feats["motif_multiloop_ratio"] = counts["M"] / (unpaired_structural + 1e-6)
    feats["motif_bulge_ratio"] = counts["B"] / (unpaired_structural + 1e-6)
    feats["stem_to_internal_ratio"] = counts["S"] / (counts["I"] + counts["B"] + 1e-6)

    # Loop type run-length statistics for each structural element
    for t in types:
        runs = [len(m) for m in re.findall(f"{t}+", loop_type)]
        feats[f"loop_{t}_num_runs"] = float(len(runs))
        feats[f"loop_{t}_mean_len"] = float(np.mean(runs)) if runs else 0.0
        feats[f"loop_{t}_max_len"] = float(np.max(runs)) if runs else 0.0

    if seq is not None and len(seq) == len(loop_type):
        seq_norm = seq.replace("T", "U")
        for t in types:
            for b in ["A", "C", "G", "U"]:
                match_count = sum(
                    1
                    for s_char, l_char in zip(seq_norm, loop_type)
                    if s_char == b and l_char == t
                )
                feats[f"joint_{b}_{t}"] = match_count / n

    return feats


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Processes a raw dataframe into an engineered tabular feature representation."""
    records = []
    has_seq = "sequence" in df.columns
    has_struct = "structure" in df.columns
    has_loop = "predicted_loop_type" in df.columns

    for _, row in df.iterrows():
        row_feat = {}
        seq = row["sequence"] if has_seq else None
        struct = row["structure"] if has_struct else None
        loop = row["predicted_loop_type"] if has_loop else None

        if has_seq and isinstance(seq, str):
            row_feat.update(extract_sequence_features(seq))
        if has_struct and isinstance(struct, str):
            row_feat.update(extract_structure_features(struct, seq=seq))
        if has_loop and isinstance(loop, str):
            row_feat.update(extract_loop_type_features(loop, seq=seq))

        for col in df.columns:
            if col not in [
                "sequence",
                "structure",
                "predicted_loop_type",
                "id",
                "reactivity",
            ]:
                val = row[col]
                if isinstance(val, (int, float, np.number)):
                    row_feat[f"meta_{col}"] = float(val)

        records.append(row_feat)

    return pd.DataFrame(records, index=df.index)


X_train_raw = process_dataframe(train_df)
X_test_raw = process_dataframe(test_df)

common_cols = sorted(
    list(set(X_train_raw.columns).intersection(set(X_test_raw.columns)))
)
X_train_raw = X_train_raw[common_cols]
X_test_raw = X_test_raw[common_cols]

if "reactivity" in train_df.columns:
    if isinstance(train_df["reactivity"].iloc[0], (list, np.ndarray)):
        y_all = np.array(
            [np.mean(np.array(v, dtype=np.float32)) for v in train_df["reactivity"]],
            dtype=np.float32,
        )
    else:
        y_all = train_df["reactivity"].values.astype(np.float32)
else:
    raise ValueError("Target column 'reactivity' not found in training data.")

# =========================================================================
# PyTorch Dataset Definition
# =========================================================================
class RNADataset(Dataset):
    """PyTorch Dataset for RNA tabular features and reactivity targets."""

    def __init__(self, features: np.ndarray, targets: Optional[np.ndarray] = None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = (
            torch.tensor(targets, dtype=torch.float32) if targets is not None else None
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        return self.features[idx]


# =========================================================================
# Loss Function & Network Architecture
# =========================================================================
class MCRMSELoss(nn.Module):
    """Differentiable Mean Columnwise Root Mean Squared Error (MCRMSE) Loss."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.dim() == 1:
            y_pred = y_pred.view(-1, 1)
        if y_true.dim() == 1:
            y_true = y_true.view(-1, 1)
        mse_per_col = torch.mean((y_pred - y_true) ** 2, dim=0)
        rmse_per_col = torch.sqrt(mse_per_col + self.eps)
        return torch.mean(rmse_per_col)


class GatedLinearUnit(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        val, gate = self.fc(x).chunk(2, dim=-1)
        return val * torch.sigmoid(gate)


class GatedResidualBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        expansion_factor: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.glu = GatedLinearUnit(hidden_dim, hidden_dim * expansion_factor)
        self.dropout = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(hidden_dim * expansion_factor, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm(x)
        out = self.glu(out)
        out = self.dropout(out)
        out = self.proj(out)
        return residual + out


class FeatureSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout_rate: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_seq = x.unsqueeze(1)
        attn_out, _ = self.mha(x_seq, x_seq, x_seq)
        out = self.norm(x + attn_out.squeeze(1))
        return out


class RNAGatedAttentionNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256, 128, 64),
        dropout_rate: float = 0.2,
        num_heads: int = 4,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.5),
        )

        self.attention = FeatureSelfAttention(
            hidden_dim=hidden_dims[0],
            num_heads=num_heads,
            dropout_rate=dropout_rate * 0.5,
        )

        self.res_blocks = nn.ModuleList()
        current_dim = hidden_dims[0]
        for h_dim in hidden_dims:
            if current_dim != h_dim:
                self.res_blocks.append(
                    nn.Sequential(
                        nn.Linear(current_dim, h_dim),
                        nn.LayerNorm(h_dim),
                        nn.SiLU(),
                    )
                )
                current_dim = h_dim
            self.res_blocks.append(
                GatedResidualBlock(
                    hidden_dim=current_dim,
                    expansion_factor=2,
                    dropout_rate=dropout_rate,
                )
            )

        self.head = nn.Sequential(
            nn.LayerNorm(current_dim),
            nn.Linear(current_dim, 32),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(32, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_norm(x)
        h = self.input_proj(h)
        h = self.attention(h)
        for block in self.res_blocks:
            h = block(h)
        out = self.head(h)
        return out.squeeze(-1) if out.shape[-1] == 1 else out


def compute_mcrmse(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Computes exact Mean Columnwise Root Mean Squared Error."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    if y_t.ndim == 1:
        y_t = y_t.reshape(-1, 1)
    if y_p.ndim == 1:
        y_p = y_p.reshape(-1, 1)
    col_rmse = np.sqrt(np.mean((y_t - y_p) ** 2, axis=0) + eps)
    return float(np.mean(col_rmse))


# =========================================================================
# 5-Fold Stratified Cross-Validation & Hybrid Dual-Paradigm Training
# =========================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = len(common_cols)

N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof_preds_nn = np.zeros(len(X_train_raw), dtype=np.float32)
oof_preds_lgb = np.zeros(len(X_train_raw), dtype=np.float32)

test_preds_nn = np.zeros(len(X_test_raw), dtype=np.float32)
test_preds_lgb = np.zeros(len(X_test_raw), dtype=np.float32)

BATCH_SIZE = 64
NUM_EPOCHS = 60
PATIENCE = 15

print("Starting 5-fold cross-validation hybrid ensemble...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_raw, y_all)):
    print(f"\n========== Fold {fold + 1}/{N_SPLITS} ==========")
    seed_everything(SEED + fold)

    X_train_fold = X_train_raw.iloc[train_idx].copy()
    y_train_fold = y_all[train_idx]
    X_val_fold = X_train_raw.iloc[val_idx].copy()
    y_val_fold = y_all[val_idx]

    # Leak-free imputation fitted solely on training fold
    train_medians = X_train_fold.median().fillna(0.0)
    X_train_filled = (
        X_train_fold.replace([np.inf, -np.inf], np.nan)
        .fillna(train_medians)
        .astype(np.float32)
    )
    X_val_filled = (
        X_val_fold.replace([np.inf, -np.inf], np.nan)
        .fillna(train_medians)
        .astype(np.float32)
    )
    X_test_filled = (
        X_test_raw.replace([np.inf, -np.inf], np.nan)
        .fillna(train_medians)
        .astype(np.float32)
    )

    # Leak-free scaling fitted solely on training fold
    scaler = RobustScaler()
    scaler.fit(X_train_filled)

    X_tr = scaler.transform(X_train_filled).astype(np.float32)
    y_tr = y_train_fold.astype(np.float32)

    X_va = scaler.transform(X_val_filled).astype(np.float32)
    y_va = y_val_fold.astype(np.float32)

    X_te = scaler.transform(X_test_filled).astype(np.float32)

    # ---------------------------------------------------------------------
    # Model 1: Gradient Boosted Decision Tree (LightGBM)
    # ---------------------------------------------------------------------
    lgb_model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=10,
        random_state=SEED + fold,
        verbose=-1,
        n_jobs=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=35, verbose=False)],
    )

    lgb_val_pred = lgb_model.predict(X_va)
    lgb_test_pred = lgb_model.predict(X_te)

    oof_preds_lgb[val_idx] = lgb_val_pred
    test_preds_lgb += lgb_test_pred / N_SPLITS

    fold_lgb_score = compute_mcrmse(y_va, lgb_val_pred)
    print(f"Fold {fold + 1} LightGBM MCRMSE: {fold_lgb_score:.5f}")

    # ---------------------------------------------------------------------
    # Model 2: Deep Gated Residual Self-Attention Network
    # ---------------------------------------------------------------------
    train_dataset = RNADataset(X_tr, y_tr)
    val_dataset = RNADataset(X_va, y_va)
    test_dataset = RNADataset(X_te)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )

    nn_model = RNAGatedAttentionNet(
        input_dim=input_dim,
        hidden_dims=(256, 256, 128, 64),
        dropout_rate=0.2,
        num_heads=4,
        output_dim=1,
    ).to(device)

    criterion = MCRMSELoss()
    decay_params = []
    no_decay_params = []
    for name, param in nn_model.named_parameters():
        if not param.requires_grad:
            continue
        if "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": 1e-4},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=1e-3,
        eps=1e-8,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    fold_best_score = float("inf")
    fold_model_path = os.path.join(WORKING_DIR, f"best_rna_model_fold_{fold}.pt")
    patience_cnt = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        nn_model.train()
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = nn_model(x_b)
            loss = criterion(preds, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        nn_model.eval()
        vpreds = []
        with torch.no_grad():
            for x_b, _ in val_loader:
                x_b = x_b.to(device)
                preds = nn_model(x_b)
                vpreds.append(preds.detach().cpu().numpy())

        vpreds_arr = np.concatenate(vpreds, axis=0).flatten()
        val_score = compute_mcrmse(y_va, vpreds_arr)

        if val_score < fold_best_score:
            fold_best_score = val_score
            patience_cnt = 0
            torch.save(nn_model.state_dict(), fold_model_path)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    # Load best NN checkpoint
    nn_model.load_state_dict(torch.load(fold_model_path, map_location=device))
    nn_model.eval()

    best_vpreds = []
    with torch.no_grad():
        for x_b, _ in val_loader:
            x_b = x_b.to(device)
            preds = nn_model(x_b)
            best_vpreds.append(preds.detach().cpu().numpy())

    best_vpreds_arr = np.concatenate(best_vpreds, axis=0).flatten()
    oof_preds_nn[val_idx] = best_vpreds_arr

    best_tpreds = []
    with torch.no_grad():
        for x_b in test_loader:
            x_b = x_b.to(device)
            preds = nn_model(x_b)
            best_tpreds.append(preds.detach().cpu().numpy())

    best_tpreds_arr = np.concatenate(best_tpreds, axis=0).flatten()
    test_preds_nn += best_tpreds_arr / N_SPLITS

    print(f"Fold {fold + 1} Neural Net MCRMSE: {fold_best_score:.5f}")


# =========================================================================
# Out-of-Fold Convex Optimization & Ensembling
# =========================================================================
overall_nn_score = compute_mcrmse(y_all, oof_preds_nn)
overall_lgb_score = compute_mcrmse(y_all, oof_preds_lgb)

print(f"\nOverall OOF NN MCRMSE: {overall_nn_score:.5f}")
print(f"Overall OOF LightGBM MCRMSE: {overall_lgb_score:.5f}")


def blend_objective(w: float) -> float:
    blend = w * oof_preds_nn + (1.0 - w) * oof_preds_lgb
    return compute_mcrmse(y_all, blend)


opt_res = minimize_scalar(blend_objective, bounds=(0.0, 1.0), method="bounded")
best_w = float(opt_res.x) if opt_res.success else 0.5

oof_blend = best_w * oof_preds_nn + (1.0 - best_w) * oof_preds_lgb
final_val_score = compute_mcrmse(y_all, oof_blend)

print(f"Optimal NN Weight: {best_w:.4f} | LGBM Weight: {1.0 - best_w:.4f}")
print(f"Final Validation Score: {final_val_score}")


# =========================================================================
# Final Inference & Submission Generation
# =========================================================================
final_test_preds = best_w * test_preds_nn + (1.0 - best_w) * test_preds_lgb

if "id" in test_df.columns:
    test_ids = test_df["id"].values
else:
    test_ids = np.arange(len(test_df), dtype=np.int64)

submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "reactivity": final_test_preds,
    }
)

submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df.to_csv(submission_path, index=False)
