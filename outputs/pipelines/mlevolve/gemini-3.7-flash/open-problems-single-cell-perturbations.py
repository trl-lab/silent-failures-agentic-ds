import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# Ensure required output directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


# ==============================================================================
# 1. EVALUATION METRIC & LOSS FUNCTION
# ==============================================================================
class MRRMSELoss(nn.Module):
    """
    Differentiable Mean Rowwise Root Mean Squared Error (MRRMSE) loss.
    Directly optimizes the competition's evaluation metric.

    For each sample i across M target genes:
        RMSE_i = sqrt( (1 / M) * sum_j (y_ij - y_hat_ij)^2 + eps )
        MRRMSE = (1 / N) * sum_i RMSE_i
    """

    def __init__(self, eps: float = 1e-8):
        super(MRRMSELoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        rowwise_mse = torch.mean((y_pred - y_true) ** 2, dim=1)
        rowwise_rmse = torch.sqrt(rowwise_mse + self.eps)
        return torch.mean(rowwise_rmse)


def compute_mrrmse_metric(
    y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12
) -> float:
    """
    Computes exact competition metric MRRMSE on NumPy arrays.
    """
    rowwise_mse = np.mean((y_pred - y_true) ** 2, axis=1)
    rowwise_rmse = np.sqrt(rowwise_mse + eps)
    return float(np.mean(rowwise_rmse))


# ==============================================================================
# 2. MODEL ARCHITECTURE
# ==============================================================================
class ResidualBlock(nn.Module):
    """
    Residual Block with LayerNorm, SiLU activation, and Dropout for high-dimensional representation learning.
    """

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.25):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class PerturbationResNet(nn.Module):
    """
    Deep Residual Multi-Output Architecture for Single-Cell Perturbation Prediction.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 18211,
        hidden_dim: int = 1024,
        latent_dim: int = 512,
        num_res_blocks: int = 3,
        dropout_rate: float = 0.25,
    ):
        super(PerturbationResNet, self).__init__()

        # Input feature projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        # Residual backbone blocks
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_res_blocks)
            ]
        )

        # Latent co-expression bottleneck
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
        )

        # Gene prediction head
        self.decoder_head = nn.Linear(latent_dim, out_features)

        # Direct linear residual shortcut from inputs to outputs for basic additive effects
        self.linear_shortcut = nn.Linear(in_features, out_features)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.linear_shortcut.weight, std=1e-4)
        if self.linear_shortcut.bias is not None:
            nn.init.zeros_(self.linear_shortcut.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.res_blocks:
            h = block(h)

        latent = self.bottleneck(h)
        out_nonlinear = self.decoder_head(latent)
        out_linear = self.linear_shortcut(x)

        return out_nonlinear + out_linear


def build_model(
    in_features: int,
    out_features: int = 18211,
    hidden_dim: int = 1024,
    latent_dim: int = 512,
    num_res_blocks: int = 3,
    dropout_rate: float = 0.25,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> nn.Module:
    model = PerturbationResNet(
        in_features=in_features,
        out_features=out_features,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_res_blocks=num_res_blocks,
        dropout_rate=dropout_rate,
    )
    model.to(device)
    return model


def build_loss_and_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 50,
    eta_min: float = 1e-6,
):
    criterion = MRRMSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    return criterion, optimizer, scheduler


# ==============================================================================
# 3. FEATURE EXTRACTION & DATA PROCESSING
# ==============================================================================
def extract_smiles_features(smiles_series):
    """
    Extract chemical descriptors and molecular properties from SMILES strings.
    """
    n_samples = len(smiles_series)
    rdkit_fps = []
    rdkit_descs = []
    has_rdkit = False

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors

        has_rdkit = True
    except ImportError:
        has_rdkit = False

    if has_rdkit:
        for smi in smiles_series:
            mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
            if mol is not None:
                fp2 = np.array(
                    AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024),
                    dtype=np.float32,
                )
                fp3 = np.array(
                    AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=512),
                    dtype=np.float32,
                )
                combined_fp = np.concatenate([fp2, fp3])

                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                tpsa = Descriptors.TPSA(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                rot = Descriptors.NumRotatableBonds(mol)
                rings = Descriptors.RingCount(mol)
                fsp3 = Descriptors.FractionCSP3(mol)
                qnum = Descriptors.HeavyAtomCount(mol)
                descs = np.array(
                    [mw, logp, tpsa, hbd, hba, rot, rings, fsp3, qnum], dtype=np.float32
                )
            else:
                combined_fp = np.zeros(1024 + 512, dtype=np.float32)
                descs = np.zeros(9, dtype=np.float32)
            rdkit_fps.append(combined_fp)
            rdkit_descs.append(descs)

        rdkit_fps = np.vstack(rdkit_fps)
        rdkit_descs = np.vstack(rdkit_descs)
    else:
        rdkit_fps = np.zeros((n_samples, 0), dtype=np.float32)
        rdkit_descs = np.zeros((n_samples, 0), dtype=np.float32)

    manual_features = []
    atoms = [
        "C",
        "N",
        "O",
        "S",
        "P",
        "F",
        "Cl",
        "Br",
        "I",
        "c",
        "n",
        "o",
        "s",
        "=",
        "#",
        "(",
        ")",
        "[",
        "]",
    ]
    for smi in smiles_series:
        smi_str = str(smi) if smi is not None else ""
        atom_counts = [smi_str.count(a) for a in atoms]
        atom_counts.append(len(smi_str))
        atom_counts.append(smi_str.count("@"))
        atom_counts.append(smi_str.count("+") - smi_str.count("-"))
        manual_features.append(atom_counts)
    manual_features = np.array(manual_features, dtype=np.float32)

    return rdkit_fps, rdkit_descs, manual_features


def prepare_datasets():
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"
    sample_sub_path = "./input/sample_submission.csv"

    print(f"Loading data from {train_path} and {test_path}...")
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    metadata_cols = ["cell_type", "sm_name", "sm_lincs_id", "SMILES", "control"]
    target_cols = [c for c in df_train.columns if c not in metadata_cols]

    if os.path.exists(sample_sub_path):
        sample_sub = pd.read_csv(sample_sub_path, nrows=1)
        sub_gene_cols = [c for c in sample_sub.columns if c != "id"]
        if len(sub_gene_cols) == len(target_cols):
            target_cols = sub_gene_cols

    print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")
    print(f"Number of target genes: {len(target_cols)}")

    Y_train = df_train[target_cols].values.astype(np.float32)

    # 1. SMILES Sub-structural TF-IDF
    tfidf = TfidfVectorizer(
        analyzer="char", ngram_range=(1, 4), max_features=1024, sublinear_tf=True
    )
    tfidf.fit(df_train["SMILES"].fillna("").astype(str))

    tfidf_train = (
        tfidf.transform(df_train["SMILES"].fillna("").astype(str))
        .toarray()
        .astype(np.float32)
    )
    tfidf_test = (
        tfidf.transform(df_test["SMILES"].fillna("").astype(str))
        .toarray()
        .astype(np.float32)
    )

    # 2. Chemical Properties & Fingerprints
    rdkit_fp_tr, rdkit_desc_tr, manual_tr = extract_smiles_features(df_train["SMILES"])
    rdkit_fp_te, rdkit_desc_te, manual_te = extract_smiles_features(df_test["SMILES"])

    cont_desc_tr = np.hstack([rdkit_desc_tr, manual_tr])
    cont_desc_te = np.hstack([rdkit_desc_te, manual_te])

    scaler_desc = StandardScaler()
    cont_desc_tr = scaler_desc.fit_transform(cont_desc_tr).astype(np.float32)
    cont_desc_te = scaler_desc.transform(cont_desc_te).astype(np.float32)

    # 3. Categorical Cell Type One-Hot Encoding
    ohe_cell = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cell_tr = ohe_cell.fit_transform(df_train[["cell_type"]]).astype(np.float32)
    cell_te = ohe_cell.transform(df_test[["cell_type"]]).astype(np.float32)

    # 4. Compound Occurrence Frequency
    compound_counts = df_train["sm_name"].value_counts()
    sm_freq_tr = (
        df_train["sm_name"]
        .map(compound_counts)
        .fillna(0)
        .values.reshape(-1, 1)
        .astype(np.float32)
    )
    sm_freq_te = (
        df_test["sm_name"]
        .map(compound_counts)
        .fillna(0)
        .values.reshape(-1, 1)
        .astype(np.float32)
    )

    scaler_freq = StandardScaler()
    sm_freq_tr = scaler_freq.fit_transform(sm_freq_tr).astype(np.float32)
    sm_freq_te = scaler_freq.transform(sm_freq_te).astype(np.float32)

    # Combine all feature components
    features_tr = [cell_tr, tfidf_train, cont_desc_tr, sm_freq_tr]
    features_te = [cell_te, tfidf_test, cont_desc_te, sm_freq_te]

    if rdkit_fp_tr.shape[1] > 0:
        features_tr.append(rdkit_fp_tr)
        features_te.append(rdkit_fp_te)

    X_train = np.hstack(features_tr).astype(np.float32)
    X_test = np.hstack(features_te).astype(np.float32)

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    return df_train, df_test, X_train, Y_train, X_test, target_cols


# ==============================================================================
# 4. TRAINING & EVALUATION PIPELINE
# ==============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using computational device: {device}")

    # Prepare datasets
    df_train, df_test, X_train, Y_train, X_test, target_genes = prepare_datasets()

    n_samples, in_features = X_train.shape
    n_targets = Y_train.shape[1]
    n_test = X_test.shape[0]

    # Cross-validation setup
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = list(skf.split(X_train, df_train["cell_type"]))

    epochs = 55
    batch_size = 32
    learning_rate = 1e-3
    weight_decay = 1e-4

    oof_predictions = np.zeros_like(Y_train, dtype=np.float32)
    test_predictions = np.zeros((n_test, n_targets), dtype=np.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\nTraining Fold {fold_idx + 1}/{n_splits}...")

        X_tr, y_tr = X_train[train_idx], Y_train[train_idx]
        X_va, y_va = X_train[val_idx], Y_train[val_idx]

        train_dataset = TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_va, dtype=torch.float32),
            torch.tensor(y_va, dtype=torch.float32),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = build_model(
            in_features=in_features,
            out_features=n_targets,
            hidden_dim=1024,
            latent_dim=512,
            num_res_blocks=3,
            dropout_rate=0.25,
            device=str(device),
        )

        criterion, optimizer, scheduler = build_loss_and_optimizer(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            eta_min=1e-6,
        )

        best_val_score = float("inf")
        best_val_preds = None
        best_model_weights = None

        for epoch in range(1, epochs + 1):
            model.train()
            total_train_loss = 0.0
            n_batches = 0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()

                pred_y = model(batch_x)
                loss = criterion(pred_y, batch_y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                total_train_loss += loss.item()
                n_batches += 1

            scheduler.step()

            # Validation evaluation
            model.eval()
            val_preds_list = []
            with torch.no_grad():
                for batch_x, _ in val_loader:
                    batch_x = batch_x.to(device)
                    val_pred = model(batch_x)
                    val_preds_list.append(val_pred.cpu().numpy())

            current_val_preds = np.vstack(val_preds_list)
            val_mrrmse = compute_mrrmse_metric(y_va, current_val_preds)
            avg_train_loss = total_train_loss / max(1, n_batches)

            if val_mrrmse < best_val_score:
                best_val_score = val_mrrmse
                best_val_preds = current_val_preds
                best_model_weights = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }

            if epoch % 10 == 0 or epoch == epochs:
                print(
                    f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val MRRMSE: {val_mrrmse:.4f} (Best: {best_val_score:.4f})"
                )

        oof_predictions[val_idx] = best_val_preds

        # Test inference using best model weights of this fold
        model.load_state_dict({k: v.to(device) for k, v in best_model_weights.items()})
        model.eval()
        with torch.no_grad():
            fold_test_pred = model(X_test_tensor).cpu().numpy()
            test_predictions += fold_test_pred / len(folds)

    # Compute overall Out-Of-Fold MRRMSE
    overall_oof_mrrmse = compute_mrrmse_metric(Y_train, oof_predictions)
    print(f"\nOverall Out-of-Fold MRRMSE: {overall_oof_mrrmse:.5f}")

    # Generate submission file matching the exact competition format
    if "id" in df_test.columns:
        test_ids = df_test["id"].values
    else:
        test_ids = np.arange(n_test)

    submission_df = pd.DataFrame(test_predictions, columns=target_genes)
    submission_df.insert(0, "id", test_ids)

    sub_path = "./submission/submission.csv"
    submission_df.to_csv(sub_path, index=False)
    print(
        f"Submission successfully saved to {sub_path} with shape {submission_df.shape}"
    )

    # Final validation score output required for evaluation
    print(f"Final Validation Score: {overall_oof_mrrmse}")


if __name__ == "__main__":
    main()