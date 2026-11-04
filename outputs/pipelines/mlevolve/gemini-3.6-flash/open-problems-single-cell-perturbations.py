import os
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


# Set random seeds for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define directory paths
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# 1. Load Data
train_df = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
test_df = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")
if os.path.exists(sample_sub_path):
    sample_sub = pd.read_csv(sample_sub_path)
    gene_cols = [c for c in sample_sub.columns if c != "id"]
else:
    meta_cols = {"cell_type", "sm_name", "sm_lincs_id", "SMILES", "control", "id"}
    gene_cols = [c for c in train_df.columns if c not in meta_cols]

train_y = train_df[gene_cols].values.astype(np.float32)


# 2. Cheminformatics Feature Extraction from SMILES
def extract_rdkit_features(smiles_list):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors, MACCSkeys

        has_rdkit = True
    except ImportError:
        has_rdkit = False

    fps = []
    maccs_list = []
    phys_list = []
    string_feats = []

    for sm in smiles_list:
        str_f = [
            len(sm),
            sm.count("C"),
            sm.count("N"),
            sm.count("O"),
            sm.count("S"),
            sm.count("F"),
            sm.count("Cl"),
            sm.count("Br"),
            sm.count("P"),
            sm.count("="),
            sm.count("#"),
            sm.count("("),
            sm.count(")"),
            sm.count("["),
            sm.count("]"),
        ]
        string_feats.append(str_f)

        if has_rdkit:
            mol = Chem.MolFromSmiles(sm) if isinstance(sm, str) else None
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fp_arr = np.array(fp, dtype=np.float32)

                maccs = MACCSkeys.GenMACCSKeys(mol)
                maccs_arr = np.array(maccs, dtype=np.float32)

                phys = [
                    Descriptors.ExactMolWt(mol),
                    Descriptors.MolLogP(mol),
                    Descriptors.TPSA(mol),
                    Descriptors.NumHDonors(mol),
                    Descriptors.NumHAcceptors(mol),
                    Descriptors.NumRotatableBonds(mol),
                    Descriptors.NumAromaticRings(mol),
                    Descriptors.FractionCSP3(mol),
                    Descriptors.HeavyAtomCount(mol),
                ]
            else:
                fp_arr = np.zeros(1024, dtype=np.float32)
                maccs_arr = np.zeros(167, dtype=np.float32)
                phys = [0.0] * 9
            fps.append(fp_arr)
            maccs_list.append(maccs_arr)
            phys_list.append(phys)

    string_feats = np.array(string_feats, dtype=np.float32)
    if has_rdkit and len(fps) > 0:
        fps = np.array(fps, dtype=np.float32)
        maccs_list = np.array(maccs_list, dtype=np.float32)
        phys_list = np.array(phys_list, dtype=np.float32)
        return np.hstack([fps, maccs_list, phys_list, string_feats])
    else:
        return string_feats


train_chem = extract_rdkit_features(train_df["SMILES"].tolist())
test_chem = extract_rdkit_features(test_df["SMILES"].tolist())

# 3. Categorical Encodings & Leak-Free Feature Scaling
cell_ohe_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
sm_ohe_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

train_cell_ohe = cell_ohe_encoder.fit_transform(train_df[["cell_type"]])
test_cell_ohe = cell_ohe_encoder.transform(test_df[["cell_type"]])

train_sm_ohe = sm_ohe_encoder.fit_transform(train_df[["sm_name"]])
test_sm_ohe = sm_ohe_encoder.transform(test_df[["sm_name"]])

train_control = train_df["control"].values.astype(np.float32).reshape(-1, 1)
test_control = test_df["control"].values.astype(np.float32).reshape(-1, 1)


def build_features(cell_ohe, sm_ohe, control, chem, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        chem_scaled = scaler.fit_transform(chem)
    else:
        chem_scaled = scaler.transform(chem)
    interactions = []
    for i in range(cell_ohe.shape[1]):
        cell_col = cell_ohe[:, i : i + 1]
        interaction = cell_col * chem_scaled
        interactions.append(interaction)
    interactions_arr = np.hstack(interactions)
    feat_matrix = np.hstack([cell_ohe, sm_ohe, control, chem_scaled, interactions_arr]).astype(np.float32)
    return feat_matrix, scaler


train_x, global_scaler = build_features(
    train_cell_ohe, train_sm_ohe, train_control, train_chem, scaler=None
)
test_x, _ = build_features(
    test_cell_ohe, test_sm_ohe, test_control, test_chem, scaler=global_scaler
)

# 4. K-Fold Split Assignment
kf = KFold(n_splits=5, shuffle=True, random_state=42)
folds = np.zeros(len(train_df), dtype=np.int32)
for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    folds[val_idx] = fold


# 5. Model Architecture & Loss Definition
class ResidualBlock(nn.Module):

    def __init__(self, dim: int, dropout: float = 0.25):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.drop(self.act(self.norm1(self.fc1(x))))
        out = self.norm2(self.fc2(out))
        return self.act(out + residual)


class MultiTaskLowRankGeneNet(nn.Module):

    def __init__(
        self,
        in_features: int,
        num_targets: int = 18211,
        hidden_dim: int = 512,
        latent_dim: int = 128,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res1 = ResidualBlock(hidden_dim, dropout=dropout)
        self.res2 = ResidualBlock(hidden_dim, dropout=dropout)
        self.latent_layer = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2.0),
        )
        self.decoder = nn.Linear(latent_dim, num_targets, bias=True)
        self.skip_proj = nn.Linear(hidden_dim, num_targets, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.res1(h)
        h = self.res2(h)
        z = self.latent_layer(h)
        out = self.decoder(z) + self.skip_proj(h)
        return out


class MRRMSELoss(nn.Module):

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        row_mse = torch.mean((y_pred - y_true) ** 2, dim=1)
        row_rmse = torch.sqrt(row_mse + self.eps)
        return torch.mean(row_rmse)


def compute_mrrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    row_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))
    return float(np.mean(row_rmse))


# 6. Multi-Seed Training, Cross-Validation, & Ensemble Loop
num_samples, num_features = train_x.shape
num_targets = train_y.shape[1]
num_folds = len(np.unique(folds))

oof_ridge_preds = np.zeros_like(train_y, dtype=np.float32)
oof_nn_preds = np.zeros_like(train_y, dtype=np.float32)

test_ridge_preds = np.zeros((test_x.shape[0], num_targets), dtype=np.float32)
test_nn_preds = np.zeros((test_x.shape[0], num_targets), dtype=np.float32)

num_epochs = 150
batch_size = 32
nn_seeds = [42, 2024]

for fold in range(num_folds):
    val_mask = folds == fold
    train_mask = ~val_mask

    X_tr, fold_scaler = build_features(
        train_cell_ohe[train_mask],
        train_sm_ohe[train_mask],
        train_control[train_mask],
        train_chem[train_mask],
        scaler=None,
    )
    X_va, _ = build_features(
        train_cell_ohe[val_mask],
        train_sm_ohe[val_mask],
        train_control[val_mask],
        train_chem[val_mask],
        scaler=fold_scaler,
    )
    test_x_fold, _ = build_features(
        test_cell_ohe,
        test_sm_ohe,
        test_control,
        test_chem,
        scaler=fold_scaler,
    )
    y_tr = train_y[train_mask]
    y_va = train_y[val_mask]

    # Fit Ridge Regression
    ridge_model = Ridge(alpha=100.0, random_state=42)
    ridge_model.fit(X_tr, y_tr)
    fold_ridge_oof = ridge_model.predict(X_va).astype(np.float32)
    fold_ridge_test = ridge_model.predict(test_x_fold).astype(np.float32)

    oof_ridge_preds[val_mask] = fold_ridge_oof
    test_ridge_preds += fold_ridge_test / num_folds

    # Multi-Seed Fit for Neural Network (MultiTaskLowRankGeneNet)
    tr_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    va_ds = TensorDataset(
        torch.tensor(X_va, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32),
    )

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    test_ds = TensorDataset(torch.tensor(test_x_fold, dtype=torch.float32))
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )

    fold_seed_nn_oof = np.zeros_like(y_va, dtype=np.float32)
    fold_seed_nn_test = np.zeros((test_x.shape[0], num_targets), dtype=np.float32)

    for seed in nn_seeds:
        seed_everything(seed)
        model = MultiTaskLowRankGeneNet(
            in_features=num_features, num_targets=num_targets
        ).to(device)
        criterion = MRRMSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-5
        )

        best_val_loss = float("inf")
        best_model_path = os.path.join(WORKING_DIR, f"best_model_fold_{fold}_seed_{seed}.pt")
        patience = 30
        patience_counter = 0

        for epoch in range(num_epochs):
            model.train()
            for bx, by in tr_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()

            scheduler.step()

            model.eval()
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for bx, by in va_loader:
                    bx, by = bx.to(device), by.to(device)
                    pred = model(bx)
                    loss = criterion(pred, by)
                    val_loss += loss.item() * len(bx)
                    val_count += len(bx)
            val_loss /= val_count

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Load best checkpoint for OOF and Test Inference
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        va_preds_list = []
        with torch.no_grad():
            for bx, _ in va_loader:
                bx = bx.to(device)
                pred = model(bx)
                va_preds_list.append(pred.cpu().numpy())
        fold_seed_nn_oof += np.vstack(va_preds_list) / len(nn_seeds)

        test_preds_list = []
        with torch.no_grad():
            for (bx,) in test_loader:
                bx = bx.to(device)
                pred = model(bx)
                test_preds_list.append(pred.cpu().numpy())
        fold_seed_nn_test += np.vstack(test_preds_list) / len(nn_seeds)

    oof_nn_preds[val_mask] = fold_seed_nn_oof
    test_nn_preds += fold_seed_nn_test / num_folds

# 7. Optimize OOF Blending Weight & Compute Scores
ridge_score = compute_mrrmse(train_y, oof_ridge_preds)
nn_score = compute_mrrmse(train_y, oof_nn_preds)

best_alpha = 0.0
best_blend_score = float("inf")

for alpha in np.linspace(0.0, 1.0, 101):
    blend_oof = alpha * oof_ridge_preds + (1.0 - alpha) * oof_nn_preds
    score = compute_mrrmse(train_y, blend_oof)
    if score < best_blend_score:
        best_blend_score = score
        best_alpha = alpha

oof_score = best_blend_score

print(f"Ridge OOF MRRMSE Score: {ridge_score:.5f}")
print(f"Neural Net OOF MRRMSE Score: {nn_score:.5f}")
print(f"Optimal Ridge Weight (Alpha): {best_alpha:.2f}")
print(f"Final Validation Score: {oof_score}")

# 8. Save Final Submission File
test_blend_preds = best_alpha * test_ridge_preds + (1.0 - best_alpha) * test_nn_preds
sub_df = pd.DataFrame(test_blend_preds, columns=gene_cols)
sub_df.insert(0, "id", np.arange(len(sub_df)))
sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
sub_df.to_csv(sub_path, index=False)