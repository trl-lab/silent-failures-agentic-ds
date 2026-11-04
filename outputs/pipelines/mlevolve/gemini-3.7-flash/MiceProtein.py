import copy
import os
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, RobustScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set random seed for exact reproducibility across libraries
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "class"
ignore_cols = {"MouseID", "id", "Genotype", "Treatment", "Behavior", target_col}

# Identify numeric protein features
protein_cols = [
    c
    for c in train_df.columns
    if c not in ignore_cols and pd.api.types.is_numeric_dtype(train_df[c])
]

X_raw = train_df[protein_cols].copy()
y_raw = train_df[target_col].copy()
X_test_raw = test_df[protein_cols].copy()

# Encode target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)

# 2. Strict Leak-Free Validation Split
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, val_idx = next(splitter.split(X_raw, y_encoded))

X_tr_raw = X_raw.iloc[train_idx].copy().reset_index(drop=True)
y_train = y_encoded[train_idx]
X_va_raw = X_raw.iloc[val_idx].copy().reset_index(drop=True)
y_val = y_encoded[val_idx]
X_te_raw = X_test_raw.copy().reset_index(drop=True)


# 3. Domain-Specific Feature Engineering Function
def extract_protein_features(df, base_cols, p_pairs):
    """Computes biological phosphorylation signaling ratios, relative activation differences,

    cortical pathway statistics, and missing value indicators.
    """
    feat_df = df.copy()

    # Missingness indicators
    missing_indicators = df[base_cols].isnull().astype(np.float32)
    missing_indicators.columns = [f"{c}_isna" for c in base_cols]

    # Phosphorylation activation ratios and differences
    ratio_features = {}
    for p_col, base_col in p_pairs:
        if p_col in df.columns and base_col in df.columns:
            ratio_features[f"ratio_{p_col}_{base_col}"] = (df[p_col] + 1e-6) / (
                df[base_col] + 1e-6
            )
            ratio_features[f"diff_{p_col}_{base_col}"] = df[p_col] - df[base_col]

    ratio_df = pd.DataFrame(ratio_features, index=df.index)

    # Sample-level aggregate profile statistics across all proteins
    stats_df = pd.DataFrame(
        {
            "protein_mean": df[base_cols].mean(axis=1),
            "protein_std": df[base_cols].std(axis=1),
            "protein_median": df[base_cols].median(axis=1),
            "protein_min": df[base_cols].min(axis=1),
            "protein_max": df[base_cols].max(axis=1),
            "protein_l2_norm": np.sqrt(np.square(df[base_cols].fillna(0)).sum(axis=1)),
            "protein_skew": df[base_cols].skew(axis=1),
            "protein_iqr": df[base_cols].quantile(0.75, axis=1)
            - df[base_cols].quantile(0.25, axis=1),
        },
        index=df.index,
    )

    combined_df = pd.concat([feat_df, ratio_df, stats_df, missing_indicators], axis=1)
    return combined_df


# Detect phosphorylated and base protein pairs dynamically
phospho_pairs = []
for col in protein_cols:
    if col.startswith("p") or col.startswith("P"):
        base_candidate = col[1:]
        if base_candidate in protein_cols:
            phospho_pairs.append((col, base_candidate))
        elif base_candidate.lower() in [c.lower() for c in protein_cols]:
            match = [c for c in protein_cols if c.lower() == base_candidate.lower()][0]
            phospho_pairs.append((col, match))

# Extract features
X_tr_engineered = extract_protein_features(X_tr_raw, protein_cols, phospho_pairs)
X_va_engineered = extract_protein_features(X_va_raw, protein_cols, phospho_pairs)
X_te_engineered = extract_protein_features(X_te_raw, protein_cols, phospho_pairs)

engineered_cols = list(X_tr_engineered.columns)

# 4. Imputation and Scaling (Fitted STRICTLY on Train)
imputer = SimpleImputer(strategy="median")
X_tr_imp = imputer.fit_transform(X_tr_engineered)
X_va_imp = imputer.transform(X_va_engineered)
X_te_imp = imputer.transform(X_te_engineered)

scaler = RobustScaler()
X_tr_scaled = scaler.fit_transform(X_tr_imp)
X_va_scaled = scaler.transform(X_va_imp)
X_te_scaled = scaler.transform(X_te_imp)

# 5. Unsupervised Manifold Features (PCA & Factor Analysis fitted strictly on Train)
n_components_pca = min(12, X_tr_scaled.shape[1])
pca = PCA(n_components=n_components_pca, random_state=RANDOM_STATE)
pca_tr = pca.fit_transform(X_tr_scaled)
pca_va = pca.transform(X_va_scaled)
pca_te = pca.transform(X_te_scaled)

n_components_fa = min(8, X_tr_scaled.shape[1])
fa = FactorAnalysis(n_components=n_components_fa, random_state=RANDOM_STATE)
fa_tr = fa.fit_transform(X_tr_scaled)
fa_va = fa.transform(X_va_scaled)
fa_te = fa.transform(X_te_scaled)

# Assemble final processed feature matrices
X_train_final = np.hstack([X_tr_scaled, pca_tr, fa_tr])
X_val_final = np.hstack([X_va_scaled, pca_va, fa_va])
X_test_final = np.hstack([X_te_scaled, pca_te, fa_te])


# 6. Deep Residual Tabular Model Definition
class FeatureGating(nn.Module):

    def __init__(self, input_dim: int, reduction: int = 4):
        super().__init__()
        reduced_dim = max(8, input_dim // reduction)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, reduced_dim),
            nn.GELU(),
            nn.Linear(reduced_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class ResidualDenseBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.25):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(hidden_dim * 2)
        self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm1(x)
        out = self.linear1(out)
        out = self.act1(out)
        out = self.drop1(out)
        out = self.norm2(out)
        out = self.linear2(out)
        out = self.drop2(out)
        return residual + out


class ProteinResNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 8,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.feature_gate = FeatureGating(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.res_blocks = nn.ModuleList(
            [
                ResidualDenseBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = self.feature_gate(x)
        h = self.input_proj(gated)
        for block in self.res_blocks:
            h = block(h)
        h = self.head_norm(h)
        return self.classifier(h)


# 7. Model Training Setup
X_tr_tensor = torch.tensor(X_train_final, dtype=torch.float32)
y_tr_tensor = torch.tensor(y_train, dtype=torch.long)
X_va_tensor = torch.tensor(X_val_final, dtype=torch.float32)
y_va_tensor = torch.tensor(y_val, dtype=torch.long)
X_te_tensor = torch.tensor(X_test_final, dtype=torch.float32)

batch_size = 32
train_loader = DataLoader(
    TensorDataset(X_tr_tensor, y_tr_tensor),
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
)
val_loader = DataLoader(
    TensorDataset(X_va_tensor, y_va_tensor),
    batch_size=batch_size * 2,
    shuffle=False,
    drop_last=False,
)
test_loader = DataLoader(
    TensorDataset(X_te_tensor),
    batch_size=batch_size * 2,
    shuffle=False,
    drop_last=False,
)

input_dim = X_train_final.shape[1]
num_classes = len(label_encoder.classes_)
max_epochs = 100

model = ProteinResNet(
    input_dim=input_dim,
    num_classes=num_classes,
    hidden_dim=256,
    num_blocks=3,
    dropout_rate=0.25,
).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.AdamW(
    model.parameters(), lr=1e-3, weight_decay=1e-3, betas=(0.9, 0.999), eps=1e-8
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max_epochs, eta_min=1e-6
)

# 8. Training Loop with Validation Accuracy Tracking and Early Stopping
best_val_acc = 0.0
best_model_state = None
patience = 25
patience_counter = 0

for epoch in range(max_epochs):
    model.train()
    total_train_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_train_loss += loss.item() * batch_x.size(0)

    train_loss = total_train_loss / len(train_loader.dataset)
    scheduler.step()

    # Validation Phase
    model.eval()
    total_val_loss = 0.0
    all_val_preds = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_val_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_val_preds.extend(preds)

    val_loss = total_val_loss / len(val_loader.dataset)
    val_acc = accuracy_score(y_val, np.array(all_val_preds))

    print(
        f"Epoch {epoch+1:03d}/{max_epochs:03d} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

# 9. Evaluation and Submission Generation
if best_model_state is not None:
    model.load_state_dict(best_model_state)

model.eval()
final_val_preds = []
with torch.no_grad():
    for batch_x, _ in val_loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        preds = logits.argmax(dim=1).cpu().numpy()
        final_val_preds.extend(preds)

final_val_accuracy = accuracy_score(y_val, np.array(final_val_preds))

test_preds = []
with torch.no_grad():
    for (batch_x,) in test_loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        preds = logits.argmax(dim=1).cpu().numpy()
        test_preds.extend(preds)

test_pred_labels = label_encoder.inverse_transform(test_preds)

submission_df = pd.DataFrame(
    {"id": np.arange(len(test_pred_labels)), "class": test_pred_labels}
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_val_accuracy}")
