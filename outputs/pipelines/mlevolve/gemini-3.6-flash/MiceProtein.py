import os
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.decomposition import PCA, FastICA, TruncatedSVD

warnings.filterwarnings("ignore")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FeatureGating(nn.Module):
    """Dynamic contextual gating mechanism to model signaling pathway activation/suppression."""

    def __init__(self, in_features: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(in_features, in_features), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class DenseResBlock(nn.Module):
    """Residual MLP block with Batch Normalization, GELU activations, and Dropout."""

    def __init__(self, dim: int, dropout_rate: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class ProteinGatedResNet(nn.Module):
    """Gated Residual Deep Neural Network for multi-class protein expression classification."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 8,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.gating = FeatureGating(input_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList(
            [DenseResBlock(hidden_dim, dropout_rate=dropout) for _ in range(num_blocks)]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_bn(x)
        x = self.gating(x)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        out = self.head(x)
        return out


def fit_and_transform_fold(
    df_tr_raw, df_va_raw, df_te_raw, protein_features, top_pairs
):
    X_tr = df_tr_raw[protein_features].copy()
    X_va = df_va_raw[protein_features].copy()
    X_te = df_te_raw[protein_features].copy()

    # Missing indicators
    for col in protein_features:
        if (
            X_tr[col].isnull().sum() > 0
            or X_va[col].isnull().sum() > 0
            or X_te[col].isnull().sum() > 0
        ):
            X_tr[f"{col}_isna"] = X_tr[col].isnull().astype(float)
            X_va[f"{col}_isna"] = X_va[col].isnull().astype(float)
            X_te[f"{col}_isna"] = X_te[col].isnull().astype(float)

    imputer = SimpleImputer(strategy="median")
    X_tr_imp = pd.DataFrame(
        imputer.fit_transform(X_tr[protein_features]),
        columns=protein_features,
        index=X_tr.index,
    )
    X_va_imp = pd.DataFrame(
        imputer.transform(X_va[protein_features]),
        columns=protein_features,
        index=X_va.index,
    )
    X_te_imp = pd.DataFrame(
        imputer.transform(X_te[protein_features]),
        columns=protein_features,
        index=X_te.index,
    )

    # Phosphorylation Ratios & Differences
    phospho_pairs = [
        (col, col[1:])
        for col in protein_features
        if col.startswith("p") and len(col) > 1 and col[1:] in protein_features
    ]

    for df in [X_tr_imp, X_va_imp, X_te_imp]:
        for p_col, base_col in phospho_pairs:
            df[f"ratio_{p_col}_{base_col}"] = df[p_col] / (df[base_col] + 1e-5)
            df[f"diff_{p_col}_{base_col}"] = df[p_col] - df[base_col]

        # Group summary features
        p_cols = [c for c in protein_features if c.startswith("p")]
        non_p_cols = [c for c in protein_features if not c.startswith("p")]

        if p_cols:
            df["p_group_mean"] = df[p_cols].mean(axis=1)
            df["p_group_std"] = df[p_cols].std(axis=1)
        if non_p_cols:
            df["non_p_group_mean"] = df[non_p_cols].mean(axis=1)
            df["non_p_group_std"] = df[non_p_cols].std(axis=1)
        if p_cols and non_p_cols:
            df["p_non_p_ratio"] = df["p_group_mean"] / (df["non_p_group_mean"] + 1e-5)

        # Row-level protein summary
        df["protein_mean"] = df[protein_features].mean(axis=1)
        df["protein_std"] = df[protein_features].std(axis=1)
        df["protein_median"] = df[protein_features].median(axis=1)
        df["protein_min"] = df[protein_features].min(axis=1)
        df["protein_max"] = df[protein_features].max(axis=1)
        df["protein_iqr"] = df[protein_features].quantile(0.75, axis=1) - df[
            protein_features
        ].quantile(0.25, axis=1)
        df["protein_skew"] = df[protein_features].skew(axis=1)
        df["protein_l2_norm"] = np.sqrt((df[protein_features] ** 2).sum(axis=1))

        # Correlated pair interactions
        for col1, col2 in top_pairs:
            df[f"prod_{col1}_{col2}"] = df[col1] * df[col2]
            df[f"ratio_{col1}_{col2}"] = df[col1] / (df[col2] + 1e-5)

    # Re-attach missing indicators
    extra_cols = [c for c in X_tr.columns if c.endswith("_isna")]
    if extra_cols:
        X_tr_imp = pd.concat([X_tr_imp, X_tr[extra_cols]], axis=1)
        X_va_imp = pd.concat([X_va_imp, X_va[extra_cols]], axis=1)
        X_te_imp = pd.concat([X_te_imp, X_te[extra_cols]], axis=1)

    scaler = RobustScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_imp)
    X_va_scaled = scaler.transform(X_va_imp)
    X_te_scaled = scaler.transform(X_te_imp)

    pca = PCA(n_components=8, random_state=42)
    X_tr_pca = pca.fit_transform(X_tr_scaled)
    X_va_pca = pca.transform(X_va_scaled)
    X_te_pca = pca.transform(X_te_scaled)

    ica = FastICA(n_components=6, random_state=42, max_iter=1000)
    X_tr_ica = ica.fit_transform(X_tr_scaled)
    X_va_ica = ica.transform(X_va_scaled)
    X_te_ica = ica.transform(X_te_scaled)

    svd = TruncatedSVD(n_components=6, random_state=42)
    X_tr_svd = svd.fit_transform(X_tr_scaled)
    X_va_svd = svd.transform(X_va_scaled)
    X_te_svd = svd.transform(X_te_scaled)

    X_tr_final = np.hstack([X_tr_scaled, X_tr_pca, X_tr_ica, X_tr_svd])
    X_va_final = np.hstack([X_va_scaled, X_va_pca, X_va_ica, X_va_svd])
    X_te_final = np.hstack([X_te_scaled, X_te_pca, X_te_ica, X_te_svd])

    return X_tr_final, X_va_final, X_te_final


def main():
    set_seed(42)
    os.makedirs("./submission", exist_ok=True)
    os.makedirs("./working", exist_ok=True)

    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    target_col = "class"
    ignore_cols = {
        target_col,
        "MouseID",
        "Mouse ID",
        "id",
        "Genotype",
        "Treatment",
        "Behavior",
    }

    protein_cols = [
        c for c in df_train.columns if c in df_test.columns and c not in ignore_cols
    ]
    protein_cols = [
        c for c in protein_cols if pd.api.types.is_numeric_dtype(df_train[c])
    ]

    # Calculate correlated feature pairs for interaction features
    temp_imp = SimpleImputer(strategy="median").fit_transform(df_train[protein_cols])
    temp_df = pd.DataFrame(temp_imp, columns=protein_cols)
    corr_matrix = temp_df.corr().abs()
    np.fill_diagonal(corr_matrix.values, 0)
    top_pairs = []
    for _ in range(12):
        max_idx = corr_matrix.values.argmax()
        i, j = np.unravel_index(max_idx, corr_matrix.shape)
        c1, c2 = corr_matrix.columns[i], corr_matrix.columns[j]
        top_pairs.append((c1, c2))
        corr_matrix.loc[c1, c2] = 0
        corr_matrix.loc[c2, c1] = 0

    groups = None
    if "MouseID" in df_train.columns:
        groups = df_train["MouseID"].astype(str).apply(lambda x: x.split("_")[0]).values
    elif "Mouse ID" in df_train.columns:
        groups = (
            df_train["Mouse ID"].astype(str).apply(lambda x: x.split("_")[0]).values
        )

    le = LabelEncoder()
    y_train = le.fit_transform(df_train[target_col])
    num_classes = len(le.classes_)

    n_splits = 5
    if groups is not None and len(np.unique(groups)) >= n_splits:
        cv = StratifiedGroupKFold(n_splits=n_splits)
        splits = list(cv.split(df_train, y_train, groups=groups))
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(cv.split(df_train, y_train))

    oof_probs = np.zeros((len(df_train), num_classes))
    test_probs = np.zeros((len(df_test), num_classes))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    epochs = 120

    for fold, (train_idx, val_idx) in enumerate(splits):
        df_tr_fold = df_train.iloc[train_idx]
        df_va_fold = df_train.iloc[val_idx]

        X_tr, X_va, X_te = fit_and_transform_fold(
            df_tr_fold, df_va_fold, df_test, protein_cols, top_pairs
        )
        y_tr, y_va = y_train[train_idx], y_train[val_idx]

        train_ds = TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.long),
        )
        val_ds = TensorDataset(
            torch.tensor(X_va, dtype=torch.float32),
            torch.tensor(y_va, dtype=torch.long),
        )
        test_ds = TensorDataset(torch.tensor(X_te, dtype=torch.float32))

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        input_dim = X_tr.shape[1]
        model = ProteinGatedResNet(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=256,
            num_blocks=3,
            dropout=0.25,
        ).to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        best_val_loss = float("inf")
        best_model_path = f"./working/model_fold_{fold}.pt"

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(by)
            scheduler.step()

            train_loss /= len(train_ds)

            model.eval()
            val_loss = 0.0
            correct = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    logits = model(bx)
                    loss = criterion(logits, by)
                    val_loss += loss.item() * len(by)
                    preds = torch.argmax(logits, dim=-1)
                    correct += (preds == by).sum().item()

            val_loss /= len(val_ds)
            val_acc = correct / len(val_ds)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)

            print(
                f"Fold {fold+1}/{n_splits} | Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

        # Load best fold checkpoint for validation and test inference
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        fold_val_logits = []
        with torch.no_grad():
            for bx, _ in val_loader:
                bx = bx.to(device)
                logits = model(bx)
                fold_val_logits.append(logits.cpu())
        val_probs = F.softmax(torch.cat(fold_val_logits, dim=0), dim=-1).numpy()
        oof_probs[val_idx] = val_probs

        fold_test_logits = []
        with torch.no_grad():
            for (bx,) in test_loader:
                bx = bx.to(device)
                logits = model(bx)
                fold_test_logits.append(logits.cpu())
        fold_test_probs = F.softmax(torch.cat(fold_test_logits, dim=0), dim=-1).numpy()
        test_probs += fold_test_probs / n_splits

    oof_preds = np.argmax(oof_probs, axis=1)
    val_score = accuracy_score(y_train, oof_preds)

    test_preds_idx = np.argmax(test_probs, axis=1)
    test_preds_labels = le.inverse_transform(test_preds_idx)

    sub_id = df_test["id"] if "id" in df_test.columns else np.arange(len(df_test))
    sub_df = pd.DataFrame({"id": sub_id, "class": test_preds_labels})
    sub_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {val_score:.6f}")


if __name__ == "__main__":
    main()
