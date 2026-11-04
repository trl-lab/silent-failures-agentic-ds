import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss
import lightgbm as lgb
import xgboost as xgb
from scipy.optimize import minimize


# Set seed for reproducibility
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

# Ensure output directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

y = train_df["Activity"].values
X_raw = train_df.drop(columns=["Activity"]).copy()

if "Activity" in test_df.columns:
    X_test_raw = test_df.drop(columns=["Activity"]).copy()
else:
    X_test_raw = test_df.copy()


# 2. Extract Row-wise Molecular Descriptor Statistics
def extract_row_features(df):
    val_matrix = df.values.astype(np.float32)

    is_binary = np.all((val_matrix == 0) | (val_matrix == 1), axis=0)
    bin_indices = np.where(is_binary)[0]
    cont_indices = np.where(~is_binary)[0]

    feats = {}
    if len(bin_indices) > 0:
        bin_vals = val_matrix[:, bin_indices]
        feats["bin_sum"] = bin_vals.sum(axis=1)
        feats["bin_mean"] = bin_vals.mean(axis=1)
        feats["bin_std"] = bin_vals.std(axis=1)

    if len(cont_indices) > 0:
        cont_vals = val_matrix[:, cont_indices]
        feats["cont_mean"] = cont_vals.mean(axis=1)
        feats["cont_std"] = cont_vals.std(axis=1)
        feats["cont_max"] = cont_vals.max(axis=1)
        feats["cont_min"] = cont_vals.min(axis=1)
        feats["cont_median"] = np.median(cont_vals, axis=1)
        feats["cont_q25"] = np.percentile(cont_vals, 25, axis=1)
        feats["cont_q75"] = np.percentile(cont_vals, 75, axis=1)
        feats["cont_iqr"] = feats["cont_q75"] - feats["cont_q25"]
        feats["cont_nonzero_cnt"] = (cont_vals != 0).sum(axis=1)
        feats["cont_nonzero_ratio"] = feats["cont_nonzero_cnt"] / float(
            len(cont_indices)
        )

    feats["global_sum"] = val_matrix.sum(axis=1)
    feats["global_mean"] = val_matrix.mean(axis=1)
    feats["global_std"] = val_matrix.std(axis=1)
    feats["global_nonzero_cnt"] = (val_matrix != 0).sum(axis=1)

    return pd.DataFrame(feats, index=df.index)


X_train_row = extract_row_features(X_raw)
X_test_row = extract_row_features(X_test_raw)

X_train_full = pd.concat([X_raw, X_train_row], axis=1)
X_test_full = pd.concat([X_test_raw, X_test_row], axis=1)

# 3. Stratified K-Fold Data Preparation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y)):
    X_tr = X_train_full.iloc[train_idx].copy()
    y_tr = y[train_idx]
    X_va = X_train_full.iloc[val_idx].copy()
    y_va = y[val_idx]
    X_te = X_test_full.copy()

    vt = VarianceThreshold(threshold=1e-4)
    X_tr_vt = vt.fit_transform(X_tr)
    X_va_vt = vt.transform(X_va)
    X_te_vt = vt.transform(X_te)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_vt)
    X_va_scaled = scaler.transform(X_va_vt)
    X_te_scaled = scaler.transform(X_te_vt)

    qt = QuantileTransformer(
        n_quantiles=100, output_distribution="normal", random_state=42
    )
    X_tr_qt = qt.fit_transform(X_tr_vt)
    X_va_qt = qt.transform(X_va_vt)
    X_te_qt = qt.transform(X_te_vt)

    pca = PCA(n_components=16, random_state=42)
    pca_tr = pca.fit_transform(X_tr_scaled)
    pca_va = pca.transform(X_va_scaled)
    pca_te = pca.transform(X_te_scaled)

    svd = TruncatedSVD(n_components=16, random_state=42)
    svd_tr = svd.fit_transform(X_tr_scaled)
    svd_va = svd.transform(X_va_scaled)
    svd_te = svd.transform(X_te_scaled)

    X_tr_final = np.hstack([X_tr_scaled, pca_tr, svd_tr])
    X_va_final = np.hstack([X_va_scaled, pca_va, svd_va])
    X_te_final = np.hstack([X_te_scaled, pca_te, svd_te])

    np.savez_compressed(
        f"./working/fold_{fold}.npz",
        X_tr_scaled=X_tr_final,
        X_va_scaled=X_va_final,
        X_te_scaled=X_te_final,
        y_tr=y_tr,
        y_va=y_va,
        val_idx=val_idx,
    )


# 4. Neural Architecture Modules
class GatedLinearUnit(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.2):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features * 2)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        h = self.fc(x)
        val, gate = h.chunk(2, dim=-1)
        out = val * torch.sigmoid(gate)
        return self.dropout(self.norm(out))


class FeatureAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout_rate=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout_rate, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.glu = GatedLinearUnit(embed_dim, embed_dim, dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x_seq = x.unsqueeze(1)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        x_attn = self.norm1(x + attn_out.squeeze(1))
        out = self.norm2(x_attn + self.glu(x_attn))
        return out


class MolecularGLUTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_blocks=3, dropout_rate=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.blocks = nn.ModuleList(
            [
                FeatureAttentionBlock(
                    hidden_dim, num_heads=4, dropout_rate=dropout_rate
                )
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        logits = self.head(h).squeeze(-1)
        return logits


class LabelSmoothBCEWithLogitsLoss(nn.Module):
    def __init__(self, smoothing=0.005):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        targets_smooth = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, targets_smooth)


def create_model_and_optimizer(
    input_dim,
    hidden_dim=256,
    num_blocks=3,
    dropout_rate=0.3,
    lr=1e-3,
    weight_decay=1e-2,
    smoothing=0.005,
    device="cpu",
):
    model = MolecularGLUTransformer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout_rate=dropout_rate,
    ).to(device)

    criterion = LabelSmoothBCEWithLogitsLoss(smoothing=smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-6
    )

    return model, criterion, optimizer, scheduler


# 5. Training and Evaluation Loop
n_samples_train = len(train_df)
n_samples_test = len(test_df)

num_models = 4  # 0: LightGBM, 1: XGBoost, 2: ExtraTrees, 3: Neural Network
model_names = ["LightGBM", "XGBoost", "ExtraTrees", "NeuralNet"]

oof_preds_matrix = np.zeros((n_samples_train, num_models), dtype=np.float32)
oof_targets = np.zeros(n_samples_train, dtype=np.float32)
test_preds_matrix = np.zeros((n_samples_test, num_models), dtype=np.float32)

max_epochs = 60
batch_size = 128
patience = 12

for fold in range(n_splits):
    fold_data = np.load(f"./working/fold_{fold}.npz")
    X_tr = fold_data["X_tr_scaled"]
    y_tr = fold_data["y_tr"]
    X_va = fold_data["X_va_scaled"]
    y_va = fold_data["y_va"]
    X_te = fold_data["X_te_scaled"]
    val_idx = fold_data["val_idx"]

    oof_targets[val_idx] = y_va

    # --- Model 0: LightGBM ---
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    lgb_va_pred = lgb_model.predict_proba(X_va)[:, 1]
    lgb_te_pred = lgb_model.predict_proba(X_te)[:, 1]
    oof_preds_matrix[val_idx, 0] = np.clip(lgb_va_pred, 1e-15, 1 - 1e-15)
    test_preds_matrix[:, 0] += np.clip(lgb_te_pred, 1e-15, 1 - 1e-15) / n_splits

    # --- Model 1: XGBoost ---
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        eval_metric="logloss",
        early_stopping_rounds=50,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    xgb_va_pred = xgb_model.predict_proba(X_va)[:, 1]
    xgb_te_pred = xgb_model.predict_proba(X_te)[:, 1]
    oof_preds_matrix[val_idx, 1] = np.clip(xgb_va_pred, 1e-15, 1 - 1e-15)
    test_preds_matrix[:, 1] += np.clip(xgb_te_pred, 1e-15, 1 - 1e-15) / n_splits

    # --- Model 2: ExtraTrees ---
    et_model = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=18,
        min_samples_split=5,
        random_state=42 + fold,
        n_jobs=-1,
    )
    et_model.fit(X_tr, y_tr)
    et_va_pred = et_model.predict_proba(X_va)[:, 1]
    et_te_pred = et_model.predict_proba(X_te)[:, 1]
    oof_preds_matrix[val_idx, 2] = np.clip(et_va_pred, 1e-15, 1 - 1e-15)
    test_preds_matrix[:, 2] += np.clip(et_te_pred, 1e-15, 1 - 1e-15) / n_splits

    # --- Model 3: Neural Network ---
    input_dim = X_tr.shape[1]
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.float32)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)

    train_dataset = TensorDataset(X_tr_t, y_tr_t)
    val_dataset = TensorDataset(X_va_t, y_va_t)
    test_dataset = TensorDataset(X_te_t)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model, criterion, optimizer, scheduler = create_model_and_optimizer(
        input_dim=input_dim,
        hidden_dim=256,
        num_blocks=3,
        dropout_rate=0.3,
        lr=1e-3,
        weight_decay=1e-2,
        smoothing=0.005,
        device=device,
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(by)

        train_loss /= len(train_dataset)
        scheduler.step()

        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                logits = model(bx)
                probs = torch.sigmoid(logits)
                val_preds_list.extend(probs.cpu().numpy())

        val_preds_arr = np.array(val_preds_list)
        val_preds_clipped = np.clip(val_preds_arr, 1e-15, 1 - 1e-15)
        epoch_val_loss = log_loss(y_va, val_preds_clipped)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"./working/model_fold_{fold}.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(torch.load(f"./working/model_fold_{fold}.pt"))
    model.eval()

    fold_val_preds = []
    with torch.no_grad():
        for bx, _ in val_loader:
            bx = bx.to(device)
            logits = model(bx)
            probs = torch.sigmoid(logits)
            fold_val_preds.extend(probs.cpu().numpy())

    oof_preds_matrix[val_idx, 3] = np.clip(np.array(fold_val_preds), 1e-15, 1 - 1e-15)

    fold_test_preds = []
    with torch.no_grad():
        for (bx,) in test_loader:
            bx = bx.to(device)
            logits = model(bx)
            probs = torch.sigmoid(logits)
            fold_test_preds.extend(probs.cpu().numpy())

    test_preds_matrix[:, 3] += np.clip(np.array(fold_test_preds), 1e-15, 1 - 1e-15) / n_splits

    print(f"Fold {fold+1} complete.")

# --- Evaluate Individual Model OOF Loss ---
print("\n--- OOF Log Loss Per Model ---")
for i, name in enumerate(model_names):
    m_loss = log_loss(oof_targets, oof_preds_matrix[:, i])
    print(f"{name} OOF Log Loss: {m_loss:.5f}")

# --- Optimize Ensemble Weights (SLSQP) ---
def optimize_weights(oof_matrix, targets):
    n_m = oof_matrix.shape[1]

    def loss_func(weights):
        w_norm = weights / np.sum(weights)
        blend = np.dot(oof_matrix, w_norm)
        blend = np.clip(blend, 1e-15, 1 - 1e-15)
        return log_loss(targets, blend)

    init_weights = np.ones(n_m) / n_m
    bounds = [(0, 1) for _ in range(n_m)]
    constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}

    res = minimize(
        loss_func,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return res.x / np.sum(res.x)

opt_weights = optimize_weights(oof_preds_matrix, oof_targets)

print("\n--- Optimized Ensemble Weights ---")
for name, w in zip(model_names, opt_weights):
    print(f"{name}: {w:.4f}")

oof_blend = np.dot(oof_preds_matrix, opt_weights)
oof_blend = np.clip(oof_blend, 1e-15, 1 - 1e-15)
final_oof_loss = log_loss(oof_targets, oof_blend)

print(f"\nFinal Validation Score (Optimized Ensemble): {final_oof_loss:.5f}")

final_test_preds = np.dot(test_preds_matrix, opt_weights)
final_test_preds = np.clip(final_test_preds, 1e-15, 1 - 1e-15)

submission_df = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "Activity": final_test_preds}
)
submission_df.to_csv("./submission/submission.csv", index=False)