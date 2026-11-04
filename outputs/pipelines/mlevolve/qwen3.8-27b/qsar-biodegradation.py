import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import os

# ============================================================
# STEP 1: DATA LOADING
# ============================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Target distribution:\n{train_df['Class'].value_counts()}")

target_col = "Class"
feature_cols = [c for c in train_df.columns if c != target_col]
print(f"Number of features: {len(feature_cols)}")

# ============================================================
# STEP 3: TRAIN/VALIDATION SPLIT (Stratified)
# ============================================================
y_train = pd.to_numeric(train_df[target_col], errors='coerce').values
y_train = (y_train > 1).astype(int)
print(f"Target converted to binary: 0={np.sum(y_train==0)}, 1={np.sum(y_train==1)}")
X_train_full = train_df[feature_cols].copy()

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# ============================================================
# STEP 2: DATA CLEANING (fit imputer on train split only)
# ============================================================
print(f"Missing values in train: {X_tr.isnull().sum().sum()}")
print(f"Missing values in test: {test_df[feature_cols].isnull().sum().sum()}")

train_medians = X_tr.median()
X_tr = X_tr.fillna(train_medians)
X_val = X_val.fillna(train_medians)
test_df[feature_cols] = test_df[feature_cols].fillna(train_medians)

print(f"Train split: {X_tr.shape[0]} samples")
print(f"Validation split: {X_val.shape[0]} samples")
print(f"Val class dist: {np.bincount(y_val.astype(int))}")


# ============================================================
# STEP 4: FEATURE ENGINEERING
# ============================================================
def engineer_features(X):
    """Apply domain-informed feature engineering to dataframe X.
    Only creates derived features if the required source columns exist."""
    X = X.copy()
    cols = set(X.columns)

    def has(*names):
        return all(n in cols for n in names)

    # --- Derived ratio features (structural complexity) ---
    if has("nHM", "nCIR"):
        X["ratio_nHM_nCIR"] = X["nHM"] / (X["nCIR"] + 1)
    if has("nO", "nHM"):
        X["ratio_nO_nHM"] = X["nO"] / (X["nHM"] + 1)
    if has("nN", "nHM"):
        X["ratio_nN_nHM"] = X["nN"] / (X["nHM"] + 1)
    if has("nX", "nHM"):
        X["ratio_nX_nHM"] = X["nX"] / (X["nHM"] + 1)
    if has("nHDon", "nHM"):
        X["ratio_nHDon_nHM"] = X["nHDon"] / (X["nHM"] + 1)

    # --- Interaction features: Electronic × Structural ---
    if has("Me", "nX"):
        X["Me_x_nX"] = X["Me"] * X["nX"]
    if has("Me", "nO"):
        X["Me_x_nO"] = X["Me"] * X["nO"]
    if has("Mi", "nN"):
        X["Mi_x_nN"] = X["Mi"] * X["nN"]
    if has("Me", "nCIR"):
        X["Me_x_nCIR"] = X["Me"] * X["nCIR"]

    # --- Interaction features: Topological × Structural ---
    if has("SpMax_L", "nHM"):
        X["SpMax_L_x_nHM"] = X["SpMax_L"] * X["nHM"]
    if has("SM6_L", "nCIR"):
        X["SM6_L_x_nCIR"] = X["SM6_L"] * X["nCIR"]
    if has("HyWi_B(m)", "nCIR"):
        X["HyWi_x_nCIR"] = X["HyWi_B(m)"] * X["nCIR"]

    # --- Interaction features: Electronic × Electronic ---
    if has("Me", "Mi"):
        X["Me_x_Mi"] = X["Me"] * X["Mi"]
    if has("Psi_i_1d", "Me"):
        X["Psi1d_x_Me"] = X["Psi_i_1d"] * X["Me"]
    if has("Psi_i_A", "Me"):
        X["PsiA_x_Me"] = X["Psi_i_A"] * X["Me"]

    # --- Interaction features: Connectivity × Structural ---
    if has("F03[C-N]", "nN"):
        X["F03CN_x_nN"] = X["F03[C-N]"] * X["nN"]
    if has("F03[C-O]", "nO"):
        X["F03CO_x_nO"] = X["F03[C-O]"] * X["nO"]
    if has("F01[N-N]", "nN"):
        X["F01NN_x_nN"] = X["F01[N-N]"] * X["nN"]

    # --- Burden matrix interactions ---
    if has("SM6_B(m)", "nHM"):
        X["SM6Bm_x_nHM"] = X["SM6_B(m)"] * X["nHM"]
    if has("SpMax_B(m)", "nCIR"):
        X["SpMaxBm_x_nCIR"] = X["SpMax_B(m)"] * X["nCIR"]

    # --- E-state interactions ---
    if has("SdO", "nO"):
        X["SdO_x_nO"] = X["SdO"] * X["nO"]
    if has("SdssC", "C%"):
        X["SdssC_x_Cpct"] = X["SdssC"] * X["C%"]

    # --- Composite molecular complexity scores ---
    if has("nO", "nN", "nX"):
        X["n_hetero"] = X["nO"] + X["nN"] + X["nX"]
        if has("nHM"):
            X["hetero_frac"] = X["n_hetero"] / (X["nHM"] + 1)
    if has("nCb-", "nArNO2", "nArCOOR"):
        X["aromatic_sub"] = X["nCb-"] + X["nArNO2"] + X["nArCOOR"]
    if has("nArNO2", "nArCOOR", "nX", "nN-N", "nHDon"):
        X["fg_richness"] = (
            (X["nArNO2"] > 0).astype(int)
            + (X["nArCOOR"] > 0).astype(int)
            + (X["nX"] > 0).astype(int)
            + (X["nN-N"] > 0).astype(int)
            + (X["nHDon"] > 0).astype(int)
        )

    # --- Log transforms for heavy-tailed features ---
    if has("nHM"):
        X["log_nHM"] = np.log1p(X["nHM"])
    if has("SpMax_L"):
        X["log_SpMax_L"] = np.log1p(X["SpMax_L"].clip(lower=0))
    if has("SM6_L"):
        X["log_SM6_L"] = np.log1p(X["SM6_L"].clip(lower=0))
    if has("HyWi_B(m)"):
        X["log_HyWi"] = np.log1p(X["HyWi_B(m)"].clip(lower=0))
    if has("nCIR"):
        X["log_nCIR"] = np.log1p(X["nCIR"])

    # --- Squared features for non-linear effects ---
    if has("Me"):
        X["Me_sq"] = X["Me"] ** 2
    if has("Mi"):
        X["Mi_sq"] = X["Mi"] ** 2
    if has("C%"):
        X["Cpct_sq"] = X["C%"] ** 2

    # --- Difference features ---
    if has("Me", "Mi"):
        X["Me_minus_Mi"] = X["Me"] - X["Mi"]
    if has("Psi_i_1d", "Psi_i_A"):
        X["Psi_diff"] = X["Psi_i_1d"] - X["Psi_i_A"]

    return X


# Apply feature engineering
X_tr_eng = engineer_features(X_tr)
X_val_eng = engineer_features(X_val)
X_test_eng = engineer_features(test_df[feature_cols])

print(f"After feature engineering: {X_tr_eng.shape[1]} features")

# ============================================================
# STEP 5: SCALING (Fit on train only)
# ============================================================
scaler = RobustScaler()
scaler.fit(X_tr_eng)

X_tr_scaled = pd.DataFrame(
    scaler.transform(X_tr_eng), columns=X_tr_eng.columns, index=X_tr_eng.index
)
X_val_scaled = pd.DataFrame(
    scaler.transform(X_val_eng), columns=X_val_eng.columns, index=X_val_eng.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_eng), columns=X_test_eng.columns, index=X_test_eng.index
)


# ============================================================
# STEP 6: MODEL DEFINITIONS
# ============================================================
class QSARNetwork(nn.Module):
    """Wide shallow network for QSAR classification with strong regularization."""

    def __init__(self, input_dim, hidden_dims=[64, 32, 16], dropout_rate=0.3):
        super(QSARNetwork, self).__init__()
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 8))
        layers.append(nn.BatchNorm1d(8))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate * 0.5))
        layers.append(nn.Linear(8, 1))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x).squeeze(-1)


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced binary classification."""

    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        targets = targets.float()
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        pt = torch.exp(-bce_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss
        return loss.mean()


# ============================================================
# STEP 7: TRAIN NEURAL NETWORK
# ============================================================
input_dim = X_tr_scaled.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}, Input dim: {input_dim}")

nn_model = QSARNetwork(
    input_dim=input_dim, hidden_dims=[64, 32, 16], dropout_rate=0.3
).to(device)
criterion = FocalLoss(alpha=0.75, gamma=2.0)
optimizer = optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-5)

X_tr_tensor = torch.FloatTensor(X_tr_scaled.values).to(device)
y_tr_tensor = torch.FloatTensor(y_tr).to(device)
X_val_tensor = torch.FloatTensor(X_val_scaled.values).to(device)
X_test_tensor = torch.FloatTensor(X_test_scaled.values).to(device)

train_dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print("\n--- Training Neural Network ---")
best_val_auc = 0.0
patience_counter = 0
patience = 40
best_nn_state = None

for epoch in range(500):
    nn_model.train()
    epoch_loss = 0.0
    n_batches = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        logits = nn_model(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1
    scheduler.step()

    nn_model.eval()
    with torch.no_grad():
        val_logits = nn_model(X_val_tensor)
        val_probs = torch.sigmoid(val_logits).cpu().numpy()
        val_auc = roc_auc_score(y_val, val_probs)

    if (epoch + 1) % 50 == 0:
        print(
            f"Epoch {epoch+1:3d} | Loss: {epoch_loss/n_batches:.4f} | Val AUC: {val_auc:.4f}"
        )

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        patience_counter = 0
        best_nn_state = {k: v.clone() for k, v in nn_model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} (best AUC: {best_val_auc:.4f})")
            break

nn_model.load_state_dict(best_nn_state)
nn_model.eval()

with torch.no_grad():
    nn_val_probs = torch.sigmoid(nn_model(X_val_tensor)).cpu().numpy()
    nn_test_probs = torch.sigmoid(nn_model(X_test_tensor)).cpu().numpy()

nn_val_auc = roc_auc_score(y_val, nn_val_probs)
print(f"NN Final Val AUC: {nn_val_auc:.4f}")

# ============================================================
# STEP 8: TRAIN LIGHTGBM
# ============================================================
print("\n--- Training LightGBM ---")

n_pos = y_tr.sum()
n_neg = len(y_tr) - n_pos
pos_weight = n_neg / n_pos
print(f"Class weights: pos_weight={pos_weight:.2f}")

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 15,
    "max_depth": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": pos_weight,
    "verbose": -1,
    "seed": 42,
}

lgb_train_data = lgb.Dataset(X_tr_scaled.values, label=y_tr)
lgb_val_data = lgb.Dataset(X_val_scaled.values, label=y_val, reference=lgb_train_data)

lgb_model = lgb.train(
    lgb_params,
    lgb_train_data,
    num_boost_round=2000,
    valid_sets=[lgb_val_data],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(-1)],
)

lgb_val_probs = lgb_model.predict(
    X_val_scaled.values, num_iteration=lgb_model.best_iteration
)
lgb_test_probs = lgb_model.predict(
    X_test_scaled.values, num_iteration=lgb_model.best_iteration
)

lgb_val_auc = roc_auc_score(y_val, lgb_val_probs)
print(f"LightGBM Best Iteration: {lgb_model.best_iteration}")
print(f"LightGBM Val AUC: {lgb_val_auc:.4f}")

# ============================================================
# STEP 9: ENSEMBLE OPTIMIZATION
# ============================================================
print("\n--- Ensemble Optimization ---")

best_ensemble_auc = 0.0
best_w = 0.5

for w in np.arange(0.0, 1.01, 0.02):
    ensemble_probs = w * nn_val_probs + (1 - w) * lgb_val_probs
    auc = roc_auc_score(y_val, ensemble_probs)
    if auc > best_ensemble_auc:
        best_ensemble_auc = auc
        best_w = w

print(f"Best ensemble weight (NN): {best_w:.2f}")
print(f"Best ensemble Val AUC: {best_ensemble_auc:.4f}")

# ============================================================
# STEP 10: FINAL PREDICTIONS & SUBMISSION
# ============================================================
ensemble_test_probs = best_w * nn_test_probs + (1 - best_w) * lgb_test_probs

os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame(
    {"id": range(len(ensemble_test_probs)), "Class": ensemble_test_probs}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(
    f"Prediction stats: mean={ensemble_test_probs.mean():.4f}, std={ensemble_test_probs.std():.4f}"
)

# ============================================================
# FINAL VALIDATION SCORE (Official Metric: ROC-AUC)
# ============================================================
score = best_ensemble_auc
print(f"Final Validation Score: {score}")