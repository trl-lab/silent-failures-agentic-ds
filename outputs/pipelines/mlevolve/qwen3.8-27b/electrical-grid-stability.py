import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import os
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Target distribution:\n{train_df['stabf'].value_counts(normalize=True)}")

# Encode string target to numeric (0/1)
target_encoding = {'stable': 0, 'unstable': 1}
train_df['stabf'] = train_df['stabf'].map(target_encoding)
print(f"Encoded target distribution:\n{train_df['stabf'].value_counts()}")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
base_features = [
    "tau1",
    "tau2",
    "tau3",
    "tau4",
    "p1",
    "p2",
    "p3",
    "p4",
    "g1",
    "g2",
    "g3",
    "g4",
]
target_col = "stabf"


def engineer_features(df):
    """Physics-informed feature engineering for 4-node star grid system."""
    features = {}

    # Base features
    for col in base_features:
        features[col] = df[col].values

    # Per-group aggregates
    for group_name, prefix in [("tau", "tau"), ("p", "p"), ("g", "g")]:
        cols = [f"{prefix}{i}" for i in range(1, 5)]
        group = df[cols].values

        features[f"{group_name}_sum"] = group.sum(axis=1)
        features[f"{group_name}_mean"] = group.mean(axis=1)
        features[f"{group_name}_std"] = group.std(axis=1)
        features[f"{group_name}_min"] = group.min(axis=1)
        features[f"{group_name}_max"] = group.max(axis=1)
        features[f"{group_name}_range"] = group.max(axis=1) - group.min(axis=1)
        features[f"{group_name}_prod"] = group.prod(axis=1)

        mean_val = group.mean(axis=1)
        std_val = group.std(axis=1)
        features[f"{group_name}_cv"] = std_val / (np.abs(mean_val) + 1e-10)

    # Cross-group interactions (same node)
    for i in range(1, 5):
        features[f"tau{i}_x_p{i}"] = df[f"tau{i}"].values * df[f"p{i}"].values
        features[f"tau{i}_x_g{i}"] = df[f"tau{i}"].values * df[f"g{i}"].values
        features[f"p{i}_x_g{i}"] = df[f"p{i}"].values * df[f"g{i}"].values

    # Cross-group aggregate interactions
    features["tau_sum_x_p_sum"] = features["tau_sum"] * features["p_sum"]
    features["tau_sum_x_g_sum"] = features["tau_sum"] * features["g_sum"]
    features["p_sum_x_g_sum"] = features["p_sum"] * features["g_sum"]
    features["tau_mean_x_p_mean"] = features["tau_mean"] * features["p_mean"]
    features["tau_mean_x_g_mean"] = features["tau_mean"] * features["g_mean"]
    features["p_mean_x_g_mean"] = features["p_mean"] * features["g_mean"]

    # Inter-node ratios and differences
    for prefix in ["tau", "p", "g"]:
        for i in range(1, 5):
            for j in range(i + 1, 5):
                col_i = f"{prefix}{i}"
                col_j = f"{prefix}{j}"
                features[f"{col_i}_div_{col_j}"] = df[col_i].values / (
                    df[col_j].values + 1e-10
                )
                features[f"{col_i}_minus_{col_j}"] = df[col_i].values - df[col_j].values

    # Pairwise cross-group interactions (all node pairs)
    for i in range(1, 5):
        for j in range(1, 5):
            if i != j:
                features[f"tau{i}_x_p{j}"] = df[f"tau{i}"].values * df[f"p{j}"].values

    # Higher-order
    features["tau_mean_x_p_mean_x_g_mean"] = (
        features["tau_mean"] * features["p_mean"] * features["g_mean"]
    )
    features["tau_std_x_p_std"] = features["tau_std"] * features["p_std"]
    features["tau_std_x_g_std"] = features["tau_std"] * features["g_std"]
    features["p_std_x_g_std"] = features["p_std"] * features["g_std"]

    # Log transforms
    for prefix in ["tau", "p", "g"]:
        for i in range(1, 5):
            col = f"{prefix}{i}"
            features[f"log_{col}"] = np.log1p(np.abs(df[col].values)) * np.sign(
                df[col].values
            )

    # Quadratic features
    for prefix in ["tau", "p", "g"]:
        for i in range(1, 5):
            col = f"{prefix}{i}"
            features[f"{col}_sq"] = df[col].values ** 2

    # Normalized coupling
    for i in range(1, 5):
        tau_i = df[f"tau{i}"].values
        p_i = df[f"p{i}"].values
        g_i = df[f"g{i}"].values
        features[f"tau{i}_norm_p{i}"] = tau_i / (np.abs(tau_i) + np.abs(p_i) + 1e-10)
        features[f"tau{i}_norm_g{i}"] = tau_i / (np.abs(tau_i) + np.abs(g_i) + 1e-10)
        features[f"p{i}_norm_g{i}"] = p_i / (np.abs(p_i) + np.abs(g_i) + 1e-10)

    return pd.DataFrame(features)


print("\n--- Engineering features ---")
train_features = engineer_features(train_df)
test_features = engineer_features(test_df)
print(f"Train features shape: {train_features.shape}")
print(f"Test features shape: {test_features.shape}")

# ============================================================
# 3. SPLIT AND SCALE
# ============================================================
y = train_df[target_col].values
X = train_features

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"X_train: {X_train.shape}, X_val: {X_val.shape}")
print(f"Train target dist: {np.bincount(y_train.astype(int))}")
print(f"Val target dist: {np.bincount(y_val.astype(int))}")
print(f"Train target mean: {y_train.mean():.4f}, Val target mean: {y_val.mean():.4f}")

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
)
X_val_scaled = pd.DataFrame(
    scaler.transform(X_val), columns=X_val.columns, index=X_val.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(test_features),
    columns=test_features.columns,
    index=test_features.index,
)

X_train_scaled = X_train_scaled.replace([np.inf, -np.inf], np.nan).fillna(0)
X_val_scaled = X_val_scaled.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test_scaled = X_test_scaled.replace([np.inf, -np.inf], np.nan).fillna(0)

print(
    f"After scaling - Any NaN: train={X_train_scaled.isnull().any().any()}, "
    f"val={X_val_scaled.isnull().any().any()}, test={X_test_scaled.isnull().any().any()}"
)


# ============================================================
# 4. PREPARE TENSORS
# ============================================================
def build_graph_features(X_df):
    """Reshape into node-feature matrix: (N, 4, 3)"""
    N = len(X_df)
    node_features = np.zeros((N, 4, 3), dtype=np.float32)
    for node in range(4):
        node_features[:, node, 0] = X_df[f"tau{node+1}"].values
        node_features[:, node, 1] = X_df[f"p{node+1}"].values
        node_features[:, node, 2] = X_df[f"g{node+1}"].values
    return node_features


X_train_nodes = torch.tensor(build_graph_features(X_train_scaled), dtype=torch.float32)
X_val_nodes = torch.tensor(build_graph_features(X_val_scaled), dtype=torch.float32)
X_test_nodes = torch.tensor(build_graph_features(X_test_scaled), dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)

X_train_flat = X_train_scaled.values.astype(np.float32)
X_val_flat = X_val_scaled.values.astype(np.float32)
X_test_flat = X_test_scaled.values.astype(np.float32)

test_ids = np.arange(len(test_df))


# ============================================================
# 5. MODEL DEFINITIONS
# ============================================================
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.attention = nn.Linear(in_features, out_features, bias=False)
        self.fc = nn.Linear(in_features * 2, out_features, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        N, V, _ = x.shape
        h = self.attention(x)
        center_h = h[:, 0:1, :]
        consumer_h = h[:, 1:4, :]
        center_att = self.leaky_relu(center_h + consumer_h)
        center_att = F.softmax(center_att, dim=1)
        center_agg = (center_att * consumer_h).sum(dim=1, keepdim=True)
        # Each consumer attends to the single center: broadcast center to all consumers
        consumer_agg = center_h.expand_as(consumer_h)
        center_out = torch.cat([h[:, 0:1, :], center_agg], dim=-1)
        consumer_out = torch.cat([h[:, 1:4, :], consumer_agg], dim=-1)
        center_out = self.fc(center_out)
        consumer_out = self.fc(consumer_out)
        out = torch.cat([center_out, consumer_out], dim=1)
        return self.dropout(out)


class StarGAT(nn.Module):
    def __init__(self, node_dim=3, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.gat1 = GATLayer(hidden_dim, hidden_dim, dropout=dropout)
        self.gat2 = GATLayer(hidden_dim, hidden_dim, dropout=dropout)
        self.gat3 = GATLayer(hidden_dim, hidden_dim, dropout=dropout)
        self.pool_attn = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        h = F.relu(self.input_proj(x))
        h1 = self.gat1(h)
        h2 = self.gat2(h1)
        h3 = self.gat3(h2)
        attn_weights = F.softmax(self.pool_attn(h3), dim=1)
        pooled = (attn_weights * h3).sum(dim=1)
        mean_pooled = h3.mean(dim=1)
        combined = torch.cat([pooled, mean_pooled], dim=-1)
        logit = self.classifier(combined).squeeze(-1)
        return logit


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * (1 - pt) ** self.gamma * bce
        return focal.mean()


# ============================================================
# 6. TRAIN GAT
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model = StarGAT(node_dim=3, hidden_dim=64, dropout=0.3).to(device)
print(f"GAT parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = FocalLoss(alpha=0.25, gamma=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=200, eta_min=1e-5
)

train_dataset = TensorDataset(X_train_nodes, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)

best_val_auc_gat = 0.0
patience_counter = 0
patience = 20
best_model_state_gat = None

X_val_nodes_dev = X_val_nodes.to(device)

for epoch in range(200):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    scheduler.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_nodes_dev)
        val_probs = torch.sigmoid(val_logits).cpu().numpy()
        val_auc = roc_auc_score(y_val, val_probs)

    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(
            f"GAT Epoch {epoch+1:3d} | Loss: {total_loss/n_batches:.4f} | Val AUC: {val_auc:.4f}"
        )

    if val_auc > best_val_auc_gat:
        best_val_auc_gat = val_auc
        best_model_state_gat = {
            k: v.cpu().clone() for k, v in model.state_dict().items()
        }
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"GAT Early stopping at epoch {epoch+1}")
            break

model.load_state_dict(best_model_state_gat)
model.to(device)
model.eval()
print(f"Best GAT Val AUC: {best_val_auc_gat:.4f}")

# ============================================================
# 7. TRAIN XGBOOST
# ============================================================
xgb_model = xgb.XGBClassifier(
    n_estimators=1500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric="auc",
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
)

xgb_model.fit(X_train_flat, y_train, eval_set=[(X_val_flat, y_val)], verbose=False)
print(f"XGBoost best iteration: {xgb_model.best_iteration}")

# ============================================================
# 7b. TRAIN LIGHTGBM
# ============================================================
lgb_model = lgb.LGBMClassifier(
    num_leaves=63,
    learning_rate=0.03,
    n_estimators=1500,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

lgb_model.fit(
    X_train_flat, y_train,
    eval_set=[(X_val_flat, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
)
print(f"LightGBM best iteration: {lgb_model.best_iteration_}")

# ============================================================
# 8. GENERATE PREDICTIONS
# ============================================================
with torch.no_grad():
    val_probs_gat = torch.sigmoid(model(X_val_nodes_dev)).cpu().numpy()
    test_probs_gat = torch.sigmoid(model(X_test_nodes.to(device))).cpu().numpy()

val_probs_xgb = xgb_model.predict_proba(X_val_flat)[:, 1]
test_probs_xgb = xgb_model.predict_proba(X_test_flat)[:, 1]

val_probs_lgb = lgb_model.predict_proba(X_val_flat)[:, 1]
test_probs_lgb = lgb_model.predict_proba(X_test_flat)[:, 1]

# ============================================================
# 9. OPTIMIZE 3-MODEL ENSEMBLE WEIGHTS
# ============================================================
best_w_gat = 0.33
best_w_xgb = 0.33
best_w_lgb = 0.34
best_ensemble_auc = 0.0

for w_gat in np.arange(0.0, 1.05, 0.1):
    for w_xgb in np.arange(0.0, 1.05 - w_gat, 0.1):
        w_lgb = 1.0 - w_gat - w_xgb
        if w_lgb < -0.01:
            continue
        w_lgb = max(w_lgb, 0.0)
        ens = w_gat * val_probs_gat + w_xgb * val_probs_xgb + w_lgb * val_probs_lgb
        auc = roc_auc_score(y_val, ens)
        if auc > best_ensemble_auc:
            best_ensemble_auc = auc
            best_w_gat = w_gat
            best_w_xgb = w_xgb
            best_w_lgb = w_lgb

print(f"Optimal weights - GAT: {best_w_gat:.2f}, XGB: {best_w_xgb:.2f}, LGB: {best_w_lgb:.2f}")

# ============================================================
# 10. FINAL ENSEMBLE AND EVALUATION
# ============================================================
val_probs_ensemble = best_w_gat * val_probs_gat + best_w_xgb * val_probs_xgb + best_w_lgb * val_probs_lgb
test_probs_ensemble = best_w_gat * test_probs_gat + best_w_xgb * test_probs_xgb + best_w_lgb * test_probs_lgb

val_auc_gat = roc_auc_score(y_val, val_probs_gat)
val_auc_xgb = roc_auc_score(y_val, val_probs_xgb)
val_auc_lgb = roc_auc_score(y_val, val_probs_lgb)
val_auc_ensemble = roc_auc_score(y_val, val_probs_ensemble)

print(f"\n--- Validation Results (ROC-AUC) ---")
print(f"GAT:        {val_auc_gat:.4f}")
print(f"XGBoost:    {val_auc_xgb:.4f}")
print(f"LightGBM:   {val_auc_lgb:.4f}")
print(f"Ensemble:   {val_auc_ensemble:.4f}")

# ============================================================
# 11. SAVE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame({"id": test_ids.astype(int), "stabf": test_probs_ensemble})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(submission.head())

# ============================================================
# FINAL OUTPUT
# ============================================================
final_val_score = val_auc_ensemble
print(f"Final Validation Score: {final_val_score}")
