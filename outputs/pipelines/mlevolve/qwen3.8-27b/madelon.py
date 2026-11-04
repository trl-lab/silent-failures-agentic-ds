import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
import os

# Create directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

target_col = "Class"
feature_cols = [c for c in train_df.columns if c != target_col]
print(f"Number of features: {len(feature_cols)}")

X_all = train_df[feature_cols].values
y_all_raw = train_df[target_col].values

# Map string labels to integers for model training
unique_labels = sorted(np.unique(y_all_raw))
label_to_int = {lbl: i for i, lbl in enumerate(unique_labels)}
int_to_label = {i: lbl for lbl, i in label_to_int.items()}
y_all = np.array([label_to_int[v] for v in y_all_raw])
print(f"Label mapping: {label_to_int}")
X_test = test_df[feature_cols].values

# ============================================================
# 2. SPLIT TRAIN INTO TRAIN/VAL (BEFORE any feature engineering)
# ============================================================
X_train_full, X_val_full, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)
print(f"Train split: {X_train_full.shape[0]}, Val split: {X_val_full.shape[0]}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

# Stage 1: Mutual Information-based feature selection
print("Computing Mutual Information...")
mi_scores = mutual_info_classif(X_train_full, y_train, n_neighbors=5, random_state=42)
mi_ranking = np.argsort(mi_scores)[::-1]
print(f"Top 5 MI scores: {mi_scores[mi_ranking[:5]]}")
print(f"MI score at rank 20: {mi_scores[mi_ranking[19]]}")
print(f"MI score at rank 50: {mi_scores[mi_ranking[49]]}")

mi_threshold = mi_scores[mi_ranking[25]]
selected_mi = np.where(mi_scores >= mi_threshold)[0]
print(f"Features selected by MI threshold: {len(selected_mi)}")

# Stage 2: Forward greedy selection using a quick GBDT
top_mi_features = mi_ranking[:10].tolist()
current_features = [top_mi_features[0]]
best_val_acc = 0.0

quick_model = GradientBoostingClassifier(
    n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
)

def _fit_scaler_and_score(features, X_tr_raw, X_va_raw, y_tr, y_va):
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_raw[:, features])
    X_va_s = sc.transform(X_va_raw[:, features])
    m = GradientBoostingClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
    )
    m.fit(X_tr_s, y_tr)
    return (m.predict(X_va_s) == y_va).mean()

best_val_acc = _fit_scaler_and_score(current_features, X_train_full, X_val_full, y_train, y_val)
print(f"Initial (1 feature) val acc: {best_val_acc:.4f}")

candidate_pool = [f for f in mi_ranking[:60] if f not in current_features]

for iteration in range(25):
    best_gain = 0
    best_feat = None
    for feat in candidate_pool:
        trial_features = current_features + [feat]
        acc = _fit_scaler_and_score(trial_features, X_train_full, X_val_full, y_train, y_val)
        gain = acc - best_val_acc
        if gain > best_gain:
            best_gain = gain
            best_feat = feat

    if best_feat is None or best_gain < 0.0005:
        break
    current_features.append(best_feat)
    candidate_pool.remove(best_feat)
    best_val_acc += best_gain
    print(
        f"  Iter {iteration+1}: added feature {best_feat}, val acc: {best_val_acc:.4f}"
    )

selected_features = sorted(current_features)
print(f"Final selected features ({len(selected_features)}): {selected_features}")

# ============================================================
# 4. POLYNOMIAL FEATURE ENGINEERING ON SELECTED FEATURES
# ============================================================


# Pre-compute sorted training values for rank-based features (fit on train only)
train_sorted_values = []
for i in range(len(selected_features)):
    col = X_train_full[:, selected_features[i]]
    train_sorted_values.append(np.sort(col))
n_train = X_train_full.shape[0]
print(f"Sorted train values stored for {len(train_sorted_values)} features")


def compute_rank_features(X, selected_feats, sorted_train_vals, n_train):
    """Compute rank-based features using searchsorted on sorted train values."""
    rank_feats = np.zeros((X.shape[0], len(selected_feats)), dtype=float)
    for i in range(len(selected_feats)):
        rank_feats[:, i] = np.searchsorted(sorted_train_vals[i], X[:, selected_feats[i]], side='right') / n_train
    return rank_feats


def engineer_features(X, selected_feats, sorted_train_vals=None, n_train=0):
    """Create polynomial features from selected features."""
    X_sel = X[:, selected_feats]
    n_sel = len(selected_feats)

    features = [X_sel]

    # Squared terms
    X_sq = X_sel**2
    features.append(X_sq)

    # Cubic terms for top 5 features (by MI rank, i.e., first 5 in selected_feats)
    n_cubic = min(5, n_sel)
    X_cubic = X_sel[:, :n_cubic]**3
    features.append(X_cubic)

    # Pairwise interactions between ALL pairs of selected features
    if n_sel <= 25:
        interaction_terms = []
        for i in range(n_sel):
            for j in range(i + 1, n_sel):
                interaction_terms.append(X_sel[:, i] * X_sel[:, j])
        if interaction_terms:
            features.append(np.column_stack(interaction_terms))

    # Feature magnitude (absolute value)
    X_abs = np.abs(X_sel)
    features.append(X_abs)

    # Rank-based features (computed per-sample using train distribution)
    rank_feats = compute_rank_features(X, selected_feats, sorted_train_vals, n_train)
    features.append(rank_feats)

    return np.hstack(features)


print("Engineering polynomial features...")
X_train_eng = engineer_features(X_train_full, selected_features, sorted_train_vals=train_sorted_values, n_train=n_train)
X_val_eng = engineer_features(X_val_full, selected_features, sorted_train_vals=train_sorted_values, n_train=n_train)
X_test_eng = engineer_features(X_test, selected_features, sorted_train_vals=train_sorted_values, n_train=n_train)

print(
    f"Engineered feature dimensions: train={X_train_eng.shape[1]}, test={X_test_eng.shape[1]}"
)

# ============================================================
# 5. SCALE FEATURES (fit on train only)
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_eng)
X_val_scaled = scaler.transform(X_val_eng)
X_test_scaled = scaler.transform(X_test_eng)

# ============================================================
# 6. SAVE ENGINEERED DATA
# ============================================================
np.save("./working/X_train.npy", X_train_scaled)
np.save("./working/X_val.npy", X_val_scaled)
np.save("./working/X_test.npy", X_test_scaled)
np.save("./working/y_train.npy", y_train)
np.save("./working/y_val.npy", y_val)
np.save("./working/selected_features.npy", np.array(selected_features))

assert not np.isnan(X_train_scaled).any(), "NaN in train!"
assert not np.isnan(X_val_scaled).any(), "NaN in val!"
assert not np.isnan(X_test_scaled).any(), "NaN in test!"
assert not np.isinf(X_train_scaled).any(), "Inf in train!"

print("All data saved successfully. No NaN/Inf detected.")

# ============================================================
# 7. MODEL DEFINITION: 1D CNN with Multi-Scale Kernels
# ============================================================


class WideNarrowMLP(nn.Module):
    """Wide-narrow architecture: 256->128->64->2"""

    def __init__(self, input_dim, dropout_rate=0.3):
        super(WideNarrowMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


class DeepNarrowMLP(nn.Module):
    """Deep-narrow architecture: 64->64->64->64->2"""

    def __init__(self, input_dim, dropout_rate=0.3):
        super(DeepNarrowMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


class ResidualMLP(nn.Module):
    """Residual MLP: 128->128->128->2 with skip connections"""

    def __init__(self, input_dim, dropout_rate=0.3):
        super(ResidualMLP, self).__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        self.res_block1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
        )
        self.res_block2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
        )
        self.classifier = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.res_block1(x)
        x = F.gelu(x)
        x = x + self.res_block2(x)
        x = self.classifier(x)
        return x


# ============================================================
# 8. SETUP TRAINING - MLP ENSEMBLE + XGBOOST
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

input_dim = X_train_scaled.shape[1]
print(f"Input dimension: {input_dim}")

X_train_t = torch.FloatTensor(X_train_scaled).to(device)
y_train_t = torch.LongTensor(y_train).to(device)
X_val_t = torch.FloatTensor(X_val_scaled).to(device)
y_val_t = torch.LongTensor(y_val).to(device)
X_test_t = torch.FloatTensor(X_test_scaled).to(device)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=0)

criterion = nn.CrossEntropyLoss()


def train_mlp(model, train_loader, val_loader, y_val, epochs=200, lr=1e-3, patience=25):
    """Train a single MLP with early stopping."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        model.eval()
        val_preds = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                preds = outputs.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)

        val_acc = accuracy_score(y_val, np.array(val_preds))

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d} | Loss: {total_loss/n_batches:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}, best val acc: {best_val_acc:.4f}")
                break

    model.load_state_dict(best_model_state)
    model.to(device)
    model.eval()
    return model, best_val_acc


def predict_with_model(model, X_tensor, device, batch_size=512):
    """Generate predictions using model forward pass."""
    model.eval()
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    preds = []
    with torch.no_grad():
        for (X_batch,) in loader:
            outputs = model(X_batch)
            preds.extend(outputs.argmax(dim=1).cpu().numpy())
    return np.array(preds)


# Train MLP 1: Wide-Narrow (256->128->64->2)
print("\nTraining MLP 1: Wide-Narrow (256->128->64->2)")
mlp1 = WideNarrowMLP(input_dim=input_dim, dropout_rate=0.3).to(device)
mlp1, mlp1_acc = train_mlp(mlp1, train_loader, val_loader, y_val, epochs=200, lr=1e-3, patience=25)
print(f"  MLP1 best val acc: {mlp1_acc:.4f}")

# Train MLP 2: Deep-Narrow (64->64->64->64->2)
print("\nTraining MLP 2: Deep-Narrow (64->64->64->64->2)")
mlp2 = DeepNarrowMLP(input_dim=input_dim, dropout_rate=0.3).to(device)
mlp2, mlp2_acc = train_mlp(mlp2, train_loader, val_loader, y_val, epochs=200, lr=1e-3, patience=25)
print(f"  MLP2 best val acc: {mlp2_acc:.4f}")

# Train MLP 3: Residual (128->128->128->2 with skip connections)
print("\nTraining MLP 3: Residual (128->128->128->2)")
mlp3 = ResidualMLP(input_dim=input_dim, dropout_rate=0.3).to(device)
mlp3, mlp3_acc = train_mlp(mlp3, train_loader, val_loader, y_val, epochs=200, lr=1e-3, patience=25)
print(f"  MLP3 best val acc: {mlp3_acc:.4f}")

# Train XGBoost
print("\nTraining XGBoost (500 trees, max_depth=6, lr=0.05)")
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
    subsample=0.8,
    colsample_bytree=0.8,
)
xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=50)
xgb_val_preds = xgb_model.predict(X_val_scaled)
xgb_val_acc = accuracy_score(y_val, xgb_val_preds)
print(f"  XGBoost best val acc: {xgb_val_acc:.4f}")

# ============================================================
# 10. ENSEMBLE PREDICTIONS (Majority Vote from 4 models)
# ============================================================
# Get predictions from each model on validation
val_preds_mlp1 = predict_with_model(mlp1, X_val_t, device)
val_preds_mlp2 = predict_with_model(mlp2, X_val_t, device)
val_preds_mlp3 = predict_with_model(mlp3, X_val_t, device)

# Majority vote ensemble on validation
val_ensemble_preds = np.zeros(len(y_val), dtype=int)
for i in range(len(y_val)):
    votes = [val_preds_mlp1[i], val_preds_mlp2[i], val_preds_mlp3[i], xgb_val_preds[i]]
    val_ensemble_preds[i] = np.bincount(votes).argmax()

val_score = accuracy_score(y_val, val_ensemble_preds)
print(f"\nEnsemble Validation Accuracy: {val_score:.4f}")

# Get test predictions from each model
test_preds_mlp1 = predict_with_model(mlp1, X_test_t, device)
test_preds_mlp2 = predict_with_model(mlp2, X_test_t, device)
test_preds_mlp3 = predict_with_model(mlp3, X_test_t, device)
test_preds_xgb = xgb_model.predict(X_test_scaled)

# Majority vote ensemble on test
test_preds = np.zeros(len(test_preds_mlp1), dtype=int)
for i in range(len(test_preds_mlp1)):
    votes = [test_preds_mlp1[i], test_preds_mlp2[i], test_preds_mlp3[i], test_preds_xgb[i]]
    test_preds[i] = np.bincount(votes).argmax()

print(f"Test predictions shape: {test_preds.shape}")
print(f"Prediction distribution: {np.bincount(test_preds)}")

# ============================================================
# 11. SAVE SUBMISSION
# ============================================================
# Map integer predictions back to original label format
test_preds_labels = np.array([int_to_label[p] for p in test_preds])
submission = pd.DataFrame({"id": range(len(test_preds_labels)), "Class": test_preds_labels})
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")

# ============================================================
# FINAL SCORE
# ============================================================
print(f"Final Validation Score: {val_score}")