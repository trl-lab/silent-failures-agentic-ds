import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import warnings

warnings.filterwarnings("ignore")

# Create directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("Loading data...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Identify feature columns
feature_cols = [c for c in train_df.columns if c.startswith("feat_")]
print(f"Number of features: {len(feature_cols)}")
print(f"Target classes: {sorted(train_df['target'].unique())}")

# ============================================================
# 2. SPLIT DATA FIRST (before any processing to avoid leakage)
# ============================================================
train_split, val_split = train_test_split(
    train_df, test_size=0.2, random_state=42, stratify=train_df["target"]
)
print(f"Train split: {train_split.shape}, Val split: {val_split.shape}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================


def engineer_features(df, feature_cols, train_feature_stats=None):
    X = df[feature_cols].copy().astype(np.float64)

    X_log = np.log1p(X)
    X_sqrt = np.sqrt(X)

    meta_features = pd.DataFrame(index=X.index)
    meta_features["feat_mean"] = X.mean(axis=1)
    meta_features["feat_std"] = X.std(axis=1)
    meta_features["feat_sum"] = X.sum(axis=1)
    meta_features["feat_max"] = X.max(axis=1)
    meta_features["feat_min"] = X.min(axis=1)
    meta_features["feat_median"] = X.median(axis=1)
    meta_features["feat_range"] = X.max(axis=1) - X.min(axis=1)
    meta_features["feat_skew"] = X.skew(axis=1)
    meta_features["feat_kurt"] = X.kurtosis(axis=1)
    meta_features["feat_nz"] = (X > 0).sum(axis=1)
    meta_features["feat_nz_ratio"] = (X > 0).mean(axis=1)
    meta_features["feat_top1"] = X.max(axis=1)
    meta_features["feat_top2"] = np.sort(X.values, axis=1)[:, -2]
    meta_features["feat_top3"] = np.sort(X.values, axis=1)[:, -3]
    meta_features["feat_top1_minus_top2"] = (
        meta_features["feat_top1"] - meta_features["feat_top2"]
    )
    meta_features["feat_top1_minus_top3"] = (
        meta_features["feat_top1"] - meta_features["feat_top3"]
    )
    X_log_np = X_log.values
    X_log_sum = X_log_np.sum(axis=1, keepdims=True) + 1e-10
    meta_features["feat_entropy"] = -(
        (X_log_np / X_log_sum) * np.log(X_log_np / X_log_sum + 1e-10)
    ).sum(axis=1)

    if train_feature_stats is not None:
        top_var_features = train_feature_stats["top_var_features"]
    else:
        variances = X.var()
        top_var_features = variances.nlargest(30).index.tolist()

    n_interactions = 0
    interaction_features = pd.DataFrame(index=X.index)
    for i in range(len(top_var_features)):
        for j in range(i + 1, len(top_var_features)):
            f1, f2 = top_var_features[i], top_var_features[j]
            interaction_features[f"inter_{f1}_{f2}"] = X[f1] * X[f2]
            n_interactions += 1
            if n_interactions >= 200:
                break
        if n_interactions >= 200:
            break

    if train_feature_stats is not None:
        bin_edges = train_feature_stats["bin_edges"]
    else:
        bin_edges = {}
        for col in top_var_features[:20]:
            quantiles = np.quantile(X[col], np.linspace(0, 1, 6))
            quantiles[0] = -np.inf
            quantiles[-1] = np.inf
            bin_edges[col] = quantiles

    binned_features = pd.DataFrame(index=X.index)
    for col, edges in bin_edges.items():
        binned_features[f"{col}_bin"] = np.digitize(X[col].values, edges) - 1

    ratio_features = pd.DataFrame(index=X.index)
    top5 = top_var_features[:5]
    for i in range(len(top5)):
        for j in range(i + 1, len(top5)):
            f1, f2 = top5[i], top5[j]
            ratio_features[f"ratio_{f1}_{f2}"] = X[f1] / (X[f2] + 1)

    diff_features = pd.DataFrame(index=X.index)
    for i in range(len(top5)):
        for j in range(i + 1, len(top5)):
            f1, f2 = top5[i], top5[j]
            diff_features[f"diff_{f1}_{f2}"] = X[f1] - X[f2]

    X_combined = pd.concat(
        [
            X,
            X_log,
            X_sqrt,
            meta_features,
            interaction_features,
            binned_features,
            ratio_features,
            diff_features,
        ],
        axis=1,
    )
    return X_combined


# Compute training statistics
print("Computing training feature statistics...")
train_raw = train_split[feature_cols].astype(np.float64)
variances = train_raw.var()
top_var_features = variances.nlargest(30).index.tolist()

bin_edges = {}
for col in top_var_features[:20]:
    quantiles = np.quantile(train_raw[col], np.linspace(0, 1, 6))
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf
    bin_edges[col] = quantiles

train_feature_stats = {"top_var_features": top_var_features, "bin_edges": bin_edges}

print("Engineering features for train split...")
X_train_eng = engineer_features(train_split, feature_cols, train_feature_stats)
print("Engineering features for val split...")
X_val_eng = engineer_features(val_split, feature_cols, train_feature_stats)
print("Engineering features for test...")
X_test_eng = engineer_features(test_df, feature_cols, train_feature_stats)
print(
    f"Engineered features - Train: {X_train_eng.shape}, Val: {X_val_eng.shape}, Test: {X_test_eng.shape}"
)

# ============================================================
# 4. FEATURE SELECTION using Mutual Information
# ============================================================
print("Performing feature selection...")
y_train = train_split["target"]
y_val = val_split["target"]

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)

print(f"Computing MI for {X_train_eng.shape[1]} features...")
mi_scores = mutual_info_classif(
    X_train_eng, y_train_enc, random_state=42, n_jobs=-1, discrete_features=False
)

mi_df = pd.DataFrame(
    {"feature": X_train_eng.columns, "mi_score": mi_scores}
).sort_values("mi_score", ascending=False)

n_select = min(300, len(mi_df))
selected_features = mi_df.head(n_select)["feature"].tolist()
print(f"Selected top {n_select} features by MI score")

X_train_sel = X_train_eng[selected_features]
X_val_sel = X_val_eng[selected_features]
X_test_sel = X_test_eng[selected_features]

# ============================================================
# 5. MODEL DEFINITION: TwoStreamTabularNet
# ============================================================


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.25, drop_prob=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.GELU()
        self.drop_prob = drop_prob

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        if self.training and self.drop_prob > 0:
            if torch.rand(1).item() < self.drop_prob:
                return residual
        return residual + x


class InteractionStream(nn.Module):
    def __init__(self, n_top_features, hidden_dim=128, dropout_rate=0.1):
        super().__init__()
        self.n_top = n_top_features
        n_pairs = n_top_features * (n_top_features - 1) // 2
        self.interaction_fc = nn.Linear(n_pairs, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x_top = x[:, : self.n_top]
        n = self.n_top
        i_idx = torch.triu_indices(n, n, 1)
        products = x_top[:, i_idx[0]] * x_top[:, i_idx[1]]
        h = self.interaction_fc(products)
        h = self.norm(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc_out(h)
        return h


class TwoStreamTabularNet(nn.Module):
    def __init__(
        self,
        n_features,
        n_classes,
        hidden_dim=256,
        n_blocks=4,
        n_top_interactions=30,
        dropout_rate=0.15,
    ):
        super().__init__()
        self.n_top = n_top_interactions
        self.input_norm = nn.LayerNorm(n_features)
        self.main_fc_in = nn.Linear(n_features, hidden_dim)
        self.main_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(n_blocks)]
        )
        self.main_norm = nn.LayerNorm(hidden_dim)
        self.interaction_stream = InteractionStream(
            n_top_interactions, hidden_dim // 2, dropout_rate
        )
        fusion_dim = hidden_dim + hidden_dim // 2
        self.fusion_fc = nn.Linear(fusion_dim, hidden_dim)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.fusion_act = nn.GELU()
        self.fusion_dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_norm(x)
        h_main = self.main_fc_in(x)
        for block in self.main_blocks:
            h_main = block(h_main)
        h_main = self.main_norm(h_main)
        h_inter = self.interaction_stream(x)
        h_fused = torch.cat([h_main, h_inter], dim=1)
        h_fused = self.fusion_fc(h_fused)
        h_fused = self.fusion_norm(h_fused)
        h_fused = self.fusion_act(h_fused)
        h_fused = self.fusion_dropout(h_fused)
        logits = self.classifier(h_fused)
        return logits


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, num_classes, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, logits, targets):
        confidence = 1.0 - self.smoothing
        smooth_value = self.smoothing / self.num_classes
        one_hot = torch.zeros_like(logits).scatter(1, targets.unsqueeze(1), 1.0)
        smooth_targets = one_hot * confidence + smooth_value
        log_probs = F.log_softmax(logits, dim=1)
        return -(smooth_targets * log_probs).sum(dim=1).mean()


# ============================================================
# 6. TRAINING SETUP
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

n_features = X_train_sel.shape[1]
n_classes = len(le.classes_)
label_classes = le.classes_
label_to_idx = {c: i for i, c in enumerate(label_classes)}

model = TwoStreamTabularNet(
    n_features=n_features,
    n_classes=n_classes,
    hidden_dim=128,
    n_blocks=3,
    n_top_interactions=15,
    dropout_rate=0.25,
)
model.to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {n_params:,}")

criterion = LabelSmoothingCrossEntropy(num_classes=n_classes, smoothing=0.05)

# Prepare tensors
X_train_t = torch.FloatTensor(X_train_sel.values).to(device)
X_val_t = torch.FloatTensor(X_val_sel.values).to(device)
X_test_t = torch.FloatTensor(X_test_sel.values).to(device)

y_train_t = torch.LongTensor(y_train_enc).to(device)
y_val_t = torch.LongTensor(y_val_enc).to(device)

n_train = X_train_t.shape[0]
n_val = X_val_t.shape[0]

# ============================================================
# 7. TRAINING LOOP (with mixup, warmup + cosine LR)
# ============================================================
total_epochs = 400
batch_size = 1024
warmup_epochs = 20
patience = 50
best_val_loss = float("inf")
patience_counter = 0
best_epoch = 0
mixup_alpha = 0.3


def get_lr(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs * 1e-3
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 1e-6 + 0.5 * (1e-3 - 1e-6) * (1 + np.cos(np.pi * progress))


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

for epoch in range(total_epochs):
    lr = get_lr(epoch)
    if epoch == 0:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=5e-4, betas=(0.9, 0.999)
        )
    else:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    model.train()
    perm = torch.randperm(n_train, device=device)
    epoch_loss = 0.0
    n_batches = 0

    for i in range(0, n_train, batch_size):
        idx = perm[i : i + batch_size]
        xb = X_train_t[idx]
        yb = y_train_t[idx]

        if np.random.random() < 0.5:
            xb, y_a, y_b, lam = mixup_data(xb, yb, mixup_alpha)
            optimizer.zero_grad()
            logits = model(xb)
            log_probs = F.log_softmax(logits, dim=1)
            loss_a = F.nll_loss(log_probs, y_a)
            loss_b = F.nll_loss(log_probs, y_b)
            loss = lam * loss_a + (1 - lam) * loss_b
        else:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    # Validation
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)
        val_probs = F.softmax(val_logits, dim=1).cpu().numpy()
        val_loss = log_loss(y_val_t.cpu().numpy(), val_probs, labels=range(n_classes))

    avg_train_loss = epoch_loss / n_batches

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(
            f"Epoch {epoch+1:3d} | Train: {avg_train_loss:.4f} | Val LogLoss: {val_loss:.4f} | LR: {lr:.6f}"
        )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save(model.state_dict(), "./working/best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(
                f"Early stopping at epoch {epoch+1} (best: epoch {best_epoch}, loss: {best_val_loss:.4f})"
            )
            break

print(
    f"\nTraining complete. Best epoch: {best_epoch}, Best val log_loss: {best_val_loss:.6f}"
)

# ============================================================
# 8. LOAD BEST MODEL AND GENERATE PREDICTIONS
# ============================================================
model.load_state_dict(torch.load("./working/best_model.pt"))
model.eval()

# Validation predictions
with torch.no_grad():
    val_logits = model(X_val_t)
    val_probs = F.softmax(val_logits, dim=1).cpu().numpy()

val_log_loss = log_loss(y_val_t.cpu().numpy(), val_probs, labels=range(n_classes))
print(f"Validation Log Loss (best model): {val_log_loss:.6f}")

# Test predictions with TTA
print("Generating test predictions with TTA...")
with torch.no_grad():
    test_logits = model(X_test_t)
    test_probs_base = F.softmax(test_logits, dim=1).cpu().numpy()

n_tta = 5
tta_probs = np.zeros_like(test_probs_base)
tta_probs += test_probs_base

for tta_i in range(n_tta):
    noise_std = 0.01
    with torch.no_grad():
        noisy_x = X_test_t + torch.randn_like(X_test_t) * noise_std
        noisy_logits = model(noisy_x)
        noisy_probs = F.softmax(noisy_logits, dim=1).cpu().numpy()
        tta_probs += noisy_probs

test_probs = tta_probs / (n_tta + 1)
test_probs = np.clip(test_probs, 1e-15, 1.0)
test_probs = test_probs / test_probs.sum(axis=1, keepdims=True)

# ============================================================
# 9. GENERATE SUBMISSION
# ============================================================
# Submission must have exactly two columns: id and target
# For log_loss metric, target should be the predicted class (the competition evaluates log_loss on the predicted class)
# However, since the metric is log_loss and the task says "target (predicted values)",
# we output the predicted class as integer labels
predicted_classes = np.argmax(test_probs, axis=1)
predicted_labels = le.inverse_transform(predicted_classes)
submission = pd.DataFrame({"id": range(len(test_probs)), "target": predicted_labels})
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")
print(submission.head())

# ============================================================
# 10. FINAL OUTPUT
# ============================================================
print(f"\n{'='*60}")
print(f"MODEL: TwoStreamTabularNet (hidden=128, blocks=3, interactions=15)")
print(f"AUGMENTATION: Mixup (alpha=0.3, 50% probability)")
print(f"REGULARIZATION: Dropout=0.25, Weight Decay=5e-4, Label Smoothing=0.05, Stochastic Depth=0.1")
print(f"LR SCHEDULE: Linear warmup ({warmup_epochs} epochs) + Cosine decay")
print(f"TTA: {n_tta} noise augmentations (std=0.01)")
print(f"Validation: Stratified 80/20 split, random_state=42")
print(f"Metric: log_loss (official competition metric)")
score = val_log_loss
print(f"Final Validation Score: {score}")