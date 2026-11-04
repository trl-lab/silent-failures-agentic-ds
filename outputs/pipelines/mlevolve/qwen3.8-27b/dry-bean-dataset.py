import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import os
import copy
import joblib

# Create directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Classes: {train_df['Class'].unique()}")
print(f"Class distribution:\n{train_df['Class'].value_counts()}")

# Separate target
y = train_df["Class"].copy()
X_train_full = train_df.drop(columns=["Class"]).copy()
X_test = test_df.copy()

# ============================================================
# 2. CREATE STRATIFIED TRAIN/VAL SPLIT (BEFORE any transformation)
# ============================================================
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain split: {X_train.shape}, Val split: {X_val.shape}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================


def engineer_features(df, scaler=None, fit_scaler=False):
    """
    Engineer morphological shape features from raw bean measurements.
    """
    df = df.copy()

    # --- Log transforms for scale features ---
    df["LogArea"] = np.log1p(df["Area"])
    df["LogPerimeter"] = np.log1p(df["Perimeter"])
    df["LogConvexArea"] = np.log1p(df["ConvexArea"])
    df["LogMajorAxis"] = np.log1p(df["MajorAxisLength"])
    df["LogMinorAxis"] = np.log1p(df["MinorAxisLength"])
    df["LogEquivDiameter"] = np.log1p(df["EquivDiameter"])

    # --- Shape ratio features (intrinsic properties) ---
    df["PerimeterToSqrtArea"] = df["Perimeter"] / np.sqrt(df["Area"] + 1e-8)
    df["AxisDiff"] = df["MajorAxisLength"] - df["MinorAxisLength"]
    df["AxisDiffNorm"] = df["AxisDiff"] / (df["MajorAxisLength"] + 1e-8)
    df["AxisProduct"] = df["MajorAxisLength"] * df["MinorAxisLength"]
    df["AxisProductToArea"] = df["AxisProduct"] / (df["Area"] + 1e-8)
    df["ConvexityRatio"] = df["ConvexArea"] / (df["Area"] + 1e-8)
    df["ConvexityDiff"] = df["ConvexArea"] - df["Area"]
    df["EquivToMajor"] = df["EquivDiameter"] / (df["MajorAxisLength"] + 1e-8)
    df["EquivToMinor"] = df["EquivDiameter"] / (df["MinorAxisLength"] + 1e-8)
    df["EquivToAxisMean"] = df["EquivDiameter"] / (
        (df["MajorAxisLength"] + df["MinorAxisLength"]) / 2 + 1e-8
    )
    df["ExtentSolidity"] = df["Extent"] * df["Solidity"]
    df["RoundnessCompactness"] = df["Roundness"] * df["Compactness"]
    df["EccentricityToAspect"] = df["Eccentricity"] / (df["AspectRatio"] + 1e-8)
    df["SolidityToExtent"] = df["Solidity"] / (df["Extent"] + 1e-8)

    # Polynomial features on key discriminators
    df["AspectRatioSq"] = df["AspectRatio"] ** 2
    df["EccentricitySq"] = df["Eccentricity"] ** 2
    df["SoliditySq"] = df["Solidity"] ** 2
    df["RoundnessSq"] = df["Roundness"] ** 2

    # Shape factor interactions
    df["SF1_SF2"] = df["ShapeFactor1"] * df["ShapeFactor2"]
    df["SF1_SF3"] = df["ShapeFactor1"] * df["ShapeFactor3"]
    df["SF2_SF3"] = df["ShapeFactor2"] * df["ShapeFactor3"]
    df["SF1_SF4"] = df["ShapeFactor1"] * df["ShapeFactor4"]
    df["SF2_SF4"] = df["ShapeFactor2"] * df["ShapeFactor4"]
    df["SF3_SF4"] = df["ShapeFactor3"] * df["ShapeFactor4"]
    df["ShapeFactorSum"] = (
        df["ShapeFactor1"]
        + df["ShapeFactor2"]
        + df["ShapeFactor3"]
        + df["ShapeFactor4"]
    )
    df["ShapeFactorMean"] = df["ShapeFactorSum"] / 4.0

    # Area-normalized features
    df["PerimeterPerArea"] = df["Perimeter"] / (df["Area"] + 1e-8)
    df["MajorPerArea"] = df["MajorAxisLength"] / (df["Area"] + 1e-8)
    df["MinorPerArea"] = df["MinorAxisLength"] / (df["Area"] + 1e-8)

    # Squared axis features
    df["MajorAxisSq"] = df["MajorAxisLength"] ** 2
    df["MinorAxisSq"] = df["MinorAxisLength"] ** 2
    df["AxisSqRatio"] = df["MajorAxisSq"] / (df["MinorAxisSq"] + 1e-8)

    # --- Scaling ---
    if fit_scaler:
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    elif scaler is not None:
        df = pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)

    return df, scaler


# Apply feature engineering - FIT on train only
X_train_eng, scaler = engineer_features(X_train, scaler=None, fit_scaler=True)
X_val_eng, _ = engineer_features(X_val, scaler=scaler, fit_scaler=False)
X_test_eng, _ = engineer_features(X_test, scaler=scaler, fit_scaler=False)

print(f"\nEngineered features: {X_train_eng.shape[1]}")

# ============================================================
# 4. MODEL ARCHITECTURE: Residual MLP
# ============================================================


class ResidualBlock(nn.Module):
    """Residual block with batch norm, dropout, and ReLU."""

    def __init__(self, dim, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out = self.dropout(out)
        out = out + residual
        return self.relu(out)


class BeanClassifier(nn.Module):
    """
    Deep MLP with residual connections for bean variety classification.
    """

    def __init__(
        self, input_dim, hidden_dim=128, num_classes=7, num_blocks=4, dropout_rate=0.2
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.classifier(x)
        return x


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing."""

    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        smooth_targets = torch.full_like(
            log_probs, self.smoothing / (self.num_classes - 1)
        )
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        return loss.mean()


# ============================================================
# 5. SETUP TENSORS AND DATA LOADERS
# ============================================================
class_to_idx = {c: i for i, c in enumerate(sorted(y_train.unique()))}
idx_to_class = {i: c for c, i in class_to_idx.items()}
num_classes = len(class_to_idx)
print(f"Number of classes: {num_classes}")
print(f"Class mapping: {class_to_idx}")

X_train_tensor = torch.FloatTensor(X_train_eng.values)
y_train_tensor = torch.LongTensor([class_to_idx[c] for c in y_train.values])
X_val_tensor = torch.FloatTensor(X_val_eng.values)
y_val_tensor = torch.LongTensor([class_to_idx[c] for c in y_val.values])
X_test_tensor = torch.FloatTensor(X_test_eng.values)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=2)

input_dim = X_train_tensor.shape[1]
print(f"Input dimension: {input_dim}")

# ============================================================
# 6. INITIALIZATION
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = BeanClassifier(
    input_dim=input_dim,
    hidden_dim=128,
    num_classes=num_classes,
    num_blocks=4,
    dropout_rate=0.2,
)
criterion = LabelSmoothingCrossEntropy(num_classes=num_classes, smoothing=0.1)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999)
)

num_epochs = 200
warmup_epochs = 10


def get_lr(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
    return 0.5 * (1 + np.cos(np.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)
model.to(device)

# ============================================================
# 7. MIXUP AUGMENTATION
# ============================================================


def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation to a batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    """Compute mixup loss."""
    loss_a = criterion(logits, y_a)
    loss_b = criterion(logits, y_b)
    return lam * loss_a + (1 - lam) * loss_b


# ============================================================
# 8. TRAINING LOOP WITH EARLY STOPPING
# ============================================================
best_val_acc = 0.0
best_model_state = None
patience_counter = 0
patience = 25
mixup_alpha = 0.2

print(f"\n{'='*60}")
print(f"Training BeanClassifier (Residual MLP)")
print(f"Epochs: {num_epochs}, Patience: {patience}, Mixup alpha: {mixup_alpha}")
print(f"{'='*60}")

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        X_mixed, y_a, y_b, lam = mixup_data(X_batch, y_batch, alpha=mixup_alpha)

        optimizer.zero_grad()
        logits = model(X_mixed)
        loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item() * X_batch.size(0)
        with torch.no_grad():
            orig_logits = model(X_batch)
            train_correct += (orig_logits.argmax(dim=1) == y_batch).sum().item()
        train_total += X_batch.size(0)

    scheduler.step()

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            val_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            val_total += X_batch.size(0)

    val_acc = val_correct / val_total
    train_acc = train_correct / train_total
    current_lr = optimizer.param_groups[0]["lr"]

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(
            f"Epoch {epoch+1:3d} | Train Loss: {train_loss/train_total:.4f} | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1} (best val acc: {best_val_acc:.4f})")
        break

# ============================================================
# 9. LOAD BEST MODEL AND GENERATE PREDICTIONS
# ============================================================
model.load_state_dict(best_model_state)
model.eval()

with torch.no_grad():
    val_logits = model(X_val_tensor.to(device))
    val_pred_idx = val_logits.argmax(dim=1).cpu().numpy()
    val_pred = np.array([idx_to_class[i] for i in val_pred_idx])

val_score = accuracy_score(y_val.values, val_pred)
print(f"\nBest Validation Accuracy: {val_score:.6f}")

with torch.no_grad():
    test_logits = model(X_test_tensor.to(device))
    test_pred_idx = test_logits.argmax(dim=1).cpu().numpy()
    test_pred = np.array([idx_to_class[i] for i in test_pred_idx])

# ============================================================
# 10. SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame({"id": range(len(test_pred)), "Class": test_pred})
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")
print(submission.head())

# Save best model
torch.save(best_model_state, "./working/bean_mlp_best.pt")
print(f"Best model saved to ./working/bean_mlp_best.pt")

# ============================================================
# FINAL OUTPUT
# ============================================================
score = val_score
print(f"Final Validation Score: {score}")
