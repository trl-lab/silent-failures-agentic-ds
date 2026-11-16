import os
import re
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from sklearn.svm import SVC


# ---------------------------------------------------------
# 1. Reproducibility & Directory Setup
# ---------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

INPUT_DIR = "./input"
SUBMISSION_DIR = "./submission"
WORKING_DIR = "./working"

os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)

# ---------------------------------------------------------
# 2. Data Loading & Feature Engineering
# ---------------------------------------------------------
train_path = os.path.join(INPUT_DIR, "train.parquet")
test_path = os.path.join(INPUT_DIR, "test.parquet")

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "class"
feature_cols = [c for c in train_df.columns if c != target_col]

# Encode target labels
le = LabelEncoder()
y_train_full = le.fit_transform(train_df[target_col].astype(str))


def engineer_kinematic_features(df, cols):
    """
    Synthesize physical domain features:
    - 3D Vector magnitudes from triaxial accelerometer/gyroscope signals
    - Row-wise kinetic energy, norm metrics, and distribution moments
    """
    X_raw = df[cols].copy()
    engineered_dict = {}

    # Identify triaxial triplets (matching -X, -Y, -Z patterns)
    triplet_bases = set()
    for col in cols:
        match = re.match(r"^(.*)[-_]([XxYyZz])$", col)
        if match:
            triplet_bases.add(match.group(1))

    for base in triplet_bases:
        x_col = (
            f"{base}-X"
            if f"{base}-X" in cols
            else f"{base}_X" if f"{base}_X" in cols else None
        )
        y_col = (
            f"{base}-Y"
            if f"{base}-Y" in cols
            else f"{base}_Y" if f"{base}_Y" in cols else None
        )
        z_col = (
            f"{base}-Z"
            if f"{base}-Z" in cols
            else f"{base}_Z" if f"{base}_Z" in cols else None
        )

        if x_col and y_col and z_col:
            x_val = X_raw[x_col].values
            y_val = X_raw[y_col].values
            z_val = X_raw[z_col].values

            # 3D Euclidean magnitude (invariant to sensor rotation)
            mag = np.sqrt(x_val**2 + y_val**2 + z_val**2 + 1e-8)
            engineered_dict[f"{base}_3d_magnitude"] = mag

            # Axial energy sum
            energy = (x_val**2 + y_val**2 + z_val**2) / 3.0
            engineered_dict[f"{base}_3d_energy"] = energy

            # Cross-axial normalized correlations
            engineered_dict[f"{base}_xy_corr"] = (x_val * y_val) / (mag**2 + 1e-8)
            engineered_dict[f"{base}_xz_corr"] = (x_val * z_val) / (mag**2 + 1e-8)
            engineered_dict[f"{base}_yz_corr"] = (y_val * z_val) / (mag**2 + 1e-8)

            # Spatial orientation angles (Pitch, Roll, Tilt)
            engineered_dict[f"{base}_pitch"] = np.arctan2(
                x_val, np.sqrt(y_val**2 + z_val**2 + 1e-8)
            )
            engineered_dict[f"{base}_roll"] = np.arctan2(
                y_val, np.sqrt(x_val**2 + z_val**2 + 1e-8)
            )
            engineered_dict[f"{base}_tilt"] = np.arccos(
                np.clip(z_val / mag, -1.0, 1.0)
            )

    # Postural gravity orientation angles and interactions
    for stat in ["mean()", "mean", "std()", "std"]:
        body_x = f"tBodyAcc-{stat}-X"
        body_y = f"tBodyAcc-{stat}-Y"
        body_z = f"tBodyAcc-{stat}-Z"
        grav_x = f"tGravityAcc-{stat}-X"
        grav_y = f"tGravityAcc-{stat}-Y"
        grav_z = f"tGravityAcc-{stat}-Z"

        if all(
            c in cols for c in [body_x, body_y, body_z, grav_x, grav_y, grav_z]
        ):
            bx, by, bz = (
                X_raw[body_x].values,
                X_raw[body_y].values,
                X_raw[body_z].values,
            )
            gx, gy, gz = (
                X_raw[grav_x].values,
                X_raw[grav_y].values,
                X_raw[grav_z].values,
            )

            b_mag = np.sqrt(bx**2 + by**2 + bz**2 + 1e-8)
            g_mag = np.sqrt(gx**2 + gy**2 + gz**2 + 1e-8)
            dot_bg = bx * gx + by * gy + bz * gz
            cos_sim = np.clip(dot_bg / (b_mag * g_mag + 1e-8), -1.0, 1.0)

            engineered_dict[f"gravity_body_cos_sim_{stat}"] = cos_sim
            engineered_dict[f"gravity_body_angle_{stat}"] = np.arccos(cos_sim)
            engineered_dict[f"gravity_body_diff_mag_{stat}"] = np.sqrt(
                (bx - gx) ** 2 + (by - gy) ** 2 + (bz - gz) ** 2 + 1e-8
            )

    # Frequency-domain vs time-domain spectral energy ratios
    f_cols = [c for c in cols if c.startswith("f") or "Freq" in c or "freq" in c]
    t_cols = [c for c in cols if c.startswith("t") or "Time" in c or "time" in c]
    if len(f_cols) > 0 and len(t_cols) > 0:
        f_vals = X_raw[f_cols].values
        t_vals = X_raw[t_cols].values
        f_energy = np.mean(f_vals**2, axis=1)
        t_energy = np.mean(t_vals**2, axis=1)
        engineered_dict["freq_time_energy_ratio"] = f_energy / (t_energy + 1e-8)
        engineered_dict["freq_mean_ratio"] = np.mean(
            np.abs(f_vals), axis=1
        ) / (np.mean(np.abs(t_vals), axis=1) + 1e-8)

    # Row-wise statistical distribution moments across all kinematic channels
    raw_values = X_raw.values
    engineered_dict["row_mean"] = np.mean(raw_values, axis=1)
    engineered_dict["row_std"] = np.std(raw_values, axis=1)
    engineered_dict["row_median"] = np.median(raw_values, axis=1)
    engineered_dict["row_q25"] = np.percentile(raw_values, 25, axis=1)
    engineered_dict["row_q75"] = np.percentile(raw_values, 75, axis=1)
    engineered_dict["row_iqr"] = engineered_dict["row_q75"] - engineered_dict["row_q25"]
    engineered_dict["row_l2_norm"] = np.linalg.norm(raw_values, axis=1)
    engineered_dict["row_energy"] = np.mean(raw_values**2, axis=1)
    engineered_dict["row_skew"] = skew(raw_values, axis=1)
    engineered_dict["row_kurtosis"] = kurtosis(raw_values, axis=1)

    engineered_df = pd.DataFrame(engineered_dict, index=df.index)
    full_features_df = pd.concat([X_raw, engineered_df], axis=1)
    return full_features_df


# Generate engineered feature sets
X_train_full = engineer_kinematic_features(train_df, feature_cols)
X_test = engineer_kinematic_features(test_df, feature_cols)

# Stratified Train/Validation Split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.20, random_state=42, stratify=y_train_full
)

# Stateful Preprocessing strictly fit on Training set
variance_selector = VarianceThreshold(threshold=1e-6)
variance_selector.fit(X_tr)

X_tr_filtered = variance_selector.transform(X_tr)
X_val_filtered = variance_selector.transform(X_val)
X_test_filtered = variance_selector.transform(X_test)

scaler = StandardScaler()
scaler.fit(X_tr_filtered)

X_tr_scaled = scaler.transform(X_tr_filtered)
X_val_scaled = scaler.transform(X_val_filtered)
X_test_scaled = scaler.transform(X_test_filtered)


# ---------------------------------------------------------
# 3. Model Architecture Design
# ---------------------------------------------------------
class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation channel gating module for dynamic feature recalibration.
    """

    def __init__(self, dim, reduction=4):
        super().__init__()
        reduced_dim = max(dim // reduction, 16)
        self.fc = nn.Sequential(
            nn.Linear(dim, reduced_dim),
            nn.SiLU(),
            nn.Linear(reduced_dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weights = self.fc(x)
        return x * weights


class ResidualBlock(nn.Module):
    """
    Residual block with Batch Normalization, SiLU non-linearities,
    Dropout regularization, and Squeeze-and-Excitation channel gating.
    """

    def __init__(self, dim, dropout_rate=0.2, reduction=4):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.se = SqueezeExcitation(dim, reduction=reduction)

    def forward(self, x):
        return x + self.se(self.block(x))


class KinematicResNet(nn.Module):
    """
    Kinematic Residual Network tailored for high-dimensional smartphone sensor classification.
    """

    def __init__(
        self, in_features, num_classes, hidden_dim=256, num_blocks=3, dropout_rate=0.2
    ):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        self.res_layers = nn.ModuleList(
            [
                ResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        self.classification_head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        h = self.input_projection(x)
        for block in self.res_layers:
            h = block(h)
        logits = self.classification_head(h)
        return logits


# ---------------------------------------------------------
# 4. Training and Evaluation Setup
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_tr_scaled.shape[1]
num_classes = len(np.unique(y_tr))

model = KinematicResNet(
    in_features=input_dim,
    num_classes=num_classes,
    hidden_dim=256,
    num_blocks=3,
    dropout_rate=0.2,
).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999))
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# Build PyTorch DataLoaders
batch_size = 64

train_tensor_x = torch.tensor(X_tr_scaled, dtype=torch.float32)
train_tensor_y = torch.tensor(y_tr, dtype=torch.long)
val_tensor_x = torch.tensor(X_val_scaled, dtype=torch.float32)
val_tensor_y = torch.tensor(y_val, dtype=torch.long)
test_tensor_x = torch.tensor(X_test_scaled, dtype=torch.float32)

train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
val_dataset = TensorDataset(val_tensor_x, val_tensor_y)
test_dataset = TensorDataset(test_tensor_x)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ---------------------------------------------------------
# 5. Model Training Loop with Mixup & Heterogeneous Ensembling
# ---------------------------------------------------------
def mixup_batch(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_sz = x.size(0)
    index = torch.randperm(batch_sz).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


num_epochs = 50
best_val_acc = 0.0
best_model_path = os.path.join(WORKING_DIR, "best_kinematic_resnet.pt")

for epoch in range(1, num_epochs + 1):
    model.train()
    running_train_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        if epoch <= num_epochs - 5:
            mixed_x, y_a, y_b, lam = mixup_batch(batch_x, batch_y, alpha=0.2)
            optimizer.zero_grad()
            outputs = model(mixed_x)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
        else:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * batch_x.size(0)
        _, preds = torch.max(outputs, 1)
        correct_train += (preds == batch_y).sum().item()
        total_train += batch_x.size(0)

    scheduler.step()

    train_loss = running_train_loss / total_train
    train_acc = correct_train / total_train

    # Validation step
    model.eval()
    val_loss = 0.0
    val_preds_list = []
    val_targets_list = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            val_loss += loss.item() * batch_x.size(0)
            _, preds = torch.max(outputs, 1)

            val_preds_list.extend(preds.cpu().numpy())
            val_targets_list.extend(batch_y.cpu().numpy())

    val_loss = val_loss / len(val_targets_list)
    val_acc = accuracy_score(val_targets_list, val_preds_list)

    # Save best model checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)

    print(
        f"Epoch {epoch:02d}/{num_epochs:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Best Val Acc: {best_val_acc:.4f}"
    )

# Load best ResNet checkpoint
model.load_state_dict(torch.load(best_model_path))
model.eval()

# Extract ResNet class probabilities
resnet_val_probs_list = []
with torch.no_grad():
    for batch_x, _ in val_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        probs = torch.softmax(outputs, dim=1)
        resnet_val_probs_list.append(probs.cpu().numpy())
resnet_val_probs = np.vstack(resnet_val_probs_list)

resnet_test_probs_list = []
with torch.no_grad():
    for (batch_x,) in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        probs = torch.softmax(outputs, dim=1)
        resnet_test_probs_list.append(probs.cpu().numpy())
resnet_test_probs = np.vstack(resnet_test_probs_list)

# Train LightGBM Classifier
print("Training LightGBM Classifier...")
lgbm = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
    n_jobs=-1,
)
lgbm.fit(X_tr_scaled, y_tr)
lgbm_val_probs = lgbm.predict_proba(X_val_scaled)
lgbm_test_probs = lgbm.predict_proba(X_test_scaled)
print(f"LGBM Validation Accuracy: {accuracy_score(y_val, np.argmax(lgbm_val_probs, axis=1)):.5f}")

# Train Calibrated Support Vector Classifier
print("Training Support Vector Classifier...")
svm = SVC(
    C=10.0,
    kernel="rbf",
    gamma="scale",
    probability=True,
    random_state=42,
)
svm.fit(X_tr_scaled, y_tr)
svm_val_probs = svm.predict_proba(X_val_scaled)
svm_test_probs = svm.predict_proba(X_test_scaled)
print(f"SVM Validation Accuracy: {accuracy_score(y_val, np.argmax(svm_val_probs, axis=1)):.5f}")

# ---------------------------------------------------------
# 6. Multi-Paradigm Ensemble Optimization & Submission
# ---------------------------------------------------------
best_weights = (0.34, 0.33, 0.33)
best_score = 0.0

weights_grid = []
for w1 in np.linspace(0, 1, 21):
    for w2 in np.linspace(0, 1 - w1, int(round((1 - w1) * 20)) + 1):
        w3 = 1.0 - w1 - w2
        if w3 >= -1e-5:
            weights_grid.append((w1, w2, max(0.0, w3)))

for w_res, w_lgb, w_svm in weights_grid:
    blended_val_probs = (
        w_res * resnet_val_probs + w_lgb * lgbm_val_probs + w_svm * svm_val_probs
    )
    val_preds = np.argmax(blended_val_probs, axis=1)
    acc = accuracy_score(y_val, val_preds)
    if acc > best_score:
        best_score = acc
        best_weights = (w_res, w_lgb, w_svm)

print(
    f"Optimal Ensemble Weights (ResNet, LGBM, SVM): {best_weights} with Val Acc: {best_score:.5f}"
)

# Compute final blended validation predictions
final_val_probs = (
    best_weights[0] * resnet_val_probs
    + best_weights[1] * lgbm_val_probs
    + best_weights[2] * svm_val_probs
)
final_val_preds = np.argmax(final_val_probs, axis=1)
score = accuracy_score(y_val, final_val_preds)

# Compute final blended test predictions
final_test_probs = (
    best_weights[0] * resnet_test_probs
    + best_weights[1] * lgbm_test_probs
    + best_weights[2] * svm_test_probs
)
test_preds = np.argmax(final_test_probs, axis=1)
test_labels = le.inverse_transform(test_preds)

# Save submission file
submission_df = pd.DataFrame(
    {"id": np.arange(len(test_labels), dtype=int), "class": test_labels}
)
submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df.to_csv(submission_path, index=False)

print(f"Final Validation Score: {score}")
