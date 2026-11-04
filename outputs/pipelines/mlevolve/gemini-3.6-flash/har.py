import copy
import json
import os
import numpy as np
import pandas as pd

import lightgbm as lgb
import xgboost as xgb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Ensure necessary directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ==========================================
# 1. DATA LOADING & FEATURE ENGINEERING
# ==========================================
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

print(f"Loading raw datasets from {train_path} and {test_path}...")
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "class"
ignore_cols = [target_col, "id", "ID", "Id"]

if "id" in test_df.columns:
    test_ids = test_df["id"].values
else:
    test_ids = np.arange(len(test_df))

feature_cols = [c for c in train_df.columns if c not in ignore_cols]
print(f"Raw train shape: {train_df.shape}, Raw test shape: {test_df.shape}")
print(f"Number of input signal features: {len(feature_cols)}")


def engineer_har_features(df, cols):
    X_raw = df[cols].copy()
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
    arr = X_raw.values

    # Global signal summary statistics across feature dimensions
    row_mean = np.nanmean(arr, axis=1)
    row_std = np.nanstd(arr, axis=1)
    row_max = np.nanmax(arr, axis=1)
    row_min = np.nanmin(arr, axis=1)
    row_median = np.nanmedian(arr, axis=1)
    row_l1 = np.nansum(np.abs(arr), axis=1)
    row_l2 = np.sqrt(np.nansum(arr**2, axis=1))

    q75 = np.nanpercentile(arr, 75, axis=1)
    q25 = np.nanpercentile(arr, 25, axis=1)
    row_iqr = q75 - q25

    engineered_df = pd.DataFrame(
        {
            "row_mean": row_mean,
            "row_std": row_std,
            "row_max": row_max,
            "row_min": row_min,
            "row_median": row_median,
            "row_l1": row_l1,
            "row_l2": row_l2,
            "row_iqr": row_iqr,
            "row_peak_to_std": (row_max - row_min) / (row_std + 1e-6),
        },
        index=df.index,
    )

    # Triaxial pattern matching (-X, -Y, -Z axes)
    triplet_features = {}
    for col in cols:
        for sep in ["-X", ".X", "_X"]:
            if col.endswith(sep):
                base = col[: -len(sep)]
                col_y = base + sep.replace("X", "Y")
                col_z = base + sep.replace("X", "Z")
                if col_y in cols and col_z in cols:
                    triplet_features[base] = (col, col_y, col_z)
                break

    if len(triplet_features) > 0:
        for base, (cx, cy, cz) in triplet_features.items():
            x = X_raw[cx].values
            y = X_raw[cy].values
            z = X_raw[cz].values

            engineered_df[f"{base}_mag"] = np.sqrt(x**2 + y**2 + z**2)
            engineered_df[f"{base}_xy_diff"] = x - y
            engineered_df[f"{base}_yz_diff"] = y - z
            engineered_df[f"{base}_zx_diff"] = z - x
            engineered_df[f"{base}_xy_prod"] = x * y
            engineered_df[f"{base}_yz_prod"] = y * z
            engineered_df[f"{base}_zx_prod"] = z * x
    else:
        for i in range(0, len(cols) - 2, 3):
            cx, cy, cz = cols[i], cols[i + 1], cols[i + 2]
            x = X_raw[cx].values
            y = X_raw[cy].values
            z = X_raw[cz].values

            base = f"grp_{i//3}"
            engineered_df[f"{base}_mag"] = np.sqrt(x**2 + y**2 + z**2)
            engineered_df[f"{base}_xy_diff"] = x - y
            engineered_df[f"{base}_yz_diff"] = y - z
            engineered_df[f"{base}_zx_diff"] = z - x

    return pd.concat([X_raw, engineered_df], axis=1)


print("Engineering signal interaction features...")
X_full = engineer_har_features(train_df, feature_cols)
X_test_raw = engineer_har_features(test_df, feature_cols)

# Target encoding
y_full_raw = train_df[target_col].values
label_encoder = LabelEncoder()
y_full = label_encoder.fit_transform(y_full_raw)

# ==========================================
# 2. STRATIFIED SPLIT & LEAK-FREE PREPROCESSING
# ==========================================
X_train_df, X_val_df, y_train, y_val = train_test_split(
    X_full, y_full, test_size=0.20, random_state=42, stratify=y_full
)

print(f"Train split samples: {len(X_train_df)}, Val split samples: {len(X_val_df)}")

# Fit preprocessors strictly on train split
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train_df)
X_val_imp = imputer.transform(X_val_df)
X_test_imp = imputer.transform(X_test_raw)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_val_scaled = scaler.transform(X_val_imp)
X_test_scaled = scaler.transform(X_test_imp)

feature_names = [f"feat_{i}" for i in range(X_train_scaled.shape[1])]
X_train = pd.DataFrame(X_train_scaled, columns=feature_names)
X_val = pd.DataFrame(X_val_scaled, columns=feature_names)
X_test = pd.DataFrame(X_test_scaled, columns=feature_names)

in_features = X_train.shape[1]
num_classes = len(label_encoder.classes_)


# ==========================================
# 3. MODEL ARCHITECTURE DEFINITIONS
# ==========================================
class FeatureAttention(nn.Module):
    """Squeeze-and-Excitation style feature attention for tabular sensor features."""

    def __init__(self, in_features, reduction=4):
        super().__init__()
        reduced_dim = max(8, in_features // reduction)
        self.fc = nn.Sequential(
            nn.Linear(in_features, reduced_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, in_features, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weights = self.fc(x)
        return x * weights


class ResBlock(nn.Module):
    """Residual block with LayerNorm, Dropout, and Feature Attention."""

    def __init__(self, dim, dropout_rate=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            FeatureAttention(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class TabularResNet(nn.Module):
    """Deep Tabular ResNet architecture with feature gating."""

    def __init__(
        self,
        in_features,
        num_classes=6,
        hidden_dim=256,
        num_blocks=2,
        dropout_rate=0.25,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.blocks = nn.ModuleList(
            [ResBlock(hidden_dim, dropout_rate=dropout_rate) for _ in range(num_blocks)]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def get_lgbm_params(num_classes, seed=42):
    return {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "min_child_samples": 20,
        "random_state": seed,
        "n_estimators": 1000,
        "verbose": -1,
        "n_jobs": -1,
    }


def get_xgb_params(num_classes, seed=42):
    return {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "learning_rate": 0.03,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_weight": 3,
        "random_state": seed,
        "n_estimators": 1000,
        "tree_method": "hist",
        "n_jobs": -1,
    }


# ==========================================
# 4. MODEL TRAINING & VALIDATION
# ==========================================
# Model 1: LightGBM
print("Training LightGBM model...")
lgb_params = get_lgbm_params(num_classes=num_classes, seed=42)
lgb_model = lgb.LGBMClassifier(**lgb_params)
lgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)

val_probs_lgb = lgb_model.predict_proba(X_val)
test_probs_lgb = lgb_model.predict_proba(X_test)
lgb_val_acc = accuracy_score(y_val, np.argmax(val_probs_lgb, axis=1))
print(f"LightGBM Validation Accuracy: {lgb_val_acc:.4f}")

# Model 2: XGBoost
print("Training XGBoost model...")
xgb_params = get_xgb_params(num_classes=num_classes, seed=42)
xgb_model = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=50)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

val_probs_xgb = xgb_model.predict_proba(X_val)
test_probs_xgb = xgb_model.predict_proba(X_test)
xgb_val_acc = accuracy_score(y_val, np.argmax(val_probs_xgb, axis=1))
print(f"XGBoost Validation Accuracy: {xgb_val_acc:.4f}")

# Model 3: PyTorch TabularResNet
print("Training Deep PyTorch TabularResNet...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = TensorDataset(
    torch.tensor(X_train.values, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long),
)
val_dataset = TensorDataset(
    torch.tensor(X_val.values, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.long),
)
test_dataset = TensorDataset(torch.tensor(X_test.values, dtype=torch.float32))

train_loader = DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True
)

nn_model = TabularResNet(
    in_features=in_features,
    num_classes=num_classes,
    hidden_dim=256,
    num_blocks=2,
    dropout_rate=0.25,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)

epochs = 80
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=1e-5
)

best_val_loss = float("inf")
best_weights = copy.deepcopy(nn_model.state_dict())
patience = 15
patience_counter = 0

for epoch in range(1, epochs + 1):
    nn_model.train()
    running_loss = 0.0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        logits = nn_model(bx)
        loss = criterion(logits, by)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * bx.size(0)

    scheduler.step()
    train_loss = running_loss / len(train_dataset)

    nn_model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            logits = nn_model(bx)
            loss = criterion(logits, by)
            val_loss += loss.item() * bx.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == by).sum().item()

    val_loss /= len(val_dataset)
    val_acc = correct / len(val_dataset)

    print(
        f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = copy.deepcopy(nn_model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

nn_model.load_state_dict(best_weights)

# Compute PyTorch Validation & Test Probabilities
nn_model.eval()
val_preds_nn = []
with torch.no_grad():
    for bx, _ in val_loader:
        bx = bx.to(device)
        logits = nn_model(bx)
        probs = F.softmax(logits, dim=1)
        val_preds_nn.append(probs.cpu().numpy())
val_probs_nn = np.vstack(val_preds_nn)

test_preds_nn = []
with torch.no_grad():
    for (bx,) in test_loader:
        bx = bx.to(device)
        logits = nn_model(bx)
        probs = F.softmax(logits, dim=1)
        test_preds_nn.append(probs.cpu().numpy())
test_probs_nn = np.vstack(test_preds_nn)

nn_val_acc = accuracy_score(y_val, np.argmax(val_probs_nn, axis=1))
print(f"PyTorch TabularResNet Validation Accuracy: {nn_val_acc:.4f}")

# ==========================================
# 5. ENSEMBLE BLENDING & SUBMISSION GENERATION
# ==========================================
w_lgb, w_xgb, w_nn = 0.35, 0.35, 0.30

val_probs_ensemble = w_lgb * val_probs_lgb + w_xgb * val_probs_xgb + w_nn * val_probs_nn
val_preds_ensemble = np.argmax(val_probs_ensemble, axis=1)

final_val_accuracy = accuracy_score(y_val, val_preds_ensemble)

# Test Inference & Prediction Mapping
test_probs_ensemble = (
    w_lgb * test_probs_lgb + w_xgb * test_probs_xgb + w_nn * test_probs_nn
)
test_preds_idx = np.argmax(test_probs_ensemble, axis=1)
predicted_labels = label_encoder.inverse_transform(test_preds_idx)

submission_df = pd.DataFrame({"id": test_ids, "class": predicted_labels})
sub_path = "./submission/submission.csv"
submission_df.to_csv(sub_path, index=False)

assert os.path.exists(sub_path), "Submission file generation failed!"
assert len(submission_df) == len(X_test), "Submission row count mismatch!"

print(
    f"Submission saved successfully to {sub_path}. Total predictions: {len(submission_df)}"
)
print(f"Final Validation Score: {final_val_accuracy}")
