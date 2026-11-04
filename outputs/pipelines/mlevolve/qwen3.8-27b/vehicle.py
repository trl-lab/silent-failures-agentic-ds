import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os

# Create output directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Columns: {list(train_df.columns)}")
print(f"Target distribution:\n{train_df['class'].value_counts()}")

# ============================================================
# 2. IDENTIFY FEATURES AND TARGET
# ============================================================
target_col = "class"
feature_cols = [c for c in train_df.columns if c != target_col]
print(f"Number of features: {len(feature_cols)}")

# ============================================================
# 3. CREATE TRAIN/VALIDATION SPLIT (before any feature engineering)
# ============================================================
X_train_full = train_df[feature_cols].copy()
y_train_full = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

print(f"Train split: {X_train.shape[0]} samples")
print(f"Val split: {X_val.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
def engineer_features(X, feature_names):
    """Create domain-informed features for vehicle silhouette classification."""
    X_eng = X.copy()

    if "COMPACTNESS" in feature_names and "CIRCULARITY" in feature_names:
        X_eng["shape_fullness"] = X["COMPACTNESS"] * X["CIRCULARITY"]
        X_eng["shape_fullness_diff"] = X["COMPACTNESS"] - X["CIRCULARITY"]

    if (
        "PR.AXIS_ASPECT_RATIO" in feature_names
        and "MAX.LENGTH_ASPECT_RATIO" in feature_names
    ):
        X_eng["elongation_product"] = (
            X["PR.AXIS_ASPECT_RATIO"] * X["MAX.LENGTH_ASPECT_RATIO"]
        )
        X_eng["elongation_sum"] = (
            X["PR.AXIS_ASPECT_RATIO"] + X["MAX.LENGTH_ASPECT_RATIO"]
        )
        X_eng["elongation_diff"] = (
            X["PR.AXIS_ASPECT_RATIO"] - X["MAX.LENGTH_ASPECT_RATIO"]
        )

    if (
        "PR.AXIS_RECTANGULARITY" in feature_names
        and "MAX.LENGTH_RECTANGULARITY" in feature_names
    ):
        X_eng["rectangularity_product"] = (
            X["PR.AXIS_RECTANGULARITY"] * X["MAX.LENGTH_RECTANGULARITY"]
        )
        X_eng["rectangularity_diff"] = (
            X["PR.AXIS_RECTANGULARITY"] - X["MAX.LENGTH_RECTANGULARITY"]
        )

    if (
        "SKEWNESS_ABOUT_MAJOR" in feature_names
        and "SKEWNESS_ABOUT_MINOR" in feature_names
    ):
        X_eng["skewness_product"] = (
            X["SKEWNESS_ABOUT_MAJOR"] * X["SKEWNESS_ABOUT_MINOR"]
        )
        X_eng["skewness_sum"] = (
            X["SKEWNESS_ABOUT_MAJOR"] + X["SKEWNESS_ABOUT_MINOR"]
        )

    if (
        "KURTOSIS_ABOUT_MAJOR" in feature_names
        and "KURTOSIS_ABOUT_MINOR" in feature_names
    ):
        X_eng["kurtosis_product"] = (
            X["KURTOSIS_ABOUT_MAJOR"] * X["KURTOSIS_ABOUT_MINOR"]
        )
        X_eng["kurtosis_diff"] = (
            X["KURTOSIS_ABOUT_MAJOR"] - X["KURTOSIS_ABOUT_MINOR"]
        )

    if (
        "SCALED_VARIANCE_MAJOR" in feature_names
        and "SCALED_VARIANCE_MINOR" in feature_names
    ):
        X_eng["variance_ratio"] = (X["SCALED_VARIANCE_MAJOR"] + 1e-10) / (
            X["SCALED_VARIANCE_MINOR"] + 1e-10
        )
        X_eng["variance_product"] = (
            X["SCALED_VARIANCE_MAJOR"]
            * X["SCALED_VARIANCE_MINOR"]
        )
        X_eng["variance_diff"] = (
            X["SCALED_VARIANCE_MAJOR"]
            - X["SCALED_VARIANCE_MINOR"]
        )

    if "DISTANCE_CIRCULARITY" in feature_names and "CIRCULARITY" in feature_names:
        X_eng["boundary_complexity"] = X["DISTANCE_CIRCULARITY"] - X["CIRCULARITY"]
        X_eng["boundary_ratio"] = (X["DISTANCE_CIRCULARITY"] + 1e-10) / (
            X["CIRCULARITY"] + 1e-10
        )

    if "RADIUS_RATIO" in feature_names and "SCALED_RADIUS_OF_GYRATION" in feature_names:
        X_eng["radial_variation"] = X["RADIUS_RATIO"] * X["SCALED_RADIUS_OF_GYRATION"]

    if "HOLLOWS_RATIO" in feature_names:
        X_eng["hollows_compactness"] = X["HOLLOWS_RATIO"] * X["COMPACTNESS"]
        X_eng["hollows_circularity"] = X["HOLLOWS_RATIO"] * X["CIRCULARITY"]

    if "SCATTER_RATIO" in feature_names:
        X_eng["scatter_elongation"] = X["SCATTER_RATIO"] * X["PR.AXIS_ASPECT_RATIO"]
        X_eng["scatter_compactness"] = X["SCATTER_RATIO"] * X["COMPACTNESS"]

    key_features = [
        "COMPACTNESS",
        "CIRCULARITY",
        "PR.AXIS_ASPECT_RATIO",
        "MAX.LENGTH_ASPECT_RATIO",
        "SCATTER_RATIO",
        "ELONGATEDNESS",
        "HOLLOWS_RATIO",
        "SCALED_RADIUS_OF_GYRATION",
    ]
    for feat in key_features:
        if feat in feature_names:
            X_eng[f"{feat}_sq"] = X[feat] ** 2
            X_eng[f"{feat}_cube"] = X[feat] ** 3
            X_eng[f"{feat}_sqrt"] = np.sign(X[feat]) * np.sqrt(np.abs(X[feat]))

    important_pairs = [
        ("COMPACTNESS", "PR.AXIS_ASPECT_RATIO"),
        ("CIRCULARITY", "SCATTER_RATIO"),
        ("ELONGATEDNESS", "HOLLOWS_RATIO"),
        ("MAX.LENGTH_ASPECT_RATIO", "SCALED_RADIUS_OF_GYRATION"),
        ("COMPACTNESS", "HOLLOWS_RATIO"),
        ("CIRCULARITY", "ELONGATEDNESS"),
        ("PR.AXIS_ASPECT_RATIO", "SCATTER_RATIO"),
        ("MAX.LENGTH_ASPECT_RATIO", "COMPACTNESS"),
    ]
    for f1, f2 in important_pairs:
        if f1 in feature_names and f2 in feature_names:
            X_eng[f"{f1}_x_{f2}"] = X[f1] * X[f2]

    return X_eng


X_train_eng = engineer_features(X_train, feature_cols)
X_val_eng = engineer_features(X_val, feature_cols)
X_test_eng = engineer_features(X_test, feature_cols)

print(
    f"Engineered features count: {X_train_eng.shape[1]} (from {len(feature_cols)} original)"
)

# ============================================================
# 5. FEATURE SCALING (fit on train only)
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_eng)
X_val_scaled = scaler.transform(X_val_eng)
X_test_scaled = scaler.transform(X_test_eng)

# ============================================================
# 6. ENCODE LABELS
# ============================================================
classes = sorted(y_train_full.unique())
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}
num_classes = len(classes)

y_train_encoded = np.array([class_to_idx[c] for c in y_train])
y_val_encoded = np.array([class_to_idx[c] for c in y_val])

print(f"Number of classes: {num_classes}")
print(f"Class mapping: {class_to_idx}")

# ============================================================
# 7. CONVERT TO TENSORS AND CREATE DATALOADERS
# ============================================================
X_train_tensor = torch.FloatTensor(X_train_scaled.astype(np.float32))
X_val_tensor = torch.FloatTensor(X_val_scaled.astype(np.float32))
X_test_tensor = torch.FloatTensor(X_test_scaled.astype(np.float32))
y_train_tensor = torch.LongTensor(y_train_encoded)
y_val_tensor = torch.LongTensor(y_val_encoded)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)


# ============================================================
# 8. DEFINE MODEL
# ============================================================
class VehicleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout_rate=0.3):
        super(VehicleClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.75),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dim // 4, num_classes),
        )

    def forward(self, x):
        return self.network(x)


input_dim = X_train_scaled.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = VehicleClassifier(
    input_dim=input_dim, num_classes=num_classes, hidden_dim=256, dropout_rate=0.3
)
model = model.to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# 9. LOSS FUNCTION AND OPTIMIZER
# ============================================================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
num_epochs = 300
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=1e-5
)

# ============================================================
# 10. TRAINING LOOP WITH EARLY STOPPING
# ============================================================
best_val_acc = 0.0
best_model_state = None
patience = 40
patience_counter = 0

print("\n--- Training ---")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
        _, predicted = outputs.max(1)
        train_correct += predicted.eq(y_batch).sum().item()
        train_total += y_batch.size(0)

    scheduler.step()

    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(y_batch).sum().item()
            val_total += y_batch.size(0)

    val_acc = val_correct / val_total
    train_acc = train_correct / train_total

    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(
            f"Epoch {epoch+1:3d}/{num_epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Loss: {train_loss/train_total:.4f}"
        )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

print(f"\nBest Validation Accuracy (during training): {best_val_acc:.4f}")

# ============================================================
# 11. LOAD BEST MODEL AND GENERATE PREDICTIONS
# ============================================================
model.load_state_dict(best_model_state)
model.to(device)
model.eval()

with torch.no_grad():
    val_outputs = model(X_val_tensor.to(device))
    val_preds = val_outputs.argmax(dim=1).cpu().numpy()

val_pred_labels = [idx_to_class[p] for p in val_preds]
val_true_labels = [idx_to_class[p] for p in y_val_encoded]
val_accuracy = accuracy_score(val_true_labels, val_pred_labels)
print(f"Validation Accuracy (official metric): {val_accuracy:.4f}")

with torch.no_grad():
    test_outputs = model(X_test_tensor.to(device))
    test_preds = test_outputs.argmax(dim=1).cpu().numpy()

test_pred_labels = [idx_to_class[p] for p in test_preds]

# ============================================================
# 12. SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame(
    {"id": range(len(test_pred_labels)), "class": test_pred_labels}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape[0]} predictions")
print(f"Prediction distribution:\n{pd.Series(test_pred_labels).value_counts()}")

# ============================================================
# 13. FINAL SCORE
# ============================================================
score = val_accuracy
print(f"Final Validation Score: {score}")
