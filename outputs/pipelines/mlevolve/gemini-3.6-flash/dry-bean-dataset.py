import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set random seeds for deterministic execution
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# -------------------------------------------------------------------------
# 1. Load Raw Data
# -------------------------------------------------------------------------
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Class"
id_col = "id" if "id" in test_df.columns else None

y_raw = train_df[target_col]
X_train_raw = train_df.drop(
    columns=[target_col]
    + ([id_col] if id_col in train_df.columns and id_col in train_df else [])
)
X_test_raw = test_df.drop(
    columns=[id_col] if id_col and id_col in test_df.columns else []
)

if id_col and id_col in test_df.columns:
    test_ids = test_df[id_col].values
else:
    test_ids = np.arange(len(test_df))

# Encode target classes
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
classes = label_encoder.classes_
np.save("./working/classes.npy", classes)


# -------------------------------------------------------------------------
# 2. Geometric & Domain-Specific Feature Engineering Function
# -------------------------------------------------------------------------
def extract_geometric_features(df):
    data = df.copy()
    eps = 1e-8

    area = data["Area"]
    perim = data["Perimeter"]
    major = data["MajorAxisLength"]
    minor = data["MinorAxisLength"]
    c_area = data["ConvexArea"]
    eq_diam = data["EquivDiameter"]

    # Log transformations for spatial scales
    for col in [
        "Area",
        "Perimeter",
        "MajorAxisLength",
        "MinorAxisLength",
        "ConvexArea",
        "EquivDiameter",
    ]:
        data[f"log_{col}"] = np.log1p(data[col])

    # Geometrical Ratios & Perimeter Relationships
    data["Area_Perimeter_Ratio"] = area / (perim + eps)
    data["Perimeter_Sq_Area_Ratio"] = (perim**2) / (area + eps)
    data["Perimeter_Major_Ratio"] = perim / (major + eps)
    data["Perimeter_Minor_Ratio"] = perim / (minor + eps)

    # Ellipse and Axis Ratios
    data["Axis_Ratio"] = major / (minor + eps)
    data["Axis_Diff"] = major - minor
    data["Axis_Sum"] = major + minor
    data["Axis_Product"] = major * minor

    # Ideal Ellipse Calculations
    ellipse_area = np.pi * (major / 2.0) * (minor / 2.0)
    data["Ellipse_Area"] = ellipse_area
    data["Area_Ellipse_Ratio"] = area / (ellipse_area + eps)
    data["Ellipse_Area_Diff"] = ellipse_area - area

    # Ideal Circle EquivDiameter Residuals
    theo_equiv_diam = np.sqrt(4.0 * area / np.pi)
    data["EquivDiameter_Residual"] = eq_diam - theo_equiv_diam

    # Convexity & Solidity Relations
    data["Convex_Area_Diff"] = c_area - area
    data["Solidity_Calc"] = area / (c_area + eps)
    data["Solidity_Residual"] = data["Solidity"] - data["Solidity_Calc"]

    # Roundness & Compactness Re-calculations & Residuals
    theo_roundness = (4.0 * np.pi * area) / ((perim**2) + eps)
    data["Roundness_Residual"] = data["Roundness"] - theo_roundness

    theo_compactness = theo_equiv_diam / (major + eps)
    data["Compactness_Residual"] = data["Compactness"] - theo_compactness

    # Theoretical Shape Factor Derivations & Deviations
    sf1_calc = major / (area + eps)
    sf2_calc = minor / (area + eps)
    sf3_calc = area / ((major / 2.0) ** 2 * np.pi + eps)
    sf4_calc = area / (ellipse_area + eps)

    data["ShapeFactor1_Residual"] = data["ShapeFactor1"] - sf1_calc
    data["ShapeFactor2_Residual"] = data["ShapeFactor2"] - sf2_calc
    data["ShapeFactor3_Residual"] = data["ShapeFactor3"] - sf3_calc
    data["ShapeFactor4_Residual"] = data["ShapeFactor4"] - sf4_calc

    # Cross Interactions
    data["SF1_x_SF2"] = data["ShapeFactor1"] * data["ShapeFactor2"]
    data["SF3_x_SF4"] = data["ShapeFactor3"] * data["ShapeFactor4"]
    data["Extent_x_Solidity"] = data["Extent"] * data["Solidity"]
    data["Roundness_x_Compactness"] = data["Roundness"] * data["Compactness"]

    # Clean potential infs or nans
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(0.0)

    return data


X_train_fe = extract_geometric_features(X_train_raw)
X_test_fe = extract_geometric_features(X_test_raw)

# -------------------------------------------------------------------------
# 3. Stratified Train / Validation Split & Scaling
# -------------------------------------------------------------------------
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_fe, y_encoded, test_size=0.20, random_state=RANDOM_STATE, stratify=y_encoded
)

scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_fe)

joblib.dump(scaler, "./working/scaler.joblib")


# -------------------------------------------------------------------------
# 4. TabularResNet Model Definition
# -------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TabularResNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        logits = self.head(x)
        return logits


# Setup PyTorch Tensors and DataLoaders
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_tr_tensor = torch.tensor(X_tr_scaled, dtype=torch.float32)
y_tr_tensor = torch.tensor(y_tr, dtype=torch.long)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

batch_size = 128
train_dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

in_features = X_tr_scaled.shape[1]
num_classes = len(classes)

model = TabularResNet(
    in_features=in_features,
    num_classes=num_classes,
    hidden_dim=256,
    num_blocks=3,
    dropout_rate=0.2,
).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-5
)


# -------------------------------------------------------------------------
# 5. Training & Evaluation
# -------------------------------------------------------------------------
num_epochs = 100
best_val_acc = 0.0
best_model_path = "./working/best_model.pt"

for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        logits = model(bx)
        loss = criterion(logits, by)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(by)
        total_samples += len(by)

    scheduler.step()
    train_loss = running_loss / total_samples

    # Validation Phase
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            preds = torch.argmax(logits, dim=1)
            val_correct += (preds == by).sum().item()
            val_total += len(by)

    val_acc = val_correct / val_total

    print(
        f"Epoch {epoch:03d}/{num_epochs:03d} - Train Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)

# -------------------------------------------------------------------------
# 6. Final Validation Evaluation & Inference
# -------------------------------------------------------------------------
model.load_state_dict(torch.load(best_model_path))
model.eval()

val_preds = []
val_targets = []
with torch.no_grad():
    for bx, by in val_loader:
        bx = bx.to(device)
        logits = model(bx)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        val_preds.extend(preds)
        val_targets.extend(by.numpy())

val_preds = np.array(val_preds)
val_targets = np.array(val_targets)
final_val_score = float((val_preds == val_targets).mean())

# Test Set Inference
test_preds = []
with torch.no_grad():
    for (bx,) in test_loader:
        bx = bx.to(device)
        logits = model(bx)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        test_preds.extend(preds)

test_preds = np.array(test_preds)
test_class_labels = classes[test_preds]

# Save Submission File
submission_df = pd.DataFrame({"id": test_ids, "Class": test_class_labels})
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_val_score}")
