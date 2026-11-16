import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

# ============================================
# STEP 1: Load Data & Preprocessing
# ============================================
print("Loading data...")

train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Handle missing values
for col in train_df.columns:
    if train_df[col].isnull().any():
        train_df[col] = train_df[col].fillna(train_df[col].median())

# Handle outliers in sensor features
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    mean = train_df[col].mean()
    std = train_df[col].std()
    if std > 0:
        train_df[col] = np.clip(train_df[col], mean - 3 * std, mean + 3 * std)

# No noise injection needed - data is clean and preprocessing is sufficient

# ============================================
# STEP 2: Feature Engineering & Scaling
# ============================================
print("Applying feature engineering and scaling...")

# Extract feature columns (exclude id and target)
feature_cols = [col for col in train_df.columns if col not in ["id", "class"]]
target_col = "class"

# Create feature matrix and target
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

# ============================================
# STEP 3: Create Train/Validation Split FIRST
# ============================================
print("Creating train/validation split...")

# Stratified split for multi-class problem (6 activities)
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

# Split validation into 80/20 for final validation
X_val_split, X_test_split, y_val_split, y_test_split = train_test_split(
    X_val_split, y_val_split, test_size=0.25, stratify=y_val_split, random_state=42
)

# Encode class labels to integers (required for model input)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_train_split_encoded = label_encoder.fit_transform(y_train_split)
y_val_split_encoded = label_encoder.transform(y_val_split)
y_test_split_encoded = label_encoder.transform(y_test_split)

# ============================================
# STEP 2: Feature Engineering & Scaling
# ============================================
print("Applying feature engineering and scaling...")

# Fit StandardScaler on training data only (NO leakage!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_split)
X_val_scaled = scaler.transform(X_val_split)
X_test_scaled = scaler.transform(X_test_split)

# ============================================
# STEP 4: Create Data Loaders
# ============================================
print("Creating data loaders...")

train_dataset = TensorDataset(
    torch.tensor(X_train_scaled, dtype=torch.float32),
    torch.tensor(y_train_split_encoded, dtype=torch.long),
)
val_dataset = TensorDataset(
    torch.tensor(X_val_scaled, dtype=torch.float32),
    torch.tensor(y_val_split_encoded, dtype=torch.long),
)
test_dataset = TensorDataset(
    torch.tensor(X_test_scaled, dtype=torch.float32),
    torch.tensor(y_test_split_encoded, dtype=torch.long),
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)


# ============================================
# STEP 5: Define Model Architecture
# ============================================
class HARClassifier(nn.Module):
    """Multi-Layer Perceptron for Human Activity Recognition"""

    def __init__(self, input_dim=561, num_classes=6, dropout_rate=0.3):
        super(HARClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.features(x)


# ============================================
# STEP 6: Initialize Training Components
# ============================================
model = HARClassifier(input_dim=561, num_classes=6, dropout_rate=0.3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5
)

# ============================================
# STEP 7: Training Loop with Early Stopping
# ============================================
best_val_acc = 0
best_model_state = None
patience_counter = 0
max_patience = 15
epochs = 50

for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * data.size(0)
        _, predicted = output.max(1)
        train_total += target.size(0)
        train_correct += predicted.eq(target).sum().item()

    train_acc = train_correct / train_total

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            loss = criterion(output, target)

            val_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            val_total += target.size(0)
            val_correct += predicted.eq(target).sum().item()

    val_acc = val_correct / val_total
    val_loss /= val_total
    train_loss /= len(train_loader.dataset)

    # Log metrics (1 line per epoch)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(
            f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    # Update scheduler
    scheduler.step(val_acc)

    # Track best accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    # Early stopping
    if patience_counter >= max_patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# ============================================
# STEP 8: Load Best Model
# ============================================
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"Best model saved with validation accuracy: {best_val_acc:.4f}")
else:
    print("No best model found, using final model")

# ============================================
# STEP 9: Generate Test Predictions
# ============================================
print("Generating test predictions...")

model.eval()
test_predictions = []
test_ids = []

with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = output.max(1)
        test_predictions.extend(predicted.cpu().numpy())
        test_ids.extend(range(len(predicted)))

# Create submission dataframe
submission = pd.DataFrame({"id": test_ids, "class": test_predictions})

# ============================================
# STEP 10: Save Submission
# ============================================
os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission.csv", index=False)
print(
    f"Submission saved to ./submission/submission.csv with {len(submission)} predictions"
)

# ============================================
# STEP 11: Final Validation Score
# ============================================
print(f"Final Validation Score: {best_val_acc}")